"""LAPE (3) normal-consistent evidence (GoMVS's geometrically consistent propagation).

A weakly textured pixel p has no matching evidence of its own, but its neighbours on
the same slanted surface may. With the surface normal n at p (from the RAC-aligned
monocular depth), the point of p's m-th candidate and the matching point on the ray of
a neighbour q lie on one plane:

    r_qp = (n . K^-1 p~) / (n . K^-1 q~),     d_q = r_qp * d_p^(m)

so q's matching cost at d_q is evidence for p's candidate m. GoMVS (CVPR 2024) builds
its whole 3D regulariser from such propagated costs; here one layer is added in front
of the unchanged 3D UNet:

    C*(p, m) = C(p, m) + g(p) * mean_q [ pass(p,q) W_{q-p} C~_{q->p}(m) ]

* ``C~`` reads q's z-normalised cost (per group, along depth) at u(d_q) and at
  u(d_q) +- eps on q's *own* candidate axis (stage 2/3 axes differ per pixel), and
  takes their log-sum-exp — the "look a little before and behind" tolerance for
  monocular normal error; eps = (0.5 + kappa * |q - p|) * spacing, kappa learnable.
  Values that fall outside q's window are not evidence and are masked.
* ``W`` (one G x G matrix per neighbour offset, = GoMVS's 1x1xk aggregation conv) is
  zero-initialised: at step 0 the layer adds exactly nothing.
* ``g = c_n * rho``: normal reliability (no DA3 edge nearby, not grazing, normal
  exists) times the fraction of neighbours that produced evidence.

The loop over neighbours accumulates instead of stacking, and in training every
neighbour's term is its own non-reentrant checkpoint: the backward recomputes one
offset at a time, so the layer's peak activation is a few cost volumes, not 24 x 3 of
them (the three tolerance reads share one gather on a 3D-long index).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def camera_normals(z: torch.Tensor, valid: torch.Tensor, edge: torch.Tensor, K: torch.Tensor,
                   tau_edge: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z [B,1,H,W] metric depth, valid bool, edge [B,1,H,W] (log-depth edge at this
    resolution), K [B,3,3] at this resolution. Returns (n [B,3,H,W] unit, toward the
    camera; ok [B,1,H,W] bool; |cos| between n and the viewing ray [B,1,H,W])."""
    B, _, H, W = z.shape
    dev = z.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], 0).view(1, 3, -1)
    rays = torch.linalg.inv(K.float()) @ pix                          # [B,3,HW]
    X = (rays * z.float().view(B, 1, -1)).view(B, 3, H, W)
    ok = valid.bool() & (edge.float() < tau_edge)

    def diff(axis: int):
        if axis == 3:
            fwd = F.pad(X[..., 1:] - X[..., :-1], (0, 1))
            vf = F.pad((ok[..., 1:] & ok[..., :-1]).float(), (0, 1)) > 0
        else:
            fwd = F.pad(X[..., 1:, :] - X[..., :-1, :], (0, 0, 0, 1))
            vf = F.pad((ok[..., 1:, :] & ok[..., :-1, :]).float(), (0, 0, 0, 1)) > 0
        bwd = torch.roll(fwd, 1, dims=axis)
        vb = torch.roll(vf, 1, dims=axis)
        if axis == 3:
            vb[..., 0] = False
        else:
            vb[..., 0, :] = False
        both = vf & vb
        g = torch.where(both, 0.5 * (fwd + bwd), torch.where(vf, fwd, bwd))
        return g, vf | vb

    gx, okx = diff(3)
    gy, oky = diff(2)
    n = torch.cross(gx, gy, dim=1)
    nn_ = n.norm(dim=1, keepdim=True)
    good = ok & okx & oky & (nn_ > 1e-12)
    n = n / nn_.clamp_min(1e-12)
    flip = (n * X).sum(dim=1, keepdim=True) > 0
    n = torch.where(flip, -n, n)
    rv = rays.view(B, 3, H, W)
    cosv = ((n * rv).sum(1, keepdim=True) / rv.norm(dim=1, keepdim=True).clamp_min(1e-12)).abs()
    return n * good.float(), good, cosv


def normalize_cost(cv: torch.Tensor) -> torch.Tensor:
    cv = cv.float()
    mu = cv.mean(dim=2, keepdim=True)
    sd = cv.std(dim=2, keepdim=True, unbiased=False)
    return (cv - mu) / (sd + 1e-4)


class NormalEvidence(nn.Module):
    def __init__(self, num_groups: int, radius: int = 2, dilation: int = 1, kappa_init: float = 0.25) -> None:
        super().__init__()
        self.G = int(num_groups)
        self.dilation = int(dilation)
        self.offsets = [(dy * self.dilation, dx * self.dilation)
                        for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
                        if (dy, dx) != (0, 0)]
        self.reach = radius * self.dilation
        self.W = nn.Parameter(torch.zeros(len(self.offsets), self.G, self.G))
        self.log_kappa = nn.Parameter(torch.tensor(math.log(kappa_init)))
        self.register_buffer("taps", torch.tensor([-1.0, 0.0, 1.0]), persistent=False)

    def _term(self, k: int, Cp: torch.Tensor, utq: torch.Tensor, duq: torch.Tensor, a_p: torch.Tensor,
              cx: torch.Tensor, cy: torch.Tensor, inv_p: torch.Tensor, du: torch.Tensor,
              pass_k: torch.Tensor, vmin: torch.Tensor, span: torch.Tensor):
        """Neighbour k's weighted evidence [B,G,D,H,W] and its mask [B,1,D,H,W]."""
        B, G, D = Cp.shape[:3]
        H, Wd = inv_p.shape[-2:]
        dy, dx = self.offsets[k]
        R = self.reach
        src = Cp[..., R + dy:R + dy + H, R + dx:R + dx + Wd]           # q's normalised cost
        denom = a_p + dx * cx + dy * cy
        r = a_p / torch.where(denom.abs() > 1e-9, denom, torch.full_like(denom, 1e-9))
        ok_r = (a_p * denom > 0) & (r > 0.5) & (r < 2.0) & (pass_k > 0.5)
        inv_q = inv_p / r.clamp(0.5, 2.0)
        u_q = (inv_q - vmin) / span                                     # [B,D,H,W]
        f = (utq - u_q) / duq                                           # fractional index on q's axis
        eps = (0.5 + self.log_kappa.exp() * math.hypot(dy, dx)) * du / duq
        ft = f.unsqueeze(1) + self.taps.view(1, 3, 1, 1, 1) * eps.unsqueeze(1)   # [B,3,D,H,W]
        inside = (ft >= 0.0) & (ft <= D - 1)
        fc = ft.clamp(0.0, D - 1 - 1e-4)
        i0 = fc.floor()
        wgt = (fc - i0).reshape(B, 1, 3 * D, H, Wd)
        i0 = i0.long().reshape(B, 1, 3 * D, H, Wd)
        i1 = (i0 + 1).clamp(max=D - 1)
        full = (B, G, 3 * D, H, Wd)
        v = torch.lerp(src.gather(2, i0.expand(full)), src.gather(2, i1.expand(full)), wgt.expand(full))
        msk = inside.unsqueeze(1).float()                               # [B,1,3,D,H,W]
        nval = msk.sum(2)
        lse = torch.logsumexp(v.view(B, G, 3, D, H, Wd) + torch.log(msk.clamp_min(1e-12)), dim=2) \
            - torch.log(nval.clamp_min(1.0))
        m = ok_r.unsqueeze(1).float() * (nval > 0).float()              # [B,1,D,H,W]
        return torch.einsum("ij,bjdhw->bidhw", self.W[k], lse * m), m

    def forward(self, cv: torch.Tensor, u_hyp: torch.Tensor, n: torch.Tensor, c_n: torch.Tensor,
                K: torch.Tensor, pass_mask: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor):
        """cv [B,G,D,H,W] (current stage's group correlation), u_hyp [B,D,H,W] descending
        and uniform per pixel, n [B,3,H,W], c_n [B,1,H,W], K [B,3,3] at this resolution,
        pass_mask [B,len(offsets),H,W] same-surface masks on ``self.offsets``.

        Returns (delta [B,G,D,H,W] to add to cv, gate [B,1,H,W])."""
        B, G, D, H, Wd = cv.shape
        R = self.reach
        Cp = F.pad(normalize_cost(cv), (R, R, R, R))
        u = u_hyp.float()
        du = (u[:, :1] - u[:, 1:2]).abs().clamp_min(1e-8)              # [B,1,H,W]
        span = (vmax - vmin).float()
        vmin = vmin.float()
        inv_p = vmin + u * span                                         # [B,D,H,W] 1/d of p's candidates
        Kinv = torch.linalg.inv(K.float())                             # [B,3,3]
        ys, xs = torch.meshgrid(torch.arange(H, device=cv.device, dtype=torch.float32),
                                torch.arange(Wd, device=cv.device, dtype=torch.float32), indexing="ij")
        pix = torch.stack([xs, ys, torch.ones_like(xs)], 0).view(1, 3, -1)
        ray = (Kinv @ pix).view(B, 3, H, Wd)
        a_p = (n * ray).sum(1, keepdim=True)                            # n . K^-1 p~
        cx = (n * Kinv[:, :, 0].view(B, 3, 1, 1)).sum(1, keepdim=True)   # n . K^-1 e_x
        cy = (n * Kinv[:, :, 1].view(B, 3, 1, 1)).sum(1, keepdim=True)
        utp = F.pad(u[:, :1], (R, R, R, R))
        dup = F.pad(du, (R, R, R, R), value=1.0)
        ckpt = self.training and torch.is_grad_enabled()
        acc = torch.zeros(B, G, D, H, Wd, device=cv.device)
        cnt = torch.zeros(B, 1, D, H, Wd, device=cv.device)
        for k, (dy, dx) in enumerate(self.offsets):
            sl = (Ellipsis, slice(R + dy, R + dy + H), slice(R + dx, R + dx + Wd))
            args = (Cp, utp[sl], dup[sl], a_p, cx, cy, inv_p, du, pass_mask[:, k:k + 1], vmin, span)
            if ckpt:
                term, m = checkpoint(self._term, k, *args, use_reentrant=False)
            else:
                term, m = self._term(k, *args)
            acc = acc + term
            cnt = cnt + m
        out = acc / cnt.clamp_min(1.0)
        rho = cnt.mean(dim=2) / float(len(self.offsets))                  # [B,1,H,W]
        gate = c_n.float() * rho
        return out * gate.unsqueeze(2), gate
