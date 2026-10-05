from __future__ import annotations

import torch
import torch.nn.functional as F

from base.config import LossConfig, StageWeights

from .depth_loss import normalized_huber_loss, soft_label_cross_entropy


class MVSLoss:
    """Cascade loss co-designed with the dual-branch stage-1 hypothesis axis.

    Stage 1 (global 32 + local 16, merged sorted axis):

      * ``L48``  — soft-label CE over all 48 candidates. When the local branch
        is right its dense bins carry most of the label mass; when it is wrong
        its candidates automatically become negatives and the correct global
        bin gets the positive signal. Supervised on ``valid & global-in-range``
        (the guard covers ~all valid pixels by construction).
      * ``L_global`` — auxiliary CE over the *global branch only* (softmax over
        its 32 gathered logits). This trains the guard to localize GT on every
        pixel at every step, even while the local branch wins the 48-way
        softmax, so its rescue ability never atrophies.
      * ``L_local`` — auxiliary CE over the local branch, only where GT actually
        falls inside the local window (a wrong prior must not force the local
        branch to hallucinate; L64 already presses its candidates down).
      * ``reg``  — interval-normalized Huber on the mode-centered depth over
        ALL valid pixels (no in-range gating, no hard clamp): out-of-range
        pixels keep a bounded pull toward GT instead of a blind spot.

    Stages 2-4: soft-label CE (in-range) + all-valid Huber whose normalizer is
    the stage's own bin interval; the regression keeps correcting the previous
    stage's center even when GT fell outside the current window.

    Edge-band pixels (rule-based E map from the prior) get ``edge_reg_boost``x
    regression weight: they are ~5-10% of pixels and drown at uniform weight.

    Diagnostics (per batch, window-aggregated by the training loop):
      * stage1/global_in_range — guard coverage, should sit at ~1.0;
      * stage1/local_hit       — GT inside the local window (prior quality);
      * stage1/guard_win_rate  — argmax fell on a global bin (rescue firing);
      * stage1/prior_abs_err   — the prior's own error, the baseline the
        network must beat.
    """

    stage_names = ("stage1", "stage2", "stage3", "stage4")

    def __init__(self, cfg: LossConfig, stage_weights: StageWeights) -> None:
        self.cfg = cfg
        self.stage_weights = stage_weights

    @staticmethod
    def _to_stage_res(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        if tuple(x.shape[-2:]) == hw:
            return x
        return F.interpolate(x.unsqueeze(1).float(), size=hw, mode="nearest").squeeze(1)

    def _edge_weight(self, edge: torch.Tensor) -> torch.Tensor:
        return 1.0 + (self.cfg.edge_reg_boost - 1.0) * edge.clamp(0.0, 1.0)

    def __call__(
        self,
        outputs: dict,
        batch: dict,
        step: int = 0,
        **_: object,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.cfg
        sw = self.stage_weights
        weights = {"stage1": sw.stage1, "stage2": sw.stage2,
                   "stage3": sw.stage3, "stage4": sw.stage4}

        device = outputs["stage1"]["depth"].device
        depth_gt_full = batch["depth_gt"].to(device).float()
        mask_full = batch.get("mask")
        mask_full = (depth_gt_full > 0).float() if mask_full is None else mask_full.to(device).float()
        # GT beyond the scene's physical depth range (DTU backgrounds past
        # depth_values max, ~2.5% of pixels) is unreachable by ANY hypothesis
        # confined to [dmin, dmax]: exclude it from supervision and metrics
        # instead of letting it pollute them as a permanent error floor.
        if "depth_values" in batch:
            dv = batch["depth_values"].to(device).float()
            d_lo = dv.amin(dim=1).view(-1, 1, 1)
            d_hi = dv.amax(dim=1).view(-1, 1, 1)
            in_scene = (depth_gt_full >= d_lo) & (depth_gt_full <= d_hi)
            gt_out_frac = float((~in_scene & (mask_full > 0) & (depth_gt_full > 0)).float().mean())
            mask_full = mask_full * in_scene.float()
        else:
            gt_out_frac = 0.0

        total = outputs["stage1"]["depth"].new_zeros(())
        logs: dict[str, float] = {}

        # ------------------------------ stage 1 ------------------------------ #
        s1 = outputs["stage1"]
        hypos = s1["depth_hypos"]
        logits = s1["logits"]
        hw1 = tuple(hypos.shape[-2:])
        gt1 = self._to_stage_res(depth_gt_full, hw1)
        valid1 = self._to_stage_res(mask_full, hw1).bool() & (gt1 > 0)

        B = hypos.shape[0]
        g_lo = s1["global_lo"].view(B, 1, 1)
        g_hi = s1["global_hi"].view(B, 1, 1)
        g_in_range = (gt1 >= g_lo) & (gt1 <= g_hi)
        l_in_range = (gt1 >= s1["local_lo"]) & (gt1 <= s1["local_hi"])
        sup64 = valid1 & g_in_range
        sup16 = valid1 & l_in_range

        edge1 = s1["edge"]
        w_edge1 = self._edge_weight(edge1)

        l_ce64 = soft_label_cross_entropy(logits, hypos, gt1, sup64) if cfg.use_cross_entropy \
            else logits.new_zeros(())

        logits48 = logits.gather(1, s1["global_idx"])
        hypos48 = hypos.gather(1, s1["global_idx"])
        l_ce48 = soft_label_cross_entropy(logits48, hypos48, gt1, sup64) if cfg.use_cross_entropy \
            else logits.new_zeros(())

        logits16 = logits.gather(1, s1["local_idx"])
        hypos16 = hypos.gather(1, s1["local_idx"])
        l_ce16 = soft_label_cross_entropy(logits16, hypos16, gt1, sup16) if cfg.use_cross_entropy \
            else logits.new_zeros(())

        # --- 分支校准 (branch calibration) ---
        # 目标不是 "GT 是否落在 local 跨度内", 而是 "哪个分支的最优候选更接近
        # GT"。共位测试显示 local 获胜的像素里只有 8.5% 的 GT 真在 local 跨度
        # 内 —— 用 in-span 当标签会让分支判定学到一个偏得很厉害的目标。
        #
        # 直接监督 q (= conf_used, 来自 SPRE 且可微), 不是监督 decoder 的 logits:
        # 分支先验是乘在 logits 上的常数, 只训 decoder 等于让它去抵消这个常数,
        # 梯度传不回 SPRE。
        with torch.no_grad():
            d48 = (hypos.gather(1, s1["global_idx"]) - gt1.unsqueeze(1)).abs().amin(1)
            d16 = (hypos.gather(1, s1["local_idx"]) - gt1.unsqueeze(1)).abs().amin(1)
            gi_t = s1["global_interval"].view(B, 1, 1).clamp_min(1e-4)
            # 软标签: 两个分支 oracle 误差接近时不应该给 0/1 硬标签
            branch_tgt = torch.sigmoid((d48 - d16) / (cfg.branch_tau_gi * gi_t))
            # 只监督差距明显的像素, 且 local 分支必须真的携带先验候选
            margin = (d48 - d16).abs() > cfg.branch_margin_gi * gi_t
        q = s1.get("branch_q")
        m_br = valid1 & margin & s1["branch_active"]
        if q is not None and m_br.any():
            # 手写 BCE 而不是 F.binary_cross_entropy: 后者在 autocast 下被禁用,
            # 而 q 已经过了 sigmoid (来源可能是 SPRE / cached conf / edge), 拿不到
            # 统一的 pre-sigmoid logits。在 float32 里算, 数值上也更稳。
            qc = q.float().clamp(1e-4, 1.0 - 1e-4)
            t = branch_tgt.float()
            bce = -(t * qc.log() + (1.0 - t) * (1.0 - qc).log())
            l_branch = bce[m_br].mean()
            branch_pred = qc
        else:
            l_branch = logits.new_zeros(())
            branch_pred = torch.zeros_like(branch_tgt)

        gi1 = s1["global_interval"].view(B, 1, 1).expand_as(gt1)
        l_reg1 = normalized_huber_loss(s1["depth"], gt1, valid1, gi1, weight=w_edge1)

        total = total + weights["stage1"] * (
            cfg.w_ce * l_ce64
            + cfg.w_global_aux * l_ce48
            + cfg.w_local_aux * l_ce16
            + cfg.w_branch * l_branch
            + cfg.w_reg * l_reg1
        )

        # stage-1 diagnostics
        with torch.no_grad():
            winner_local = s1["is_local"].gather(1, s1["mode_idx"]).squeeze(1)
            if isinstance(outputs.get("vis"), dict):
                for k, v in outputs["vis"].items():
                    logs[f"vis/{k}"] = float(v)
            logs["stage1/ce"] = float(l_ce64.detach())
            logs["stage1/ce_global_aux"] = float(l_ce48.detach())
            logs["stage1/ce_local_aux"] = float(l_ce16.detach())
            logs["stage1/branch_ce"] = float(l_branch.detach())
            logs["stage1/branch_active_frac"] = float(s1["branch_active"][valid1].float().mean()) if valid1.any() else 0.0
            logs["stage1/branch_sup_frac"] = float(m_br[valid1].float().mean()) if valid1.any() else 0.0
            # 条件关门率 —— 全图的 branch_active_frac 说明不了门关得对不对。
            # 要看的是: 在 local 分支确实更差的地方, 门有没有关。
            gate_off = (~s1["branch_active"]) & valid1
            lb = (branch_tgt > 0.5) & valid1        # local 分支 oracle 更好
            gb = (branch_tgt <= 0.5) & valid1       # global 分支更好 -> 该关门
            if gb.any():
                logs["stage1/gate_off_given_global_better"] = float(gate_off[gb].float().mean())
            if lb.any():
                logs["stage1/gate_off_given_local_better"] = float(gate_off[lb].float().mean())
            if "prior_corrupt_mask" in batch:
                cm1 = self._to_stage_res(
                    batch["prior_corrupt_mask"].to(gt1.device).float(), gt1.shape[-2:]) > 0.5
                if (cm1 & valid1).any():
                    logs["stage1/gate_off_given_corrupted"] = float(gate_off[cm1 & valid1].float().mean())
            # 两分支 oracle 误差差距的分位数 -> 用来定 branch_margin_gi,
            # 而不是拍一个 0.5 然后发现只监督了 11% 的像素
            dd = ((d48 - d16).abs() / gi_t)[valid1]
            if dd.numel():
                for q_, nm in ((0.25, "p25"), (0.50, "p50"), (0.75, "p75")):
                    logs[f"stage1/branch_gap_{nm}"] = float(dd.float().quantile(q_))
            if m_br.any():
                logs["stage1/branch_local_better"] = float((branch_tgt[m_br] > 0.5).float().mean())
                logs["stage1/branch_acc"] = float(
                    ((branch_pred[m_br] > 0.5) == (branch_tgt[m_br] > 0.5)).float().mean())

            # --- 分支先验 前/后 对比 (bp_*) ---
            # apply_branch_prior 在各分支内部 logsumexp 归一化后再乘 q, 结果是
            # 两个分支的概率总质量被硬指派成 (1-q, q) —— cost volume 学到的
            # "该信哪个分支" 被整段替换成一个 SPRE 标量。这组诊断直接回答它是
            # 帮忙还是添乱: bp_post_abs_err < bp_raw_abs_err 才说明它有正贡献。
            # 注意 raw logits 是在 post-prior 的损失下训练出来的, 所以这只能证明
            # "当前模型里先验是否即时伤害预测"; 结果接近时仍需消融训练确认。
            lr_ = s1.get("logits_raw")
            if lr_ is not None and valid1.any():
                idx_raw = lr_.detach().float().argmax(dim=1, keepdim=True)
                idx_post = s1["mode_idx"]
                d_raw = hypos.gather(1, idx_raw).squeeze(1)
                d_post = hypos.gather(1, idx_post).squeeze(1)
                e_raw = (d_raw - gt1).abs()
                e_post = (d_post - gt1).abs()
                logs["stage1/bp_raw_abs_err"] = float(e_raw[valid1].mean())
                logs["stage1/bp_post_abs_err"] = float(e_post[valid1].mean())
                flip = (idx_raw != idx_post).squeeze(1) & valid1
                logs["stage1/bp_flip_frac"] = float(flip.float()[valid1].mean())
                if flip.any():
                    logs["stage1/bp_flip_help"] = float((e_post[flip] < e_raw[flip]).float().mean())
                    logs["stage1/bp_flip_hurt"] = float((e_post[flip] > e_raw[flip]).float().mean())
                # GT 候选在 stage1 posterior 里排第几 —— 决定双模态值不值得做。
                # config 注释写得很清楚: 只有它通常是 rank 2 时 top-2 才真的有用,
                # 而 oor_recoverable (>=10% 峰值质量) 比 rank 弱得多。
                p1_ = s1["prob"].detach().float()
                gtb_r = (hypos - gt1.unsqueeze(1)).abs().argmin(dim=1, keepdim=True)
                p_at = p1_.gather(1, gtb_r)
                rank = (p1_ > p_at).sum(dim=1) + 1              # 1 = 已经是 winner
                rv = rank[valid1].float()
                if rv.numel():
                    for r_ in (1, 2, 3, 5):
                        logs[f"stage1/gt_rank_le{r_}"] = float((rv <= r_).float().mean())
                    logs["stage1/gt_rank_p50"] = float(rv.quantile(0.5))
                p_raw = F.softmax(lr_.detach().float(), dim=1)
                gtb_ = (hypos - gt1.unsqueeze(1)).abs().argmin(dim=1, keepdim=True)
                logs["stage1/bp_raw_p_at_gt"] = float(p_raw.gather(1, gtb_).squeeze(1)[valid1].mean())
                # 落在哪个分支 vs 哪个分支的 oracle 候选更好
                loc_raw = s1["is_local"].gather(1, idx_raw).squeeze(1) > 0.5
                loc_post = s1["is_local"].gather(1, idx_post).squeeze(1) > 0.5
                tgt_loc = branch_tgt > 0.5
                if m_br.any():
                    logs["stage1/bp_raw_branch_acc"] = float((loc_raw[m_br] == tgt_loc[m_br]).float().mean())
                    logs["stage1/bp_post_branch_acc"] = float((loc_post[m_br] == tgt_loc[m_br]).float().mean())
            logs["stage1/reg"] = float(l_reg1.detach())
            logs["stage1/in_range"] = float(g_in_range[valid1].float().mean()) if valid1.any() else 1.0
            logs["stage1/global_in_range"] = logs["stage1/in_range"]
            logs["stage1/local_hit"] = float(l_in_range[valid1].float().mean()) if valid1.any() else 1.0
            logs["stage1/guard_win_rate"] = float((1.0 - winner_local)[valid1].mean()) if valid1.any() else 0.0
            logs["stage1/p_max"] = float(s1["prob"].detach().amax(dim=1).mean())
            # p_at_gt: stage-1 posterior mass on the candidate nearest GT.
            # This decides whether the lost-tail pixels are worth chasing at all.
            # If a pixel that later falls outside a later stage's window already had
            # ~zero mass here, the matching evidence never supported the right
            # depth and no window policy can recover it — widening only coarsens
            # the bins for everyone. If instead it carries a real secondary mode,
            # the evidence exists and the window policy is what threw it away.
            p1 = s1["prob"].detach()
            gt_bin = (hypos - gt1.unsqueeze(1)).abs().argmin(dim=1, keepdim=True)
            p_at_gt = p1.gather(1, gt_bin).squeeze(1)
            p_max_px = p1.amax(dim=1)
            if valid1.any():
                logs["stage1/p_at_gt"] = float(p_at_gt[valid1].mean())
            # NOTE: interval_mm averages the merged 64-bin axis, so it is
            # dominated by the 48 guard bins and sits at ~span/63 regardless of
            # what the local branch does — it says nothing about conf/SPRE.
            logs["stage1/interval_mm"] = float(s1["interval"][valid1.unsqueeze(1).expand_as(s1["interval"])].mean()) \
                if valid1.any() else 0.0
            # The local window half-width IS conf/SPRE's only effect on the
            # search: half = (0.75 + 1.25*(1-r)) * gi, i.e. 8.1mm at r=1 and
            # 21.7mm at r=0 on DTU. The std across pixels is the falsifiable
            # part — a constant r (SPRE learning nothing) pins it near zero.
            local_half = 0.5 * (s1["local_hi"] - s1["local_lo"])
            if valid1.any():
                lh = local_half[valid1]
                logs["stage1/local_half_mm"] = float(lh.mean())
                logs["stage1/local_half_std"] = float(lh.std()) if lh.numel() > 1 else 0.0
            logs["stage1/edge_frac"] = float(edge1.mean())
            prior_err = (s1["prior"] - gt1).abs()
            prior_valid = valid1 & (s1["prior"] > 0)
            logs["stage1/prior_abs_err"] = float(prior_err[prior_valid].mean()) if prior_valid.any() else 0.0
            logs["stage1/gt_out_of_scene"] = gt_out_frac
            # supervised-pixel err split by whether the prior was corrupted (the
            # rescue-rate signal); mask arrives at full res from the dataset.
            if "prior_corrupt_mask" in batch:
                cm = self._to_stage_res(batch["prior_corrupt_mask"].to(device).float(), hw1).bool()
                err1 = (s1["depth"].detach() - gt1).abs()
                vc = valid1 & cm
                vk = valid1 & ~cm
                if vc.any():
                    logs["stage1/err_corrupted"] = float(err1[vc].mean())
                if vk.any():
                    logs["stage1/err_clean"] = float(err1[vk].mean())

        # --------------------------- stages 2 / 3 / 4 -------------------------- #
        for name in ("stage2", "stage3", "stage4"):
            stage = outputs[name]
            hypos_k = stage["depth_hypos"]
            logits_k = stage["logits"]
            depth_k = stage["depth"]
            hw = tuple(hypos_k.shape[-2:])

            gt = self._to_stage_res(depth_gt_full, hw)
            valid = self._to_stage_res(mask_full, hw).bool() & (gt > 0)

            hypo_min = hypos_k[:, 0]
            hypo_max = hypos_k[:, -1]
            in_range = (gt >= hypo_min) & (gt <= hypo_max)
            sup = valid & in_range
            # 归一化尺度: 逆深度轴在深度空间是非均匀的, span/(D-1) 不再成立。
            # 有 interval_local 时用 winner 处的**局部**间距, 没有就退回旧写法
            # (legacy 路径下两者恒等)。
            iv_local = stage.get("interval_local")
            if iv_local is not None and "mode_idx" in stage:
                interval = iv_local.detach().gather(1, stage["mode_idx"]).squeeze(1).clamp(min=1e-4)
            else:
                interval = ((hypo_max - hypo_min) / max(hypos_k.shape[1] - 1, 1)).detach().clamp(min=1e-4)

            w_edge = self._edge_weight(stage["edge"]) if "edge" in stage else None

            l_ce = soft_label_cross_entropy(logits_k, hypos_k, gt, sup) if cfg.use_cross_entropy \
                else depth_k.new_zeros(())

            if name == "stage4" and "residual" in stage:
                # ---- W1-D: 硬 MAP + 逐候选残差 ----
                # 不能沿用 "SmoothL1 跑在所有有效像素上": argmax 不可导、候选轴已
                # detach, MAP 选错表面或 GT 出界时梯度既改不了 MAP、也改不了范围、
                # 更移不动概率, 只能把残差头推向饱和, 让它在错误模式上学习。
                #
                # 逐候选残差: 第 j 个残差只负责自己那个 Voronoi 单元内的修正,
                # 与它是否赢得 argmax 无关。
                res = stage["residual"]
                iv = iv_local.detach()
                j_gt = (hypos_k.detach() - gt.unsqueeze(1)).abs().argmin(dim=1, keepdim=True)
                d_j = hypos_k.detach().gather(1, j_gt).squeeze(1)
                dd_j = iv.gather(1, j_gt).squeeze(1).clamp_min(1e-6)
                u_gt = (2.0 * (gt - d_j) / dd_j).clamp(-1.0, 1.0)
                r_j = torch.tanh(res.gather(1, j_gt).squeeze(1).float())
                if sup.any():
                    per = F.smooth_l1_loss(r_j, u_gt, beta=1.0, reduction="none")
                    l_res = per[sup].mean()
                else:
                    l_res = depth_k.new_zeros(())
                # OOR 方向损失: 只告诉后验 "正确答案在轴的哪一端"。它正好覆盖
                # CE 掩码 (valid & in_range) 的补集, 不参与任何跨表面平均。
                below = valid & (gt < hypo_min)
                above = valid & (gt > hypo_max)
                n_oor = (below.sum() + above.sum()).clamp_min(1)
                logp = F.log_softmax(logits_k.float(), dim=1)
                l_oor = -(logp[:, 0][below].sum() + logp[:, -1][above].sum()) / n_oor
                total = total + weights[name] * (
                    cfg.w_ce * l_ce
                    + getattr(cfg, "w_residual", 1.0) * l_res
                    + getattr(cfg, "w_oor", 0.1) * l_oor
                )
                logs["stage4/res"] = float(l_res.detach())
                logs["stage4/oor_dir"] = float(l_oor.detach())
                with torch.no_grad():
                    if sup.any():
                        logs["stage4/residual_abs_p90"] = float(r_j[sup].abs().float().quantile(0.9))
                        logs["stage4/res_target_is_map"] = float(
                            (j_gt == stage["mode_idx"]).squeeze(1)[sup].float().mean())
                    logs["stage4/oor_frac"] = float((below | above)[valid].float().mean()) \
                        if valid.any() else 0.0
            else:
                # ALL valid pixels: even when GT fell outside this stage's window the
                # regression keeps a bounded pull on the previous stage's center.
                l_reg = normalized_huber_loss(depth_k, gt, valid, interval, weight=w_edge)
                total = total + weights[name] * (cfg.w_ce * l_ce + cfg.w_reg * l_reg)
                logs[f"{name}/reg"] = float(l_reg.detach())

            logs[f"{name}/ce"] = float(l_ce.detach())
            logs[f"{name}/in_range"] = float(in_range[valid].float().mean()) if valid.any() else 1.0
            logs[f"{name}/p_max"] = float(stage["prob"].detach().amax(dim=1).mean())
            logs[f"{name}/interval_mm"] = float(interval[valid].mean()) if valid.any() else 0.0
            # Of the pixels this stage lost (GT outside its window), how many did
            # stage 1 have evidence for? ``oor_recoverable`` is the fraction whose
            # GT bin held at least 10% of stage 1's peak mass — a real secondary
            # mode that a posterior-driven range would have kept. Near zero means
            # the tail is a matching failure and no window policy can fix it;
            # substantially above zero means the window policy is what lost them.
            oor = valid & ~in_range
            if oor.any():
                pg = self._to_stage_res(p_at_gt, hw)[oor]
                pm = self._to_stage_res(p_max_px, hw)[oor]
                logs[f"{name}/p_at_gt_oor"] = float(pg.mean())
                logs[f"{name}/oor_recoverable"] = float((pg >= 0.1 * pm).float().mean())

        # ------------------- W1-C: 可学习范围控制器的监督 ------------------- #
        # pinball(分位数)损失。为什么不是 "hinge/interval + lambda*log(half)":
        # 那个目标在 half -> 0 时第一项有限而 log(half) -> -inf, **无下界**,
        # 最优解是窗宽塌缩。Laplace NLL |r-mu|*exp(-s)+s 有同样的病 (s -> -inf),
        # 只有在 s 被硬夹时才安全。pinball 非负、有下界, 宽度惩罚自带。
        #
        # Q_q(u) = max(q*u, (q-1)*u) 关于 v 的最小值点是 r 的 q 分位数, 所以
        # v_lo -> Q_alpha, v_hi -> Q_{1-alpha}, 区间收敛到 GT 位置的**条件**中心
        # tau 区间。注意 tau 是**优化目标不是保证** —— 实际覆盖必须在独立验证集
        # 上量 (每个像素只有一个 GT 样本、控制器容量有限、条件变量不完备、
        # 物理边界与 rho 上限会截断)。
        rc = outputs.get("range_ctrl")
        if isinstance(rc, dict) and getattr(cfg, "w_range", 0.0) > 0:
            l_range_sum = total.new_zeros(())
            for name, st in sorted(rc.items()):
                hw = tuple(st["hw"])
                gt_k = self._to_stage_res(depth_gt_full, hw)
                vk = self._to_stage_res(mask_full, hw).bool() & (gt_k > 0)
                if not vk.any():
                    continue
                v_gt = 1.0 / gt_k.clamp_min(1e-6)
                # **两个尺度不是一回事, 不能共用一个变量。**
                #   w_parent = 上一级 winner 的局部间距 —— center loss 的定义域
                #   w_loss   = pinball 的归一化分母
                # 之前两者都用 w_parent, 而 w_parent 同时又是基准尺度 A_k 的因子,
                # 且来自上一级**已经变宽**的轴: 轴宽 -> A 大 -> 更宽; 轴宽 ->
                # 分母大 -> 宽度惩罚显得更小。两条路都指向"更宽"。
                # pinball_scale=global 把分母换成与当前窗口无关的 g_v, 切断第二条。
                w_parent = st["w_bar"].detach().clamp_min(1e-12)
                _sc = str(getattr(cfg, "pinball_scale", "axis")).lower()
                w_loss = (st["w_fixed"] if (_sc == "global" and "w_fixed" in st)
                          else st["w_bar"]).detach().clamp_min(1e-12)
                a = 0.5 * (1.0 - float(st["tau"]))
                u_lo = (v_gt - st["v_lo"]) / w_loss
                u_hi = (v_gt - st["v_hi"]) / w_loss
                q_lo = torch.maximum(a * u_lo, (a - 1.0) * u_lo)
                q_hi = torch.maximum((1.0 - a) * u_hi, -a * u_hi)
                l_pin = (q_lo + q_hi)[vk].mean()
                # 中心项: 非对称的 h_lo/h_hi 单靠 pinball 可以用 "中心偏一点 +
                # 一侧更宽" 达到同样的端点, 这一项把中心钉住。只在 GT 已经落在
                # 上一级一个 bin 之内时生效 —— 那正是亚 bin 修正的定义域。
                # **这里必须用 w_parent 而不是 g_v**: center head 的职责就是修正
                # "一个父级 bin 以内" 的量化误差, 换成全局尺度就改了它的定义域。
                m_c = vk & ((v_gt - st["v_m"]).abs() <= w_parent)
                if m_c.any():
                    l_ctr = F.smooth_l1_loss(
                        ((st["v_c"] - v_gt) / w_parent)[m_c],
                        torch.zeros_like(v_gt)[m_c], beta=1.0)
                else:
                    l_ctr = total.new_zeros(())
                l_range_sum = l_range_sum + l_pin + getattr(cfg, "w_center", 0.2) * l_ctr
                with torch.no_grad():
                    cov = ((v_gt >= torch.minimum(st["v_lo"], st["v_hi"]))
                           & (v_gt <= torch.maximum(st["v_lo"], st["v_hi"])))
                    logs[f"{name}/pinball"] = float(l_pin.detach())
                    logs[f"{name}/center_l1"] = float(l_ctr.detach())
                    logs[f"{name}/coverage_at_tau"] = float(cov[vk].float().mean())
                    logs[f"{name}/tau"] = float(st["tau"])
            total = total + getattr(cfg, "w_range", 1.0) * l_range_sum
            logs["range/total"] = float(l_range_sum.detach())

        # ------------------- W3-C: 最终重建置信度 (L_conf) ------------------- #
        # v1 目标 y = 1[|d_final - gt| < tau]。不加 n_geo —— 它要等所有参考视图的
        # 深度图存完、在融合阶段才算得出来, 前向拿不到; 用 GT 算的 n_geo 只能进
        # 标签, 而 v1 不需要它 (n_valid 和相关性统计已经给了几何代理信息)。
        fc = outputs.get("fusion_conf")
        if isinstance(fc, dict) and getattr(cfg, "w_conf", 0.0) > 0:
            logit = fc["logit"].float()
            hw_c = logit.shape[-2:]
            gt_c = self._to_stage_res(depth_gt_full, hw_c)
            v_c = self._to_stage_res(mask_full, hw_c).bool() & (gt_c > 0)
            d_c = outputs["depth_full"].float()
            if d_c.shape[-2:] != hw_c:
                d_c = self._to_stage_res(d_c, hw_c)
            err = (d_c.detach() - gt_c).abs()
            y = (err < float(getattr(cfg, "conf_tau_mm", 2.0))).float()
            if v_c.any():
                zi, yi = logit[v_c], y[v_c]
                # 平衡 BCE: acc@2mm 在 0.9 附近, 不平衡的话头直接学 "全部说对"
                # 就能拿到 0.9 的准确率, 而那样的置信度当过滤器毫无用处。
                pos = yi.mean().clamp(1e-3, 1.0 - 1e-3)
                w = torch.where(yi > 0.5, 0.5 / pos, 0.5 / (1.0 - pos))
                l_conf = F.binary_cross_entropy_with_logits(zi, yi, weight=w)
                total = total + getattr(cfg, "w_conf", 0.0) * l_conf
                logs["conf/bce"] = float(l_conf.detach())
                with torch.no_grad():
                    from models.fusion_conf import (
                        brier_score, expected_calibration_error,
                        risk_at_coverage, risk_coverage,
                    )
                    # 训练期的 ECE 只是监控。真正的校准是训练**之后**在独立验证集
                    # 上拟合温度 (scripts/calibrate_conf.py) —— 在训练集上标定等于
                    # 把过拟合程度也标定进去。
                    pi = torch.sigmoid(zi)
                    ri = err[v_c]
                    logs["conf/pos_frac"] = float(pi.new_tensor(float(yi.mean())))
                    logs["conf/ece"] = expected_calibration_error(pi, yi)
                    logs["conf/brier"] = brier_score(pi, yi)
                    _, _, aurc = risk_coverage(pi, ri)
                    logs["conf/aurc_mm"] = aurc
                    logs["conf/risk_at_0.6_mm"] = risk_at_coverage(pi, ri, 0.6)
                    logs["conf/mean"] = float(pi.mean())

        # ------------------- W3-B: 源视图可见性监督 (L_vis) ------------------- #
        # y_vis_{s,j}(x) = 1[ z_{s,j}(x) <= D_s^gt(pi_{s,j}(x)) + delta_occ ]
        # **多标签**, 每个 (source, 候选) 独立 sigmoid —— 不是 source 维 softmax。
        # 真实情况常常是 4 个源视图全部可见, 强制竞争表达不了这件事, 那大概也是
        # 旧实现退化成近似恒等 (熵 0.974, 有效源视角 3.87/4) 的原因之一。
        vsup = outputs.get("vis_sup")
        if isinstance(vsup, dict) and getattr(cfg, "w_vis", 0.0) > 0 \
                and "src_depth_gt" in batch:
            zl = vsup["logits"].float()                    # [S,B,D,H,W]
            uv = vsup["uv"].float()                        # [S,B,D,H,W,2] 整幅图归一化
            zc = vsup["z"].float()                         # [S,B,D,H,W]
            sd = batch["src_depth_gt"].to(device).float()  # [B,S,Hi,Wi]
            sm = batch["src_mask_gt"].to(device).float()
            S_, Bn, Dn, Hk, Wk = zl.shape
            delta = float(getattr(cfg, "delta_occ_mm", 2.0))
            num = zl.new_zeros(()); den = 0.0
            pos = 0.0
            for si in range(min(S_, sd.shape[1])):
                g = uv[si].reshape(Bn, Dn * Hk, Wk, 2)
                # **nearest**: 在遮挡边界上双线性会把前景和背景插成一个中间深度,
                # 那正好是标签唯一有意义的地方 —— 插出来的面两边都不属于。
                dg = F.grid_sample(sd[:, si:si + 1], g, mode="nearest",
                                   padding_mode="zeros", align_corners=True)
                mg = F.grid_sample(sm[:, si:si + 1], g, mode="nearest",
                                   padding_mode="zeros", align_corners=True)
                dg = dg.view(Bn, Dn, Hk, Wk)
                mg = mg.view(Bn, Dn, Hk, Wk)
                inb = (uv[si][..., 0].abs() <= 1.0) & (uv[si][..., 1].abs() <= 1.0)
                mk = (mg > 0.5) & (dg > 0) & inb & torch.isfinite(zc[si])
                if not mk.any():
                    continue
                y = (zc[si] <= dg + delta).float()
                num = num + F.binary_cross_entropy_with_logits(
                    zl[si][mk], y[mk], reduction="sum")
                den += float(mk.sum())
                pos += float(y[mk].sum())
            if den > 0:
                l_vis = num / den
                total = total + getattr(cfg, "w_vis", 0.0) * l_vis
                logs["vis/bce"] = float(l_vis.detach())
                logs["vis/pos_frac"] = pos / den
                logs["vis/sup_frac"] = den / max(float(zl.numel()), 1.0)

        # SPRE 的逐级门。**数值本身不是消融读数** —— gamma*r*W = (c*gamma)*r*(W/c),
        # 下游卷积可以补偿门的大小。判断依赖度用 scripts/gamma_intervention.py。
        g = outputs.get("spre_gamma")
        if g is not None:
            with torch.no_grad():
                for i in range(int(g.numel())):
                    logs[f"spre/gamma_s{i + 1}"] = float(g[i])

        # 窗宽归因 (models/depth_range.refine_range_from_posterior)
        rd = outputs.get("range_diag")
        if isinstance(rd, dict):
            with torch.no_grad():
                for sname, st in rd.items():
                    for k_, v_ in st.items():
                        if k_ == "maps":
                            continue
                        logs[f"{sname}/{k_}"] = float(v_)

        # ---- 窗口需求诊断: 要覆盖 GT, stage4 的 half 实际需要多宽 ----
        # Phase 2 要决定 range_min_gi[2] 抬到多少。直接量"|GT - center| 的分位数"
        # 比试错快得多: 需求分位数就是覆盖率曲线的逆函数。同时按 floor 是否 binding
        # 拆开, 才知道抬 floor 到底能救多少 —— s3→s4 只有约 1/3 像素受 floor 支配。
        if isinstance(rd, dict) and "stage4" in rd and isinstance(rd["stage4"].get("maps"), dict):
            with torch.no_grad():
                mp = rd["stage4"]["maps"]
                hw4 = tuple(mp["half"].shape[-2:])
                gt4 = self._to_stage_res(depth_gt_full, hw4)
                v4 = self._to_stage_res(mask_full, hw4).bool() & (gt4 > 0)
                if v4.any():
                    need = (gt4 - mp["center"]).abs()          # 覆盖 GT 所需的 half
                    have = mp["half"]
                    fb = mp["half_raw"] < mp["floor"]           # 该像素由 floor 决定
                    nv = need[v4]
                    for q_, nm in ((0.5, "p50"), (0.75, "p75"), (0.9, "p90"), (0.95, "p95")):
                        logs[f"stage4/need_half_{nm}"] = float(nv.float().quantile(q_))
                    logs["stage4/have_half_mean"] = float(have[v4].mean())
                    # 覆盖率按 floor 是否 binding 拆开
                    cov = (need <= have)
                    for msk, nm in ((fb & v4, "floorbound"), ((~fb) & v4, "free")):
                        if msk.any():
                            logs[f"stage4/cover_{nm}"] = float(cov[msk].float().mean())
                            logs[f"stage4/frac_{nm}"] = float(msk[v4].float().mean())
                    # 若把 half 乘 k, 覆盖率会变成多少 —— 直接给出交换曲线
                    for k_ in (1.5, 2.0, 3.0):
                        logs[f"stage4/cover_x{k_:g}"] = float((need <= have * k_)[v4].float().mean())

        # ------------------------- SPRE reliability head ------------------------- #
        if "spre" in outputs:
            total = total + self._spre_loss(outputs, batch, depth_gt_full, mask_full, device, logs)

        logs["loss"] = float(total.detach())
        return total, logs

    def _spre_loss(self, outputs, batch, depth_gt_full, mask_full, device, logs):
        """Supervise the learned prior reliability r (see models/spre.py).

        (1) corruption BCE: corrupted prior pixels -> 0, clean valid prior -> 1
            (labels are free, from data/prior_corruption.py's corrupt mask).
        (2) prior-error soft target on GT-valid pixels: r -> exp(-(|prior-gt|/tau)^2)
            so r is calibrated to the prior's *true* local accuracy, not just to
            synthetic corruption.
        """
        spre = outputs["spre"]
        logits = spre["logits"]                       # [B, h, w]
        hw = tuple(logits.shape[-2:])
        r = torch.sigmoid(logits)

        prior = self._to_stage_res(batch["depth_prior"].to(device).float(), hw)
        gt = self._to_stage_res(depth_gt_full, hw)
        gt_valid = self._to_stage_res(mask_full, hw) > 0.5
        prior_valid = prior > 0

        # (1) corruption BCE (autocast-safe with logits)
        if "prior_corrupt_mask" in batch:
            cm = self._to_stage_res(batch["prior_corrupt_mask"].to(device).float(), hw).clamp(0.0, 1.0)
        else:
            cm = torch.zeros_like(r)
        target = 1.0 - cm
        m_bce = prior_valid.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        # 类别平衡: 腐蚀只覆盖一小部分像素, 按全体平均的话大多数目标都是 1,
        # 网络只要普遍输出高 q 就能拿到低损失 —— 实测 r_clean=0.753 vs
        # r_corrupt=0.708, 只差 0.045, 而腐蚀是合成的、本该很好分。
        #   L = 1/2 mean_clean(l) + 1/2 mean_corrupt(l)
        if self.cfg.spre_balance_corrupt:
            w_c = (m_bce * cm)
            w_k = (m_bce * (1.0 - cm))
            n_c, n_k = w_c.sum(), w_k.sum()
            if n_c > 0 and n_k > 0:
                l_bce = 0.5 * (bce * w_c).sum() / n_c + 0.5 * (bce * w_k).sum() / n_k
            else:
                l_bce = (bce * m_bce).sum() / m_bce.sum().clamp_min(1.0)
        else:
            l_bce = (bce * m_bce).sum() / m_bce.sum().clamp_min(1.0)

        # (2) prior-error soft target
        tau = float(self.cfg.spre_soft_tau_mm)
        t = torch.exp(-(((prior - gt).abs() / tau) ** 2))
        m_soft = (gt_valid & prior_valid).float()
        l_soft = ((r - t) ** 2 * m_soft).sum() / m_soft.sum().clamp_min(1.0)

        with torch.no_grad():
            logs["spre/bce"] = float(l_bce.detach())
            # 腐蚀像素比例 —— 不记这个就无法判断类别不平衡有多严重
            logs["spre/corrupt_frac"] = float((cm * m_bce).sum() / m_bce.sum().clamp_min(1.0))
            # q 的分位数: 决定 gate_hard_conf 该定在哪, 比拍一个 0.45 靠谱
            rv = r[prior_valid]
            if rv.numel():
                for q_, nm in ((0.01, "p01"), (0.05, "p05"), (0.10, "p10"),
                               (0.25, "p25"), (0.50, "p50"), (0.90, "p90")):
                    logs[f"spre/q_{nm}"] = float(rv.float().quantile(q_))
            logs["spre/soft"] = float(l_soft.detach())
            corrupt = (cm > 0.5) & prior_valid
            clean = (cm <= 0.5) & prior_valid
            # r_clean should sit well above r_corrupt — the separation is the
            # falsifiable signal (a poor-man's AUROC).
            if corrupt.any():
                logs["spre/r_corrupt"] = float(r[corrupt].mean())
            if clean.any():
                logs["spre/r_clean"] = float(r[clean].mean())

        return self.cfg.w_spre * l_bce + self.cfg.w_spre_soft * l_soft
