from __future__ import annotations

import torch
import torch.nn.functional as F


def make_pixel_grid(h: int, w: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    ones = torch.ones_like(xx)
    return torch.stack([xx, yy, ones], dim=0)


def unproject_depth(
    depth: torch.Tensor,
    K_inv: torch.Tensor,
    extrinsic_inv: torch.Tensor,
) -> torch.Tensor:
    B, H, W = depth.shape
    grid = make_pixel_grid(H, W, depth.device, depth.dtype)
    grid = grid.view(3, -1).unsqueeze(0).expand(B, -1, -1)
    cam_ray = torch.bmm(K_inv, grid)
    cam_pts = cam_ray * depth.view(B, 1, -1)
    cam_pts_h = torch.cat([cam_pts, torch.ones_like(cam_pts[:, :1])], dim=1)
    world = torch.bmm(extrinsic_inv, cam_pts_h)[:, :3]
    return world.view(B, 3, H, W)


def project_world_to_pixel(
    points_world: torch.Tensor,
    K: torch.Tensor,
    extrinsic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = points_world.shape[0]
    N = points_world.shape[-1]
    pts_h = torch.cat([points_world, torch.ones_like(points_world[:, :1])], dim=1)
    cam = torch.bmm(extrinsic, pts_h)[:, :3]
    z = cam[:, 2:3]
    uv_h = torch.bmm(K, cam)
    uv = uv_h[:, :2] / torch.clamp(uv_h[:, 2:3], min=1e-6)
    return uv, z.view(B, N)


def homography_warp_features(
    src_features: torch.Tensor,
    K_ref: torch.Tensor,
    K_src: torch.Tensor,
    E_ref: torch.Tensor,
    E_src: torch.Tensor,
    depth_hypos: torch.Tensor,
    feature_stride: int,
    return_valid: bool = False,
    return_geom: bool = False,
) -> torch.Tensor:
    """``return_valid=True`` 时额外返回逐 (source, 假设, 像素) 的几何有效 mask。

    W3-A: 这两个量 (z_src 和 uv) 本来就算出来了然后被扔掉; 出画幅的位置采到 0
    之后照样进源视图平均, 系统性压低边界和被遮挡像素的相关性。默认 False 是
    为了不改变任何现有调用方的行为。

    ``return_geom=True`` (蕴含 return_valid) 再多返回 ``(uv_img, z_src)``,
    W3-B 的遮挡标签要用: ``y_vis = 1[z_src <= D_src_gt(uv_img) + delta]``。

    **uv_img 归一化到整幅图像而不是特征图**。K 在这里是按 ``K_img/stride`` 缩过的,
    所以特征像素坐标 x stride 就是图像像素坐标 —— 精确, 没有半像素偏移。直接拿
    特征网格的归一化坐标去采全分辨率 GT 会差半个 stride, 而遮挡标签恰恰在遮挡
    边界上才有意义, 那正是半个 stride 会取到另一个表面的地方。
    """
    return_valid = return_valid or return_geom
    B, C, H, W = src_features.shape
    D = depth_hypos.shape[1]
    device = src_features.device
    feat_dtype = src_features.dtype
    # Geometry (matrix inverse, projection) is always fp32 for numerical
    # stability; only the feature sampling runs in ``feat_dtype`` (which may be
    # fp16 to save memory on the [B*D, C, H, W] warp tensor). Autocast MUST be
    # disabled here: under fp16 autocast the projection K @ cam (focal * depth)
    # overflows fp16's ~65504 range and produces inf/NaN.
    geo_dtype = torch.float32

    with torch.autocast(device_type=device.type, enabled=False):
        K_ref_s = K_ref.to(geo_dtype).clone()
        K_src_s = K_src.to(geo_dtype).clone()
        K_ref_s[:, 0, :] = K_ref_s[:, 0, :] / feature_stride
        K_ref_s[:, 1, :] = K_ref_s[:, 1, :] / feature_stride
        K_src_s[:, 0, :] = K_src_s[:, 0, :] / feature_stride
        K_src_s[:, 1, :] = K_src_s[:, 1, :] / feature_stride

        E_ref_g = E_ref.to(geo_dtype)
        E_src_g = E_src.to(geo_dtype)
        R_ref = E_ref_g[:, :3, :3]
        t_ref = E_ref_g[:, :3, 3:4]
        R_src = E_src_g[:, :3, :3]
        t_src = E_src_g[:, :3, 3:4]

        R_src_inv = R_src.transpose(1, 2)
        R_rel = R_ref @ R_src_inv
        t_rel = t_ref - R_rel @ t_src

        grid = make_pixel_grid(H, W, device, geo_dtype).view(3, -1).unsqueeze(0).expand(B, -1, -1)
        K_ref_inv = torch.inverse(K_ref_s)
        rays = torch.bmm(K_ref_inv, grid)

        rays_d = rays.unsqueeze(1) * depth_hypos.to(geo_dtype).view(B, D, 1, H * W)
        rays_d = rays_d.reshape(B * D, 3, H * W)

        R_rel_d = R_rel.unsqueeze(1).expand(B, D, 3, 3).reshape(B * D, 3, 3)
        t_rel_d = t_rel.unsqueeze(1).expand(B, D, 3, 1).reshape(B * D, 3, 1)
        cam_src = torch.bmm(R_rel_d.transpose(1, 2), rays_d - t_rel_d)
        K_src_d = K_src_s.unsqueeze(1).expand(B, D, 3, 3).reshape(B * D, 3, 3)
        pix_src = torch.bmm(K_src_d, cam_src)
        z_src = pix_src[:, 2:3].clamp(min=1e-6)
        uv = pix_src[:, :2] / z_src

        uv_x = uv[:, 0] / (W - 1) * 2.0 - 1.0
        uv_y = uv[:, 1] / (H - 1) * 2.0 - 1.0
        valid = None
        if return_valid:
            # 在 nan_to_num / clamp 之前算 —— clamp 之后出界的坐标会被压回 ±2,
            # 分不出 "刚好在边上" 和 "远在画幅外"。z 用未 clamp 的原始值。
            z_raw = pix_src[:, 2]
            valid = ((z_raw > 1e-6)
                     & (uv_x.abs() <= 1.0) & (uv_y.abs() <= 1.0)
                     & torch.isfinite(uv_x) & torch.isfinite(uv_y) & torch.isfinite(z_raw))
            valid = valid.view(B, D, H, W)
        geom = None
        if return_geom:
            # 特征像素 -> 图像像素 -> 整幅图像的归一化坐标 (align_corners=True)
            gx = (uv[:, 0] * feature_stride) / max(W * feature_stride - 1, 1) * 2.0 - 1.0
            gy = (uv[:, 1] * feature_stride) / max(H * feature_stride - 1, 1) * 2.0 - 1.0
            geom = (torch.stack([gx, gy], dim=-1).view(B, D, H, W, 2),
                    z_src.view(B, D, H, W))
        grid_sample = torch.stack([uv_x, uv_y], dim=-1)
        # Sanitize before the (possibly fp16) cast: pixels behind the camera or far
        # out of frustum produce huge / non-finite coords that overflow fp16 and turn
        # grid_sample's output into NaN. Anything with |coord| > 1 is out of bounds
        # anyway (padding_mode="zeros" -> 0), so mapping bad values to +/-2 is safe.
        grid_sample = torch.nan_to_num(grid_sample, nan=2.0, posinf=2.0, neginf=-2.0)
        grid_sample = grid_sample.clamp(-2.0, 2.0).view(B * D, H, W, 2).to(feat_dtype)

        src_features_d = src_features.unsqueeze(1).expand(B, D, C, H, W).reshape(B * D, C, H, W)
        warped = F.grid_sample(
            src_features_d,
            grid_sample,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
    out = warped.view(B, D, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
    if return_geom:
        return out, valid, geom[0], geom[1]
    if return_valid:
        return out, valid
    return out






def reproject_with_depth(
    depth_ref: torch.Tensor,
    K_ref: torch.Tensor,
    E_ref: torch.Tensor,
    K_src: torch.Tensor,
    E_src: torch.Tensor,
) -> torch.Tensor:
    B, H, W = depth_ref.shape
    E_ref_inv = torch.inverse(E_ref)
    world = unproject_depth(depth_ref, torch.inverse(K_ref), E_ref_inv)
    world_flat = world.view(B, 3, -1)
    uv, _ = project_world_to_pixel(world_flat, K_src, E_src)
    return uv.view(B, 2, H, W)






def make_depth_hypotheses_global(
    depth_min: torch.Tensor,
    depth_max: torch.Tensor,
    num_depths: int,
    h: int,
    w: int,
) -> torch.Tensor:
    B = depth_min.shape[0]
    device = depth_min.device
    dtype = depth_min.dtype
    steps = torch.linspace(0.0, 1.0, num_depths, device=device, dtype=dtype)
    span = (depth_max - depth_min).view(B, 1, 1, 1)
    base = depth_min.view(B, 1, 1, 1)
    return (base + span * steps.view(1, num_depths, 1, 1)).expand(B, num_depths, h, w).contiguous()
