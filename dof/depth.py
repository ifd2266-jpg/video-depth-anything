"""Depth estimation (Video Depth Anything), per-shot normalisation, RGB-guided refinement and
flow-aware temporal stabilisation."""
import hashlib
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from video_depth_anything.video_depth import VideoDepthAnything

MODEL_CFG = {
    "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
    "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
}

CACHE_RES = (960, 540)  # (h, w) storage size for cached raw disparity (model runs at ~924x518)


def _chunk_encoder(model, chunk):
    """Run the (per-frame, non-temporal) DINOv2 encoder in frame chunks to bound peak memory.
    The temporal DPT head still sees the full 32-frame window, so outputs are unchanged."""
    enc = model.pretrained
    orig = enc.get_intermediate_layers

    def chunked(x, *args, **kwargs):
        if x.shape[0] <= chunk:
            return orig(x, *args, **kwargs)
        parts = [orig(x[i:i + chunk], *args, **kwargs) for i in range(0, x.shape[0], chunk)]
        return tuple(
            tuple(torch.cat([p[l][k] for p in parts], 0) for k in range(len(parts[0][l])))
            if isinstance(parts[0][l], (tuple, list)) else torch.cat([p[l] for p in parts], 0)
            for l in range(len(parts[0])))

    enc.get_intermediate_layers = chunked


class DepthEstimator:
    """Wraps VDA's native temporally-consistent video mode. Each shot is inferred independently so
    no temporal context crosses a cut."""

    def __init__(self, encoder, ckpt_dir, device, input_size=518, fp32=False, cache_dir=None):
        self.encoder, self.device, self.input_size = encoder, device, input_size
        self.fp32 = fp32
        self.ckpt = os.path.join(ckpt_dir, f"video_depth_anything_{encoder}.pth")
        self.cache_dir = cache_dir
        self.model = None

    def _load(self):
        if self.model is None:
            m = VideoDepthAnything(**MODEL_CFG[self.encoder])
            m.load_state_dict(torch.load(self.ckpt, map_location="cpu"), strict=True)
            m = m.to(self.device).eval()
            _chunk_encoder(m, chunk=4 if self.device.type == "cpu" else 32)
            self.model = m
        return self.model

    def _cache_path(self, key):
        if not self.cache_dir:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{key}.npy")

    def infer(self, frames, fps, cache_key=None):
        """frames: uint8 RGB (T,H,W,3). Returns raw relative disparity float32 (T, *CACHE_RES)."""
        path = None
        if cache_key:
            tag = f"{cache_key}_{self.encoder}_{self.input_size}_{'fp32' if self.fp32 else 'amp'}"
            path = self._cache_path(hashlib.sha1(tag.encode()).hexdigest()[:16])
            if path and os.path.exists(path):
                d = np.load(path)
                if len(d) == len(frames):
                    return d.astype(np.float32)
        model = self._load()
        dev = self.device.type
        # VDA pads windows internally; it needs >= 1 frame. Autocast: fp16 on CUDA, bf16 on CPU.
        # The network resizes its input to ~518x924 anyway; feeding frames pre-scaled to CACHE_RES
        # (area filter) gives an equivalent input but keeps VDA's per-frame outputs at 540x960
        # instead of full resolution -> ~4x lower peak RAM on long shots.
        small = np.stack([cv2.resize(f, CACHE_RES[::-1], interpolation=cv2.INTER_AREA) for f in frames])
        depth, _ = model.infer_video_depth(small, fps, input_size=self.input_size, device=dev,
                                           fp32=self.fp32)
        if path:
            np.save(path, depth.astype(np.float16))
        return depth.astype(np.float32)


def shot_normalizer(raw, lo_pct=1.0, hi_pct=99.5):
    """Robust per-shot affine normalisation of VDA disparity to [0,1] (1 = nearest).
    VDA disparity is affine-invariant; the shift cancels in |d - d_focus|, so CoC stays physically
    proportional to the disparity difference."""
    sample = raw[:: max(1, len(raw) // 24), ::4, ::4]
    lo, hi = np.percentile(sample, [lo_pct, hi_pct])
    scale = 1.0 / max(hi - lo, 1e-6)
    return lambda d: np.clip((d - lo) * scale, 0.0, 1.0)


# ----------------------------------------------------------------------------------------------
# Fast colour guided filter (He et al. 2010/2015) in torch -> runs on GPU when available.
# ----------------------------------------------------------------------------------------------

def _box(x, r):
    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


def fast_guided_filter(guide, src, radius=6, eps=1e-4, subsample=4):
    """guide: (1,3,H,W) in [0,1]; src: (1,1,H,W). `radius` is in subsampled pixels."""
    H, W = guide.shape[-2:]
    h, w = max(1, H // subsample), max(1, W // subsample)
    I = F.interpolate(guide, size=(h, w), mode="area")
    p = F.interpolate(src, size=(h, w), mode="area")
    r = radius
    mI = _box(I, r)                      # 1,3,h,w
    mp = _box(p, r)                      # 1,1,h,w
    cov_Ip = _box(I * p, r) - mI * mp    # 1,3,h,w
    # 3x3 covariance per pixel
    idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    var = {}
    for a, b in idx:
        var[(a, b)] = _box(I[:, a:a + 1] * I[:, b:b + 1], r) - mI[:, a:a + 1] * mI[:, b:b + 1]
    S = torch.empty(1, h, w, 3, 3, device=guide.device, dtype=guide.dtype)
    for a, b in idx:
        v = var[(a, b)][0, 0] + (eps if a == b else 0.0)
        S[0, :, :, a, b] = v
        S[0, :, :, b, a] = v
    rhs = cov_Ip.permute(0, 2, 3, 1).unsqueeze(-1)              # 1,h,w,3,1
    a = torch.linalg.solve(S, rhs).squeeze(-1).permute(0, 3, 1, 2)  # 1,3,h,w
    b = mp - (a * mI).sum(1, keepdim=True)
    ma = F.interpolate(_box(a, r), size=(H, W), mode="bilinear", align_corners=False)
    mb = F.interpolate(_box(b, r), size=(H, W), mode="bilinear", align_corners=False)
    return (ma * guide).sum(1, keepdim=True) + mb


class DepthRefiner:
    """Per-frame: upsample -> RGB-guided edge snapping (two-scale fast guided filter) ->
    flow-compensated, confidence-masked temporal EMA. Call reset() at every shot boundary."""

    def __init__(self, device, temporal=0.5, gf_eps=1e-3, flow_scale=0.25, near_dilate=3):
        self.device, self.temporal, self.gf_eps, self.flow_scale = device, temporal, gf_eps, flow_scale
        self.near_dilate = near_dilate
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.reset()

    def reset(self):
        self.prev_depth = None
        self.prev_gray = None

    def refine(self, rgb_u8, disp_norm_lowres):
        H, W = rgb_u8.shape[:2]
        guide = torch.from_numpy(rgb_u8).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        d = torch.from_numpy(np.ascontiguousarray(disp_norm_lowres)).to(self.device)[None, None].float()
        d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        # VDA silhouettes are already fairly sharp but sit a few px off the true RGB edge.
        # 1) small near-dilation: silhouette pixels (iqal rim, shemagh fringe, hair) go with the
        #    nearer object, so they are never smeared into the background blur (dark/colour halo);
        # 2) colour guided filter snaps the boundary back onto the RGB edge;
        # 3) the filtered result is used only inside the depth-edge band -> no texture copying of
        #    the shemagh pattern into flat depth regions, no soft ramps into the wall.
        if self.near_dilate > 0:
            k = 2 * int(self.near_dilate * W / 1080.0 + 0.5) + 1
            d = F.max_pool2d(d, k, stride=1, padding=k // 2)
        g = fast_guided_filter(guide, d, radius=6, eps=self.gf_eps, subsample=2)
        s = 4
        ds = F.max_pool2d(d, s)
        kb = 2 * max(1, int(10 * W / 1080.0 / s)) + 1
        rng = F.max_pool2d(ds, kb, stride=1, padding=kb // 2) + F.max_pool2d(-ds, kb, stride=1, padding=kb // 2)
        band = ((rng - 0.03) / 0.06).clamp(0, 1)
        band = F.interpolate(F.avg_pool2d(band, 3, stride=1, padding=1), size=(H, W), mode="bilinear",
                             align_corners=False)
        d = d + band * (g - d)
        d = d.clamp(0, 1)[0, 0]
        d = self._temporal(rgb_u8, d)
        return d

    def _temporal(self, rgb_u8, d):
        if self.temporal <= 0:
            return d
        s = self.flow_scale
        gray = cv2.cvtColor(cv2.resize(rgb_u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                            cv2.COLOR_RGB2GRAY)
        if self.prev_depth is not None:
            # backward flow: for each pixel in current frame, where it was in the previous frame
            flow = self.dis.calc(gray, self.prev_gray, None)
            h, w = gray.shape
            grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            gx = (grid_x + flow[..., 0]) / max(w - 1, 1) * 2 - 1
            gy = (grid_y + flow[..., 1]) / max(h - 1, 1) * 2 - 1
            grid = torch.from_numpy(np.stack([gx, gy], -1)).to(self.device)[None]
            grid = F.interpolate(grid.permute(0, 3, 1, 2), size=d.shape, mode="bilinear",
                                 align_corners=True).permute(0, 2, 3, 1)
            warped = F.grid_sample(self.prev_depth[None, None], grid, mode="bilinear",
                                   padding_mode="border", align_corners=True)[0, 0]
            # photometric check: warped previous frame must match current colours
            prev_g = torch.from_numpy(self.prev_gray).to(self.device).float()[None, None] / 255.0
            cur_g = torch.from_numpy(gray).to(self.device).float() / 255.0
            small_grid = torch.from_numpy(np.stack([gx, gy], -1)).to(self.device)[None]
            warped_g = F.grid_sample(prev_g, small_grid, mode="bilinear", padding_mode="border",
                                     align_corners=True)[0, 0]
            photo = F.interpolate((warped_g - cur_g).abs()[None, None], size=d.shape,
                                  mode="bilinear", align_corners=False)[0, 0]
            conf = torch.exp(-((warped - d) / 0.04) ** 2) * torch.exp(-(photo / 0.06) ** 2)
            w_t = self.temporal * conf
            d = d + w_t * (warped - d)
        self.prev_depth, self.prev_gray = d, gray
        return d
