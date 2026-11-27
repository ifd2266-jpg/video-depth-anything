"""Physically motivated defocus rendering in linear light.

CoC model: for a thin lens the blur-disk diameter is proportional to |1/z - 1/z_f|, i.e. linear in
*disparity* difference -- exactly what Video Depth Anything predicts (up to an affine transform
whose shift cancels). So   coc = K * max(|d - d_f| - zone/2, 0)   with separate gains for the near
and far field.

Rendering: the signed-CoC range is quantised into fine depth layers. Layers are blurred with an
anti-aliased circular aperture (premultiplied colour + coverage, so each layer only spreads its own
pixels) and composited strictly back-to-front with the "over" operator:
  * background layers are blurred only with background pixels -> the sharp subject never bleeds
    outward, and the subject is composited on top -> background never leaks onto the subject;
  * accumulated coverage is renormalised -> no dark halos where the subject occluded the background;
  * near-field layers are composited last with their blurred coverage -> soft foreground edges
    correctly spread over the in-focus subject, as a real lens does.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

GAMMA = 2.2


def srgb_to_linear(x):
    return x.clamp(0, 1) ** GAMMA


def linear_to_srgb(x):
    return x.clamp(0, 1) ** (1.0 / GAMMA)


def compute_coc(disp, z_focus, max_blur, dof_zone, fg_scale, falloff=1.0, back_extent=None):
    """Signed CoC radius in output pixels: negative = background (farther), positive = near field.
    `back_extent` (optional) widens the far half of the in-focus zone so the whole head stays sharp."""
    h = dof_zone * 0.5
    hb = max(h, back_extent) if back_extent is not None else h
    s = disp - z_focus
    far_excess = (-s - hb).clamp(min=0)
    near_excess = (s - h).clamp(min=0)
    # one lens -> one slope K; far field reaches max_blur at the farthest point of the shot
    # far field reaches max_blur at the farthest point of the shot; near field reaches
    # fg_scale * max_blur at the nearest point (independent near/far multipliers)
    far_span = max(z_focus - hb, 0.08)
    near_span = max(1.0 - z_focus - h, 0.08)
    t_far = (far_excess / far_span).clamp(0, 1)
    t_near = (near_excess / near_span).clamp(0, 1)
    if falloff != 1.0:
        t_far = t_far ** falloff
        t_near = t_near ** falloff
    return max_blur * (fg_scale * t_near - t_far)


def _layer_levels(max_r):
    """Radii with ~12% relative spacing (<= ~1.5 px at small radii) -> no visible banding."""
    lv, r = [0.0], 1.0
    while r < max_r:
        lv.append(r)
        r = r + max(1.0, 0.13 * r)
    lv.append(max_r)
    return np.unique(np.round(np.array(lv), 2))


_KCACHE = {}


def disk_kernel(radius, device):
    key = (round(float(radius), 3), str(device))
    if key not in _KCACHE:
        R = int(math.ceil(radius + 1))
        y, x = torch.meshgrid(torch.arange(-R, R + 1, dtype=torch.float32),
                              torch.arange(-R, R + 1, dtype=torch.float32), indexing="ij")
        dist = torch.sqrt(x * x + y * y)
        k = (radius + 0.5 - dist).clamp(0, 1)          # anti-aliased circular aperture
        # very gentle centre weighting -> "creamy" (no onion-ring / soap-bubble rim)
        k = k * (1.0 - 0.12 * (dist / max(radius, 1e-3)).clamp(0, 1) ** 2)
        _KCACHE[key] = (k / k.sum()).to(device)
    return _KCACHE[key]


def _fft_conv(x, k):
    """x: (C,H,W), k: (kh,kw) centred. Zero-padded linear convolution via FFT."""
    C, H, W = x.shape
    kh, kw = k.shape
    ph, pw = H + kh - 1, W + kw - 1
    X = torch.fft.rfft2(x, s=(ph, pw))
    K = torch.fft.rfft2(k, s=(ph, pw))
    y = torch.fft.irfft2(X * K[None], s=(ph, pw))
    oh, ow = kh // 2, kw // 2
    return y[:, oh:oh + H, ow:ow + W]


def _blur_layer(rgba, radius):
    """Blur premultiplied RGBA with a disk. Large radii run at reduced resolution (the result is
    smooth anyway), then upsample."""
    if radius < 0.5:
        return rgba
    s = 1 if radius < 6 else (2 if radius < 14 else 4)
    H, W = rgba.shape[-2:]
    x = rgba if s == 1 else F.avg_pool2d(rgba[None], s, ceil_mode=True)[0]
    y = _fft_conv(x, disk_kernel(radius / s, x.device))
    if s != 1:
        y = F.interpolate(y[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    return y


def render_bokeh(rgb_lin, coc, highlight_boost=0.6, highlight_thresh=0.85):
    """rgb_lin: (3,H,W) linear; coc: (H,W) signed radius. Returns (3,H,W) linear (may exceed 1)."""
    absr = coc.abs()
    blurw = ((absr - 0.5) / 3.0).clamp(0, 1)               # 0 in focus -> 1 when clearly defocused

    # emulate scene-referred highlights that a display-referred clip lost: bright pixels in
    # defocused regions are boosted before the aperture spreads them -> luminous bokeh disks.
    if highlight_boost > 0:
        lum = 0.2126 * rgb_lin[0] + 0.7152 * rgb_lin[1] + 0.0722 * rgb_lin[2]
        t = ((lum - highlight_thresh) / (1.0 - highlight_thresh)).clamp(0, 1)
        gain = 1.0 + highlight_boost * 3.0 * (t * t) * blurweight_curve(blurw)
        rgb_lin = rgb_lin * gain[None]

    max_r = float(absr.max())
    if max_r < 0.5:
        return rgb_lin
    lv = _layer_levels(max_r)
    signed = np.concatenate([-lv[::-1][:-1], lv])           # far ... 0 ... near (back -> front)
    signed = signed[(signed >= float(coc.min()) - 1) & (signed <= float(coc.max()) + 1)]
    edges = (signed[:-1] + signed[1:]) / 2.0

    out = torch.zeros_like(rgb_lin)
    acc = torch.zeros_like(coc)
    bins = torch.bucketize(coc.contiguous(), torch.tensor(edges, dtype=coc.dtype, device=coc.device))
    for li, r in enumerate(signed):
        m = (bins == li).to(rgb_lin.dtype)
        if m.sum() < 1:
            continue
        rgba = torch.cat([rgb_lin * m[None], m[None]], 0)
        b = _blur_layer(rgba, abs(float(r)))
        a = b[3].clamp(0, 1)
        col = b[:3]                                         # premultiplied
        out = col + (1 - a)[None] * out
        acc = a + (1 - a) * acc
    return out / acc.clamp(min=1e-4)[None]


def blurweight_curve(w):
    return w * w * (3 - 2 * w)


def bloom(rgb_lin, coc, amount, thresh=0.9):
    """Subtle optical halation from bright defocused regions (masked so the subject stays crisp)."""
    if amount <= 0:
        return rgb_lin
    H, W = rgb_lin.shape[-2:]
    lum = 0.2126 * rgb_lin[0] + 0.7152 * rgb_lin[1] + 0.0722 * rgb_lin[2]
    mask = blurweight_curve(((coc.abs() - 1.0) / 4.0).clamp(0, 1))
    bright = (rgb_lin * ((lum - thresh).clamp(min=0) / (lum.clamp(min=1e-4)))[None]) * mask[None]
    s = 8
    small = F.avg_pool2d(bright[None], s, ceil_mode=True)
    sigma = 0.012 * max(H, W) / s
    ks = int(sigma * 3) * 2 + 1
    g = torch.exp(-((torch.arange(ks, device=rgb_lin.device) - ks // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(rgb_lin.dtype)
    small = F.conv2d(F.pad(small, (ks // 2,) * 4, mode="replicate"), g.view(1, 1, 1, ks).expand(3, 1, 1, ks), groups=3)
    small = F.conv2d(small, g.view(1, 1, ks, 1).expand(3, 1, ks, 1), groups=3)
    glow = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)[0]
    return rgb_lin + amount * 0.35 * glow


def soft_clip(x, knee=0.9):
    """Highlight shoulder: values above `knee` roll off smoothly toward 1 instead of hard clipping."""
    over = (x - knee).clamp(min=0)
    span = 1.0 - knee
    return torch.where(x > knee, knee + span * (1 - torch.exp(-over / span)), x)


def estimate_grain(rgb_u8_frames):
    """Robust luma noise sigma (in [0,1] sRGB units) from flat regions of a few frames."""
    import cv2
    sig = []
    for f in rgb_u8_frames:
        y = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        hp = y - cv2.GaussianBlur(y, (0, 0), 1.2)
        gx = cv2.Sobel(cv2.GaussianBlur(y, (0, 0), 2.0), cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(cv2.GaussianBlur(y, (0, 0), 2.0), cv2.CV_32F, 0, 1)
        g = np.sqrt(gx * gx + gy * gy)
        flat = (g < np.percentile(g, 30)) & (y > 0.08) & (y < 0.92)
        if flat.sum() > 1000:
            sig.append(1.4826 * np.median(np.abs(hp[flat])))
    # heavily denoised/compressed sources measure ~0; keep a floor so blurred areas still carry
    # a fine texture that survives encoding instead of looking plastic
    return max(float(np.median(sig)) if sig else 0.0, 0.9 / 255.0)


def add_grain(rgb_srgb, coc, sigma, amount, generator):
    """Re-inject luminance grain where the aperture blur removed the camera's native texture."""
    if amount <= 0 or sigma <= 0:
        return rgb_srgb
    H, W = coc.shape
    n = torch.randn((1, 1, H, W), generator=generator, device="cpu").to(coc.device)
    k = torch.tensor([0.25, 0.5, 0.25], device=coc.device)
    n = F.conv2d(F.pad(n, (1, 1, 1, 1), mode="reflect"), (k[:, None] * k[None, :])[None, None])[0, 0]
    n = n / n.std().clamp(min=1e-6)
    w = blurweight_curve(((coc.abs() - 0.5) / 3.0).clamp(0, 1))
    # grain is perceptually weaker in deep shadows/clipped highlights
    y = 0.2126 * rgb_srgb[0] + 0.7152 * rgb_srgb[1] + 0.0722 * rgb_srgb[2]
    tone = (4 * y * (1 - y)).clamp(0.25, 1)
    return rgb_srgb + (n * sigma * amount * w * tone)[None]
