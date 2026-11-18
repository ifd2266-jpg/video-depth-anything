"""Cinematic shallow depth-of-field (full-frame f/1.4 look) from Video Depth Anything.

Inputs
  --frames   folder of PNG frames (001.png ...)
  --depth    .npz from run.py --save_npz (relative disparity, larger = closer)
  --focus    .npz with key "focus" (T,H,W) float mask of what must be tack sharp
             (e.g. hand + product), optional "subject" (T,H,W) mask of the whole
             foreground object (hand + arm) used to snap depth edges.
Output
  folder of rendered PNG frames.

Rendering model (per frame)
  * disparity is snapped to the segmentation so depth never leaks across
    object edges (no sharp halo around the subject, no blur eating the edge)
  * circle-of-confusion  c = Rmax * (d - d_focus) / d_focus   (signed, px)
  * background (c<0): occlusion-aware gather blur that only uses background
    pixels, so the subject never smears into the bokeh
  * in-focus subject composited on top with its own matte
  * foreground (c>0): premultiplied scatter layers composited over everything,
    so near objects (sleeve, lap) spread over the subject like a real lens
  * blur is done in linear light with highlight boost => bright bokeh discs
  * sensor grain re-applied to blurred areas so they match the sharp areas
"""
import argparse
import glob
import os

import cv2
import numpy as np


def disc_kernel(r):
    if r < 0.5:
        k = np.zeros((1, 1), np.float32)
        k[0, 0] = 1
        return k
    R = int(np.ceil(r)) + 1
    y, x = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    dist = np.sqrt(x * x + y * y)
    k = np.clip(r + 0.5 - dist, 0, 1)  # anti-aliased disc
    # very slight rim brightening, typical of fast primes
    k *= 1.0 + 0.12 * np.clip(dist / max(r, 1e-3), 0, 1) ** 4
    return (k / k.sum()).astype(np.float32)


def blur(x, r, kcache):
    if r < 0.5:
        return x.copy()
    if r not in kcache:
        kcache[r] = disc_kernel(r)
    return cv2.filter2D(x, -1, kcache[r], borderType=cv2.BORDER_REFLECT)


def norm_conv(val, w, sigma):
    """normalized convolution: fill val using only pixels with weight w"""
    num = cv2.GaussianBlur(val * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-4), den


def snap_depth(d, subject, sigma=6):
    """Make disparity respect the subject silhouette."""
    hard = (subject > 0.5).astype(np.float32)
    inner = cv2.erode(hard, np.ones((7, 7), np.uint8))
    outer = 1 - cv2.dilate(hard, np.ones((25, 25), np.uint8))
    d_in, _ = norm_conv(d, inner, sigma)
    # fill background behind the subject progressively (large holes)
    d_out = d.copy()
    w = outer
    for s in (sigma, sigma * 4, sigma * 16):
        f, den = norm_conv(d, w, s)
        fillmask = (w < 0.5) & (den > 1e-3)
        d_out = np.where(fillmask, f, d_out)
        w = np.maximum(w, fillmask.astype(np.float32))
    a = cv2.GaussianBlur(hard, (0, 0), 1.0)
    return a * d_in + (1 - a) * d_out


def scatter(out, lin_b, rad, lv, kcache, far_to_near):
    """Composite soft radius-bins of `rad` (>0 = belongs to layer) over `out`.
    far_to_near=True: larger radius first (background, far is more blurred).
    far_to_near=False: smaller radius first (foreground, nearer is more blurred)."""
    bins = []
    for i in range(len(lv)):
        r = lv[i]
        lo_r = lv[i - 1] if i > 0 else -1.0
        hi_r = lv[i + 1] if i + 1 < len(lv) else 1e9
        if i == 0:
            w_up = (rad > 0) & (rad <= r)
            w_up = w_up.astype(np.float32)
        else:
            w_up = np.clip((rad - lo_r) / (r - lo_r), 0, 1) * (rad > lo_r) * (rad <= r)
        w_dn = np.clip((hi_r - rad) / (hi_r - r), 0, 1) * (rad > r) * (rad < hi_r)
        m = ((w_up + w_dn) * (rad > 0)).astype(np.float32)
        bins.append((r, m))
    order = bins[::-1] if far_to_near else bins
    for r, m in order:
        if m.max() < 1e-4:
            continue
        col = blur(lin_b * m[..., None], r, kcache)
        al = np.clip(blur(m, r, kcache), 0, 1)
        out = out * (1 - al[..., None]) + col
    return out


def clean_silhouette(mask, open_px=6):
    """Remove thin spikes from a soft mask, keeping its anti-aliased edge."""
    hard = (mask > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1,) * 2)
    smooth = cv2.morphologyEx(hard, cv2.MORPH_OPEN, k)
    limit = np.clip(cv2.GaussianBlur(smooth.astype(np.float32), (0, 0), 1.0) * 1.6, 0, 1)
    return np.minimum(mask, limit).astype(np.float32)


def refine_matte(mask, img, out_px=6, in_px=3, sigma=8.0, open_px=10):
    """Snap a segmentation mask to the real object edge with a local colour model.

    Near the boundary every pixel is explained as a mix of the local foreground
    colour F (from inside the mask) and local background colour B (from outside):
    alpha = <I-B, F-B> / |F-B|^2. Recovers thin edge strips the segmenter missed
    (e.g. a bright bezel or a fingertip) and removes background it wrongly took.
    """
    hard = (mask > 0.5).astype(np.uint8)
    inner = cv2.erode(hard, np.ones((2 * in_px + 1,) * 2, np.uint8)).astype(np.float32)
    outer = (1 - cv2.dilate(hard, np.ones((2 * out_px + 1,) * 2, np.uint8))).astype(np.float32)
    band = (inner == 0) & (outer == 0)
    F, wf = norm_conv(img, inner[..., None].repeat(3, 2), sigma * 2)
    B, wb = norm_conv(img, outer[..., None].repeat(3, 2), sigma * 2)
    wf, wb = wf[..., 0], wb[..., 0]
    FB = F - B
    den = (FB * FB).sum(2)
    a = ((img - B) * FB).sum(2) / np.maximum(den, 1e-6)
    a = np.clip(a, 0, 1)
    # trust the colour model only where F and B are clearly different and known
    conf = np.clip((np.sqrt(den) - 0.06) / 0.10, 0, 1) * (wf > 0.02) * (wb > 0.02)
    alpha = np.where(band, conf * a + (1 - conf) * mask, mask).astype(np.float32)
    # growth outside the segmenter's mask may only form a smooth extension of
    # the silhouette: thin protrusions (background lines that happen to match
    # the subject colour) are removed by an opening of the whole silhouette
    grown = (alpha > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1,) * 2)
    smooth = cv2.morphologyEx(grown, cv2.MORPH_OPEN, k)
    # applied to the whole silhouette (the segmenter's own mask can carry
    # such spikes too); keep a 2px margin so the soft anti-aliased edge stays
    limit = np.clip(cv2.GaussianBlur(smooth.astype(np.float32), (0, 0), 0.8) * 1.3, 0, 1)
    alpha = np.minimum(alpha, limit).astype(np.float32)
    # keep only what is connected to the object (no stray specks)
    n, lab = cv2.connectedComponents((alpha > 0.5).astype(np.uint8))
    if n > 2:
        keep = np.zeros(n, bool)
        ids = np.unique(lab[inner > 0])
        keep[ids[ids > 0]] = True
        alpha = alpha * keep[lab]
    return np.clip(cv2.GaussianBlur(alpha, (0, 0), 0.7), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True)
    ap.add_argument('--depth', required=True)
    ap.add_argument('--focus', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rmax', type=float, default=58.0, help='blur radius (px) at infinity')
    ap.add_argument('--fg_max', type=float, default=70.0)
    ap.add_argument('--grain', type=float, default=1.0)
    ap.add_argument('--focus_ids', type=int, nargs='*', default=[1, 2], help='object ids that are in focus')
    ap.add_argument('--refine_ids', type=int, nargs='*', default=[1], help='objects refined with colour matting (others only de-spiked)')
    ap.add_argument('--refine', type=int, default=1, help='snap masks to edges with local colour matting')
    ap.add_argument('--ramp', type=float, default=140.0, help='px over which the arm goes out of focus')
    ap.add_argument('--only', type=int, nargs='*', default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.frames, '*.png')))
    depths = np.load(args.depth)['depths'].astype(np.float32)
    fz = np.load(args.focus)
    focus_masks = fz['focus'].astype(np.float32)
    subj_masks = fz['subject'].astype(np.float32) if 'subject' in fz else focus_masks
    po, obj_by_id = None, {}
    if 'per_object' in fz and 'ids' in fz:
        # arm / sleeve objects: temporal median over 5 frames so a one-frame
        # segmentation dropout (e.g. on a prompt frame) can't make blur pop
        ids = list(fz['ids'])
        po = fz['per_object']
        obj_by_id = {int(oid): i for i, oid in enumerate(ids)}
        arm_idx = [i for i, oid in enumerate(ids) if int(oid) not in args.focus_ids]
        if arm_idx:
            arm = po[arm_idx].astype(np.float32).max(0)
            Tm = arm.shape[0]
            arm_med = np.stack([np.median(arm[max(0, t - 2):min(Tm, t + 3)], axis=0) for t in range(Tm)])
            subj_masks = np.maximum(focus_masks, arm_med)
    T = len(files)
    H, W = cv2.imread(files[0]).shape[:2]
    os.makedirs(args.out, exist_ok=True)

    # global disparity normalisation (VDA is temporally consistent)
    lo, hi = np.percentile(depths, 0.5), np.percentile(depths, 99.5)

    def up(m):
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR) if m.shape != (H, W) else m

    # focus distance per frame (disparity at the hand), temporally smoothed
    dfs = []
    for t in range(T):
        d = (up(depths[t]) - lo) / (hi - lo)
        fm = up(focus_masks[t]) > 0.5
        dfs.append(np.median(d[fm]) if fm.sum() > 100 else np.nan)
    dfs = np.array(dfs)
    if np.isnan(dfs).all():
        raise SystemExit('focus mask empty')
    idx = np.arange(T)
    dfs = np.interp(idx, idx[~np.isnan(dfs)], dfs[~np.isnan(dfs)])
    dfs = cv2.GaussianBlur(dfs.reshape(-1, 1).astype(np.float32), (1, 0), 2.0).ravel() if T > 3 else dfs
    print('focus disparity range', dfs.min(), dfs.max())

    levels = [1.5, 3, 5, 8, 12, 17, 23, 31, 41, 53, 66, 80]
    kcache = {}
    rng = np.random.default_rng(7)
    frames_to_do = args.only if args.only else range(T)

    for t in frames_to_do:
        img8 = cv2.imread(files[t])
        img = img8.astype(np.float32) / 255.0
        lin = img ** 2.2
        lum = lin @ np.array([0.0722, 0.7152, 0.2126], np.float32)
        # highlight boost so specular points become bright bokeh discs
        # only small specular points (much brighter than their surroundings),
        # never large bright areas like the sky -> no blown-out background
        local = cv2.GaussianBlur(lum, (0, 0), 25)
        spec = np.clip((lum - local - 0.12) / 0.3, 0, 1) * np.clip((lum - 0.45) / 0.3, 0, 1)
        subj = up(subj_masks[t])
        foc = up(focus_masks[t])
        if args.refine:
            if po is not None and args.refine_ids:
                # colour-matting refinement only for hard-edged objects (the
                # device); skin edges from the segmenter are already accurate
                # and skin-like background (tan trim) would be pulled in
                parts = []
                for oid in args.focus_ids:
                    if oid not in obj_by_id:
                        continue
                    m = up(po[obj_by_id[oid], t].astype(np.float32))
                    parts.append(refine_matte(m, img) if oid in args.refine_ids else clean_silhouette(m))
                foc = np.max(parts, 0)
                # fill thin gaps where separately segmented objects touch
                # (fingertip against the device edge) without growing outward
                hard = (foc > 0.5).astype(np.uint8)
                kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                gap = (cv2.morphologyEx(hard, cv2.MORPH_CLOSE, kc) > 0) & (hard == 0)
                foc = np.where(gap, 1.0, foc).astype(np.float32)
                foc = np.maximum(foc, cv2.GaussianBlur(gap.astype(np.float32), (0, 0), 0.7))
            else:
                foc = refine_matte(foc, img)
            subj = np.maximum(subj, foc)
        # bright pixels hugging the subject are its own edge, never bokeh points
        near = cv2.dilate((foc > 0.3).astype(np.uint8), np.ones((61, 61), np.uint8)).astype(np.float32)
        spec = spec * (1 - cv2.GaussianBlur(near, (0, 0), 8))
        boost = 1.0 + 3.0 * spec ** 1.5
        lin_b = lin * boost[..., None]
        d = (up(depths[t]) - lo) / (hi - lo)
        d = snap_depth(d, subj)
        df = dfs[t]
        c = args.rmax * (d - df) / max(df, 1e-3)
        c = np.clip(c, -args.rmax * 1.05, args.fg_max)
        # what is in the focus mask is tack sharp; soft falloff along depth elsewhere
        foc_soft = cv2.GaussianBlur(foc, (0, 0), 0.8)
        # along the arm (connected to the hand) blur must grow gradually from
        # the wrist, never jump at the mask edge
        dist = cv2.distanceTransform((foc < 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        ramp = np.clip(dist / args.ramp, 0, 1)
        ramp = ramp * ramp * (3 - 2 * ramp)
        arm = (subj > 0.5) & (foc < 0.5)
        c = np.where(arm, c * ramp, c)
        c = c * (1 - foc_soft)
        c[np.abs(c) < 0.6] = 0.0

        # ---------- background: occlusion-aware gather ----------
        bg_w = ((c < -0.5) & (foc_soft < 0.5)).astype(np.float32)
        r_pix = np.abs(np.minimum(c, 0))
        bg_stack = []
        for r in levels:
            # only pixels that are themselves at least ~this blurry contribute
            wr = bg_w * np.clip((r_pix - 0.45 * r) / (0.25 * r + 1e-3), 0, 1)
            wr = np.maximum(wr, bg_w * 1e-3)
            num = blur(lin_b * wr[..., None], r, kcache)
            den = blur(wr, r, kcache)
            bg_stack.append(num / np.maximum(den, 1e-6)[..., None])
        # per-pixel interpolation across levels
        all_r = np.array([0.0] + levels, np.float32)
        stack = [lin_b] + bg_stack
        bg = np.zeros_like(lin)
        rp = np.clip(r_pix, 0, all_r[-1])
        for i in range(len(all_r) - 1):
            a, b = all_r[i], all_r[i + 1]
            wa = np.clip((b - rp) / (b - a), 0, 1) * (rp >= a) * (rp <= b)
            wb = np.clip((rp - a) / (b - a), 0, 1) * (rp > a) * (rp <= b)
            bg += stack[i] * wa[..., None] + stack[i + 1] * wb[..., None]
        # for pixels outside the background (subject / fg), fill with widest-scale bg
        not_bg = (r_pix < 1e-6)[..., None]
        bg = np.where(not_bg, bg_stack[2], bg)
        # foreground pixels sit on top of themselves (opaque interior);
        # their spread outside the silhouette is added by the scatter layers
        bg = np.where((c > 0)[..., None], lin_b, bg)
        # every background depth layer spreads its own blurred edge over what is
        # behind it (gather result above is the hole-free plate underneath)
        bg = scatter(bg, lin_b, r_pix * bg_w, [0.0] + levels, kcache, far_to_near=True)

        # ---------- in-focus layer ----------
        sharp_a = np.where(c > 0, np.clip(1 - c / 1.5, 0, 1), np.clip(1 - np.abs(c) / 1.2, 0, 1))
        sharp_a = np.maximum(sharp_a, foc_soft)
        out = bg * (1 - sharp_a[..., None]) + lin * sharp_a[..., None]

        # ---------- foreground: premultiplied scatter, far -> near ----------
        out = scatter(out, lin_b, np.maximum(c, 0), [0.0] + [l for l in levels if l <= args.fg_max + 15],
                      kcache, far_to_near=False)

        # undo highlight boost (soft, keeps bokeh balls bright but not clipped)
        out = np.clip(out, 0, None)
        knee = 0.97
        out = np.where(out > knee, knee + (1 - knee) * np.tanh((out - knee) / (1 - knee)), out)
        res = np.clip(out, 0, 1) ** (1 / 2.2)

        # ---------- grain matching ----------
        if args.grain > 0:
            g = img - cv2.GaussianBlur(img, (0, 0), 1.2)
            sigma = float(np.median(np.abs(g[sharp_a > 0.5]))) * 1.48 if (sharp_a > 0.5).any() else 0.01
            sigma = np.clip(sigma, 0.004, 0.03) * args.grain
            n = rng.normal(0, 1, (H, W)).astype(np.float32)
            n = cv2.GaussianBlur(n, (0, 0), 0.6)
            n = n / (n.std() + 1e-6) * sigma
            blurred_amt = np.clip(np.abs(c) / 3.0, 0, 1) * (1 - sharp_a)
            res = res + (n * blurred_amt)[..., None]

        cv2.imwrite(os.path.join(args.out, f'{t + 1:03d}.png'),
                    np.clip(res * 255 + 0.5, 0, 255).astype(np.uint8))
        print('frame', t + 1, '/', T, flush=True)


if __name__ == '__main__':
    main()
