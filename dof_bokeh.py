#!/usr/bin/env python3
"""Cinema-grade shallow depth of field for video (85mm f/1.4 look) on top of Video Depth Anything.

Two-step workflow:
  1) python dof_bokeh.py --input in.mov --output out.mov --preview-only
       -> diagnostic PNGs  [Original | Refined depth + focus box | Final bokeh]
  2) python dof_bokeh.py --input in.mov --output out.mov
       -> full-resolution render, original FPS, BT.709, audio copied (-c:a copy)

Per shot (shots split at hard cuts and black frames) the pipeline resets depth normalisation,
temporal smoothing and focus tracking, so nothing bleeds across a cut.
"""
import argparse
import hashlib
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dof import bokeh as B                                     # noqa: E402
from dof.depth import DepthEstimator, DepthRefiner, shot_normalizer  # noqa: E402
from dof.focus import (FaceDetector, OneEuro, head_back_extent, sample_focus, smooth_boxes,
                       track_subject)  # noqa: E402
from dof.scenes import detect_shots                            # noqa: E402
from dof.video_io import VideoWriter, probe, read_frames       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help="output video path (preview images go next to it)")
    p.add_argument("--max-blur", type=float, default=26.0,
                   help="max CoC radius in px at 1080 px short side (scaled with resolution)")
    p.add_argument("--dof-zone", type=float, default=0.10,
                   help="in-focus deadzone width in normalised disparity units (0-1)")
    p.add_argument("--fg-blur-scale", type=float, default=0.35, help="near-field blur ratio vs far field")
    p.add_argument("--highlight-boost", type=float, default=0.6, help="bokeh highlight boost + bloom (0=off)")
    p.add_argument("--falloff", type=float, default=0.85,
                   help="CoC falloff exponent (1=strictly linear in disparity, <1 = faster onset)")
    p.add_argument("--grain", type=float, default=1.0, help="grain re-injection strength (0=off)")
    p.add_argument("--temporal", type=float, default=0.5, help="flow-aware temporal depth smoothing 0-0.9")
    p.add_argument("--focus-min-cutoff", type=float, default=0.4, help="One-Euro min cutoff (Hz)")
    p.add_argument("--focus-beta", type=float, default=0.8, help="One-Euro speed coefficient")
    p.add_argument("--no-head-lock", action="store_true",
                   help="disable the subject-adaptive far half of the focus zone (head/iqal lock)")
    p.add_argument("--head-margin", type=float, default=0.02, help="extra focus margin behind the head")
    p.add_argument("--preview-only", action="store_true")
    p.add_argument("--preview-times", type=str, default="1.0,8.0",
                   help="comma-separated seconds for preview frames")
    p.add_argument("--encoder", choices=["vits", "vitb", "vitl"], default="vitl")
    p.add_argument("--input-size", type=int, default=518)
    p.add_argument("--fp32", action="store_true", help="disable mixed precision for the depth model")
    p.add_argument("--checkpoints", default=os.path.join(HERE, "checkpoints"))
    p.add_argument("--face-model", default=None, help="YuNet onnx (default checkpoints/face_detection_yunet_2023mar.onnx)")
    p.add_argument("--cache-dir", default=os.path.join(HERE, "outputs", "depth_cache"))
    p.add_argument("--crf", type=int, default=12)
    p.add_argument("--preset", default="slow")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def file_key(path):
    st = os.stat(path)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    h.update(f"{st.st_size}".encode())
    return h.hexdigest()[:16]


class ShotProcessor:
    """Holds all per-shot state. A new instance is created at every cut."""

    def __init__(self, args, device, frames, raw_disp, face_det, fps, scale):
        self.args, self.device, self.frames, self.fps = args, device, frames, fps
        self.norm = shot_normalizer(raw_disp)
        self.raw = raw_disp
        self.refiner = DepthRefiner(device, temporal=args.temporal)
        self.euro = OneEuro(fps, args.focus_min_cutoff, args.focus_beta)
        self.euro_back = OneEuro(fps, args.focus_min_cutoff, args.focus_beta)
        self.back = None
        dets = [face_det(f) for f in frames]
        self.boxes = smooth_boxes(track_subject(dets, frames[0].shape))
        self.n_faces = sum(1 for d in dets if d)
        self.max_blur = args.max_blur * scale
        self.grain_sigma = B.estimate_grain(frames[:: max(1, len(frames) // 4)][:4])
        self.gen = torch.Generator().manual_seed(args.seed)

    def process(self, i):
        """Must be called with consecutive i (temporal state). Returns dict of results."""
        a = self.args
        rgb = self.frames[i]
        disp = self.refiner.refine(rgb, self.norm(self.raw[i]))
        z_raw = sample_focus(disp.cpu().numpy(), self.boxes[i])
        z = self.euro(z_raw)
        back = self._back(disp, i, z)
        coc = B.compute_coc(disp, z, self.max_blur, a.dof_zone, a.fg_blur_scale, a.falloff, back)
        img = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float() / 255.0
        lin = B.srgb_to_linear(img)
        out = B.render_bokeh(lin, coc, a.highlight_boost)
        out = B.bloom(out, coc, a.highlight_boost)
        out = B.linear_to_srgb(B.soft_clip(out))
        out = B.add_grain(out, coc, self.grain_sigma, a.grain, self.gen)
        out_u8 = (out.clamp(0, 1) * 255.0 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
        return dict(out=out_u8, disp=disp, coc=coc, z=z, z_raw=z_raw, box=self.boxes[i], back=back)

    def _back(self, disp, i, z):
        """Smoothed far half-width of the focus zone that keeps the head (iqal, headdress) sharp."""
        if self.args.no_head_lock:
            return None
        e = head_back_extent(disp.cpu().numpy(), self.boxes[i], z)
        if e is not None:
            self.back = float(np.clip(z - e + self.args.head_margin, self.args.dof_zone / 2, 0.3))
        if self.back is None:
            return None
        return self.euro_back(self.back)


def diag_panel(rgb, res, shot_id, t):
    H, W = rgb.shape[:2]
    d = (res["disp"].cpu().numpy() * 255).astype(np.uint8)
    dv = cv2.cvtColor(cv2.applyColorMap(d, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    # outline the in-focus zone (|CoC| < 1 px) in cyan
    infocus = (res["coc"].abs() < 1.0).cpu().numpy().astype(np.uint8)
    cnts, _ = cv2.findContours(infocus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(dv, cnts, -1, (0, 255, 255), 3)
    if res["box"] is not None:
        x, y, w, h = map(int, res["box"])
        cv2.rectangle(dv, (x, y), (x + w, y + h), (0, 255, 0), 5)
    zone = f"zone -{res['back']:.2f}" if res.get("back") is not None else "zone sym"
    lab = [f"shot {shot_id}  t={t:.2f}s", f"z_focus={res['z']:.3f} (raw {res['z_raw']:.3f}) {zone}",
           "green: face box  cyan: in-focus zone"]
    for k, s in enumerate(lab):
        cv2.putText(dv, s, (24, 60 + 52 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(dv, s, (24, 60 + 52 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    sep = np.full((H, 12, 3), 255, np.uint8)
    return np.concatenate([rgb, sep, dv, sep, res["out"]], 1)


def zoom_panel(rgb, res):
    """1:1 crops around the face/foreground boundary to judge edges (original | bokeh)."""
    H, W = rgb.shape[:2]
    if res["box"] is not None:
        x, y, w, h = map(int, res["box"])
        cx, cy = x + w // 2, y + h // 2
    else:
        cx, cy = W // 2, H // 3
    cw, ch = 540, 720
    x0 = int(np.clip(cx - cw * 0.6, 0, W - cw)); y0 = int(np.clip(cy - ch * 0.45, 0, H - ch))
    a = rgb[y0:y0 + ch, x0:x0 + cw]; b = res["out"][y0:y0 + ch, x0:x0 + cw]
    return np.concatenate([a, np.full((ch, 8, 3), 255, np.uint8), b], 1)


def main():
    a = parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    info = probe(a.input)
    fps = info["fps"]
    print(f"[info] {info['width']}x{info['height']} @ {info['fps_str']} fps, audio={info['has_audio']}, device={device}")

    t0 = time.time()
    frames, n = read_frames(a.input, info)
    H, W = frames.shape[1:3]
    scale = min(H, W) / 1080.0
    shots, black = detect_shots(frames)
    print(f"[scenes] {n} frames, {int(black.sum())} black, shots: " +
          ", ".join(f"#{k}:{s}-{e - 1} ({s / fps:.2f}-{(e - 1) / fps:.2f}s)" for k, (s, e) in enumerate(shots)))

    face_model = a.face_model or os.path.join(a.checkpoints, "face_detection_yunet_2023mar.onnx")
    face_det = FaceDetector(face_model)
    est = DepthEstimator(a.encoder, a.checkpoints, device, a.input_size, a.fp32, a.cache_dir)
    vkey = file_key(a.input)
    out_dir = os.path.dirname(os.path.abspath(a.output)) or "."
    os.makedirs(out_dir, exist_ok=True)

    if a.preview_only:
        stem = os.path.splitext(os.path.basename(a.output))[0]
        for t in [float(x) for x in a.preview_times.split(",")]:
            fi = min(int(round(t * fps)), n - 1)
            sid = next((k for k, (s, e) in enumerate(shots) if s <= fi < e), None)
            if sid is None:
                print(f"[preview] t={t}s is a black frame, skipped"); continue
            s, e = shots[sid]
            fi = min(max(fi, s), e - 1)
            # use cached full-shot depth if available, else a local 22-frame window (1 VDA pass)
            key_full = f"{vkey}_{s}_{e}"
            tag = f"{key_full}_{a.encoder}_{a.input_size}_{'fp32' if a.fp32 else 'amp'}"
            cpath = os.path.join(a.cache_dir, hashlib.sha1(tag.encode()).hexdigest()[:16] + ".npy")
            if os.path.exists(cpath):
                ws, we = s, e
                raw = np.load(cpath).astype(np.float32)
                print(f"[preview] shot {sid}: using cached full-shot depth")
            else:
                ws = max(s, fi - 16); we = min(e, ws + 22); ws = max(s, we - 22)
                print(f"[preview] shot {sid}: depth on local window {ws}-{we - 1} ({a.encoder})")
                raw = est.infer(frames[ws:we], fps, cache_key=f"{vkey}_{ws}_{we}")
            proc = ShotProcessor(a, device, frames[ws:we], raw, face_det, fps, scale)
            # warm temporal + focus state on the frames leading to the target
            res = None
            for j in range(0, fi - ws + 1):
                tr = time.time()
                res = proc.process(j) if j == fi - ws else _warm(proc, j)
            print(f"[preview] render time for one frame: {time.time() - tr:.2f}s")
            panel = diag_panel(frames[fi], res, sid, fi / fps)
            p1 = os.path.join(out_dir, f"{stem}_preview_shot{sid}_{fi:04d}.jpg")
            cv2.imwrite(p1, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
            p2 = os.path.join(out_dir, f"{stem}_preview_shot{sid}_{fi:04d}_zoom.png")
            cv2.imwrite(p2, cv2.cvtColor(zoom_panel(frames[fi], res), cv2.COLOR_RGB2BGR))
            print(f"[preview] frame {fi}: z_focus={res['z']:.3f}, faces found in {proc.n_faces}/{we - ws} "
                  f"frames, grain sigma={proc.grain_sigma * 255:.2f}/255 -> {p1}")
        print(f"[done] {time.time() - t0:.1f}s")
        return

    writer = VideoWriter(a.output, W, H, info["fps_str"], a.input if info["has_audio"] else None,
                         a.crf, a.preset, info["color_range"])
    if not info["has_audio"]:
        print("[audio] input has no audio stream -> video-only output")
    shot_at = {s: (k, e) for k, (s, e) in enumerate(shots)}
    log = []
    i = 0
    while i < n:
        if i not in shot_at:              # black / transition frames pass through untouched
            writer.write(frames[i]); i += 1; continue
        sid, e = shot_at[i]
        ts = time.time()
        print(f"[shot {sid}] depth {i}-{e - 1} ...", flush=True)
        raw = est.infer(frames[i:e], fps, cache_key=f"{vkey}_{i}_{e}")
        proc = ShotProcessor(a, device, frames[i:e], raw, face_det, fps, scale)
        print(f"[shot {sid}] depth {time.time() - ts:.0f}s, faces in {proc.n_faces}/{e - i} frames; rendering",
              flush=True)
        for j in range(e - i):
            r = proc.process(j)
            writer.write(r["out"])
            log.append((i + j, sid, r["z_raw"], r["z"]))
            if j % 20 == 0:
                print(f"  frame {i + j}/{n}  z_focus={r['z']:.3f}  ({time.time() - ts:.0f}s)", flush=True)
        del proc, raw
        i = e
    writer.close()
    np.savetxt(os.path.splitext(a.output)[0] + "_focus_log.csv", np.array(log), delimiter=",",
               header="frame,shot,z_raw,z_focus", fmt=["%d", "%d", "%.4f", "%.4f"])
    print(f"[done] {a.output}  ({time.time() - t0:.0f}s)")


def _warm(proc, j):
    """Advance temporal depth + focus filter state without rendering."""
    disp = proc.refiner.refine(proc.frames[j], proc.norm(proc.raw[j]))
    z = proc.euro(sample_focus(disp.cpu().numpy(), proc.boxes[j]))
    proc._back(disp, j, z)
    return None


if __name__ == "__main__":
    main()
