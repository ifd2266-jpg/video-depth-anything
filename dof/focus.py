"""Face-locked focus: YuNet face detection, main-subject tracking, robust z_focus sampling and
One-Euro temporal filtering. All state is per shot."""
import math

import cv2
import numpy as np


class FaceDetector:
    def __init__(self, model_path, det_width=540, score=0.6):
        self.model_path, self.det_width, self.score = model_path, det_width, score
        self.net = None

    def __call__(self, rgb):
        H, W = rgb.shape[:2]
        s = self.det_width / W
        w, h = self.det_width, int(round(H * s))
        if self.net is None:
            self.net = cv2.FaceDetectorYN.create(self.model_path, "", (w, h), self.score, 0.3, 50)
        self.net.setInputSize((w, h))
        bgr = cv2.cvtColor(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        _, faces = self.net.detect(bgr)
        out = []
        if faces is not None:
            for f in faces:
                x, y, bw, bh = f[:4] / s
                out.append((float(x), float(y), float(bw), float(bh), float(f[-1])))
        return out


def _iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-6)


def track_subject(detections, frame_shape):
    """Pick the main facing subject per frame and keep it locked across the shot.

    detections: list (per frame) of [(x,y,w,h,score), ...]. Back-turned people produce no face, so
    they can never become the focus target, even though they are closest to the lens.
    Returns list of boxes (x,y,w,h) or None if the shot contains no face at all.
    """
    H, W = frame_shape[:2]

    def prior(d):  # large, confident, reasonably central faces are the hero
        cx = (d[0] + d[2] / 2) / W
        cy = (d[1] + d[3] / 2) / H
        centre = math.exp(-((cx - 0.55) ** 2) / 0.18 - ((cy - 0.4) ** 2) / 0.5)
        return d[4] * math.sqrt(d[2] * d[3]) * centre

    # seed: strongest detection over the whole shot (robust to a bad first frame)
    best = max(((prior(d), i, d) for i, ds in enumerate(detections) for d in ds), default=None)
    if best is None:
        return [None] * len(detections)
    _, seed_i, seed = best
    boxes = [None] * len(detections)
    boxes[seed_i] = seed[:4]

    def sweep(order):
        prev = seed[:4]
        for i in order:
            cands = detections[i]
            pick = None
            if cands:
                scored = [(prior(d) * (0.25 + 2.0 * _iou(prev, d)), d) for d in cands]
                s, d = max(scored, key=lambda t: t[0])
                # reject jumps to a different (usually smaller, background) face
                if _iou(prev, d) > 0.05 or d[2] * d[3] > 0.6 * prev[2] * prev[3]:
                    pick = d[:4]
            prev = pick if pick is not None else prev   # hold last box on a miss
            boxes[i] = prev

    sweep(range(seed_i + 1, len(detections)))
    sweep(range(seed_i - 1, -1, -1))
    return boxes


def smooth_boxes(boxes, alpha=0.5):
    """Light EMA on box geometry (forward-backward, zero lag) to stabilise the sampling region."""
    if boxes[0] is None:
        return boxes
    b = np.array(boxes, np.float64)
    f = b.copy()
    for i in range(1, len(b)):
        f[i] = alpha * b[i] + (1 - alpha) * f[i - 1]
    r = f.copy()
    for i in range(len(b) - 2, -1, -1):
        r[i] = alpha * f[i] + (1 - alpha) * r[i + 1]
    return [tuple(x) for x in r]


def sample_focus(depth, box, shrink=(0.25, 0.2)):
    """Robust median of refined disparity inside the inner face region (eyes/nose/beard),
    avoiding background pixels at the box borders."""
    H, W = depth.shape
    if box is None:  # no face in the shot: centre-weighted fallback
        y0, y1, x0, x1 = int(H * 0.3), int(H * 0.5), int(W * 0.35), int(W * 0.65)
    else:
        x, y, w, h = box
        x0 = int(np.clip(x + w * shrink[0], 0, W - 1)); x1 = int(np.clip(x + w * (1 - shrink[0]), x0 + 1, W))
        y0 = int(np.clip(y + h * shrink[1], 0, H - 1)); y1 = int(np.clip(y + h * (1 - shrink[1]), y0 + 1, H))
    patch = depth[y0:y1, x0:x1].ravel()
    # trimmed median: drop the farthest 20% (background seen through gaps)
    patch = np.sort(patch)[int(0.2 * len(patch)):]
    return float(np.median(patch))


def head_back_extent(depth, box, z_focus, bg_gap=0.3, pct=10):
    """Disparity of the far side of the subject's head (iqal / headdress) measured in the column
    above the face box. Background is excluded by a disparity gap. Returns None if not measurable."""
    if box is None:
        return None
    H, W = depth.shape
    x, y, w, h = box
    x0, x1 = int(np.clip(x + 0.1 * w, 0, W - 1)), int(np.clip(x + 0.9 * w, 1, W))
    y0, y1 = int(np.clip(y - 0.7 * h, 0, H - 1)), int(np.clip(y + 0.15 * h, 1, H))
    r = depth[y0:y1, x0:x1].ravel()
    r = r[r > z_focus - bg_gap]
    if r.size < 50:
        return None
    return float(np.percentile(r, pct))


class OneEuro:
    """One-Euro filter (Casiez et al. 2012)."""

    def __init__(self, freq, min_cutoff=0.4, beta=0.8, d_cutoff=1.0):
        self.freq, self.min_cutoff, self.beta, self.d_cutoff = freq, min_cutoff, beta, d_cutoff
        self.x_prev = self.dx_prev = None

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * freq)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x, 0.0
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat
