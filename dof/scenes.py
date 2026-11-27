"""Scene-cut and black-frame detection."""
import cv2
import numpy as np


def _small_gray(f):
    return cv2.cvtColor(cv2.resize(f, (96, 160), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)


def _hist(f):
    hsv = cv2.cvtColor(cv2.resize(f, (96, 160), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()


def detect_shots(frames, black_luma=16.0, black_ratio=0.98, cut_hist=0.55, cut_diff=40.0, min_len=3):
    """Return (shots, black_mask).

    shots: list of (start, end_exclusive) for non-black content segments.
    A cut is declared on a black frame run, or when both the HSV-histogram correlation drops below
    `cut_hist` and the mean absolute luma difference exceeds `cut_diff` (hard cut).
    """
    n = len(frames)
    black = np.zeros(n, bool)
    grays, hists = [], []
    for i, f in enumerate(frames):
        g = _small_gray(f)
        black[i] = (g < black_luma).mean() >= black_ratio
        grays.append(g.astype(np.float32))
        hists.append(_hist(f))

    shots, start = [], None
    for i in range(n):
        if black[i]:
            if start is not None:
                shots.append((start, i))
                start = None
            continue
        if start is None:
            start = i
            continue
        corr = cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_CORREL)
        diff = np.abs(grays[i] - grays[i - 1]).mean()
        if corr < cut_hist and diff > cut_diff and i - start >= min_len:
            shots.append((start, i))
            start = i
    if start is not None:
        shots.append((start, n))
    return shots, black
