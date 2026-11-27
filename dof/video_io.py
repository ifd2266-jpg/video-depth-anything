"""FFmpeg-based video I/O with explicit BT.709 colour handling."""
import json
import subprocess

import numpy as np


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    num, den = map(int, v["r_frame_rate"].split("/"))
    return {
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": num / den,
        "fps_str": v["r_frame_rate"],
        "nb_frames": int(v.get("nb_frames", 0) or 0),
        "has_audio": has_audio,
        "color_range": v.get("color_range", "tv"),
        "color_space": v.get("color_space", "bt709"),
        "rotation": _rotation(v),
    }


def _rotation(v):
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(sd["rotation"])
    return int(v.get("tags", {}).get("rotate", 0) or 0)


def _matrix(info):
    cs = info.get("color_space", "bt709")
    return "bt601" if cs in ("bt470bg", "smpte170m") else "bt709"


def read_frames(path, info, frames=None):
    """Decode to uint8 RGB (T,H,W,3). `frames` = optional sorted list of indices to keep."""
    rng = "pc" if info["color_range"] == "pc" else "tv"
    vf = f"scale=in_color_matrix={_matrix(info)}:in_range={rng}:flags=accurate_rnd+full_chroma_int,format=rgb24"
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    w, h = info["width"], info["height"]
    if abs(info["rotation"]) in (90, 270):
        w, h = h, w
    size = w * h * 3
    keep = None if frames is None else set(frames)
    out, idx = [], 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        if keep is None or idx in keep:
            out.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
        idx += 1
    proc.stdout.close()
    proc.wait()
    return np.stack(out), idx


class VideoWriter:
    """Pipe RGB frames into FFmpeg (x264, BT.709 tagged) and copy audio from `audio_src` if present."""

    def __init__(self, path, width, height, fps_str, audio_src=None, crf=12, preset="slow",
                 color_range="tv"):
        rng = "pc" if color_range == "pc" else "tv"
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", fps_str, "-i", "-"]
        if audio_src:
            cmd += ["-i", audio_src, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy"]
        cmd += ["-vf", f"scale=out_color_matrix=bt709:out_range={rng}:flags=accurate_rnd+full_chroma_int",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                "-color_range", rng, "-r", fps_str, "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb_uint8):
        self.proc.stdin.write(np.ascontiguousarray(rgb_uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg encoder failed")
