# Cinematic Depth of Field (`dof_bokeh.py`)

Shallow-depth-of-field optical bokeh for video (85mm f/1.4 look), built on top of
Video Depth Anything in this repo. GPU-accelerated when CUDA is available (depth model,
guided filter, layered disk convolution and compositing all run in PyTorch); falls back to CPU.

## Setup

```bash
pip install torch torchvision opencv-python-headless einops easydict tqdm   # + ffmpeg on PATH
bash get_weights.sh                                                          # VDA checkpoints
curl -L -o checkpoints/face_detection_yunet_2023mar.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

`xformers`, `decord`, `imageio` are **not** needed (the repo's attention falls back to PyTorch SDPA,
and video I/O goes through FFmpeg directly).

## Two-step workflow

```bash
# 1) preview: diagnostics for t=1s (shot 1) and t=8s (shot 2)
python dof_bokeh.py --input in.mov --output outputs/out.mov --preview-only
#    -> outputs/out_preview_shot<k>_<frame>.jpg       [Original | Refined depth + focus box | Final]
#    -> outputs/out_preview_shot<k>_<frame>_zoom.png  1:1 face crops (original | final)

# 2) full render (original resolution / FPS, BT.709, audio stream-copied)
python dof_bokeh.py --input in.mov --output outputs/out.mov
```

Depth is cached per shot in `outputs/depth_cache/`, so once it has been computed, re-renders and
previews with different blur settings skip the depth model and stay identical to each other.

| flag | default | meaning |
|---|---|---|
| `--max-blur` | 26 | max CoC radius in px (at 1080 px short side, scaled with resolution) |
| `--dof-zone` | 0.10 | in-focus deadzone width (normalised disparity) |
| `--fg-blur-scale` | 0.35 | near-field blur relative to far field |
| `--highlight-boost` | 0.6 | luminous bokeh highlights + subtle bloom (0 = off) |
| `--falloff` | 0.85 | CoC curve exponent (1 = strictly linear in disparity) |
| `--grain` | 1.0 | grain re-injection in defocused areas |
| `--temporal` | 0.5 | flow-aware temporal depth smoothing |
| `--no-head-lock` | off | disable the subject-adaptive far half of the focus zone |
| `--encoder` | vitl | `vits` / `vitb` / `vitl` depth model |
| `--preview-times` | `1.0,8.0` | seconds for preview frames |

## How it works

1. **Shots**: black-frame runs and hard cuts (HSV histogram + luma diff) split the video. Black frames
   pass through untouched. For every shot, *all* state is reset: VDA temporal context (each shot is
   inferred separately in VDA's native temporal video mode), depth normalisation, flow smoothing,
   face tracking and the focus filter.
2. **Depth**: VDA relative disparity → robust per-shot percentile normalisation → edge refinement:
   small near-dilation (silhouette pixels stay with the nearer object), colour fast guided filter,
   applied only inside the depth-edge band (no texture copying from the shemagh pattern) →
   DIS-optical-flow-warped temporal EMA with depth + photometric confidence.
3. **Focus (face lock, not nearest object)**: YuNet face detection; the main facing subject is seeded
   by size/confidence/centrality over the whole shot and tracked by IoU (back-turned people have no
   detectable face and can never take focus). `z_focus` = trimmed median disparity of the inner face
   box → One-Euro filter. *Head lock*: the far half of the focus zone widens to cover the measured
   disparity of the iqal/headdress above the face, so the whole head stays sharp.
4. **CoC**: thin-lens blur is linear in disparity difference, which is what VDA predicts (the unknown
   affine shift cancels). Far field reaches `max_blur` at the farthest point of the shot; near field
   reaches `fg_blur_scale × max_blur` at the nearest point.
5. **Rendering, in linear light** (`x**2.2`): highlight boost → signed CoC quantised into ~25 fine
   depth layers → each layer blurred with an anti-aliased circular aperture (premultiplied
   colour + coverage, FFT; large radii at reduced resolution) → back-to-front *over* compositing with
   coverage renormalisation. Background never leaks onto the subject, the subject never bleeds into the
   background, there are no dark occlusion halos, and defocused foreground edges spread over the subject
   as in a real lens. Then bloom → highlight shoulder → sRGB (`x**(1/2.2)`) → matched luminance grain
   in the defocused areas.
6. **Encode**: FFmpeg, explicit BT.709 matrix in and out, x264 CRF 12, original FPS, `-c:a copy`.
