"""Low-memory Video Depth Anything (vitl) inference for CPU: the temporal head runs its
high-resolution layers one frame at a time (same result, ~7 GB instead of >15 GB).
Usage: python tools/run_depth_lowmem.py input.mp4 depths.npz   (run from repo root)
"""
import sys, gc, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything import dpt_temporal
from utils.dc_utils import read_video_frames
torch.set_num_threads(4)

def fwd(self, out_features, patch_h, patch_w, frame_length, micro_batch_size=1, cached_hidden_state_list=None):
    assert cached_hidden_state_list is None and not self.use_clstoken
    feats = [f[0] for f in out_features]
    def proj(i, x):
        x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()
        return self.resize_layers[i](self.projects[i](x))
    B, T = feats[0].shape[0] // frame_length, frame_length
    def mm(k, x):
        y, h = self.motion_modules[k](x.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None, None)
        return y.permute(0, 2, 1, 3, 4).flatten(0, 1), h
    layer_3, h0 = mm(0, proj(2, feats[2]))
    layer_4, h1 = mm(1, proj(3, feats[3]))
    layer_3_rn = self.scratch.layer3_rn(layer_3); layer_4_rn = self.scratch.layer4_rn(layer_4)
    del layer_3, layer_4
    size2 = proj(1, feats[1][:1]).shape[2:]
    path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
    path_4, h2 = mm(2, path_4)
    path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=size2)
    path_3, h3 = mm(3, path_3)
    del path_4, layer_3_rn, layer_4_rn
    ret = []
    for i in range(feats[0].shape[0]):
        l1 = self.scratch.layer1_rn(proj(0, feats[0][i:i+1]))
        l2 = self.scratch.layer2_rn(proj(1, feats[1][i:i+1]))
        p2 = self.scratch.refinenet2(path_3[i:i+1], l2, size=l1.shape[2:])
        p1 = self.scratch.refinenet1(p2, l1)
        out = self.scratch.output_conv1(p1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        ret.append(self.scratch.output_conv2(out.float()))
    return torch.cat(ret, 0), h0 + h1 + h2 + h3
dpt_temporal.DPTHeadTemporal.forward = fwd

m = VideoDepthAnything(encoder='vitl', features=256, out_channels=[256,512,1024,1024])
sd = torch.load('./checkpoints/video_depth_anything_vitl.pth', map_location='cpu'); m.load_state_dict(sd, strict=True); del sd; gc.collect()
m = m.eval()
frames, fps = read_video_frames(sys.argv[1], -1, -1, 1920)
print(frames.shape, fps, flush=True)
depths, fps = m.infer_video_depth(frames, fps, input_size=518, device='cpu', fp32=True)
np.savez_compressed(sys.argv[2], depths=np.asarray(depths, np.float32))
print('saved', np.asarray(depths).shape, flush=True)
