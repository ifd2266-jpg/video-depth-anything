"""SAM2 video segmentation of the in-focus subject (hand + product) and the arm.

Writes an .npz with
  focus   (T,H,W) float  hand + device  -> rendered tack sharp
  subject (T,H,W) float  hand + device + forearm/sleeve -> used to snap depth edges
Prompts are point clicks (x, y in full-res pixels) on a few key frames.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True)
    ap.add_argument('--prompts', required=True, help='json: {frame_idx: {obj_id: [[x,y,label],...]}}')
    ap.add_argument('--focus_ids', default='1,2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='facebook/sam2.1-hiera-large')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.frames, '*.png')))
    video = [Image.open(f).convert('RGB') for f in files]
    prompts = json.load(open(args.prompts))
    torch.set_num_threads(os.cpu_count())

    model = Sam2VideoModel.from_pretrained(args.model).eval()
    proc = Sam2VideoProcessor.from_pretrained(args.model)
    sess = proc.init_video_session(video=video, inference_device='cpu', dtype=torch.float32)

    obj_ids_all = set()
    for fidx, objs in prompts.items():
        ids = sorted(int(k) for k in objs)
        obj_ids_all.update(ids)
        pts = [[[p[:2] for p in objs[str(i)]] for i in ids]]
        lbl = [[[p[2] for p in objs[str(i)]] for i in ids]]
        proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=int(fidx),
                                             obj_ids=ids, input_points=pts, input_labels=lbl)
        with torch.no_grad():
            model(inference_session=sess, frame_idx=int(fidx))

    T = len(files)
    H, W = sess.video_height, sess.video_width
    obj_ids_all = sorted(obj_ids_all)
    masks = np.zeros((len(obj_ids_all), T, H, W), np.float16)
    start = min(int(k) for k in prompts)
    with torch.no_grad():
        for o in model.propagate_in_video_iterator(sess, start_frame_idx=start):
            m = proc.post_process_masks([o.pred_masks], original_sizes=[[H, W]], binarize=False)[0]
            m = torch.sigmoid(m[:, 0]).numpy()
            for j, oid in enumerate(o.object_ids if hasattr(o, 'object_ids') else obj_ids_all):
                masks[obj_ids_all.index(oid), o.frame_idx] = m[j]
            print('seg frame', o.frame_idx, flush=True)
        if start > 0:
            for o in model.propagate_in_video_iterator(sess, start_frame_idx=start, reverse=True):
                m = proc.post_process_masks([o.pred_masks], original_sizes=[[H, W]], binarize=False)[0]
                m = torch.sigmoid(m[:, 0]).numpy()
                for j, oid in enumerate(obj_ids_all):
                    masks[obj_ids_all.index(oid), o.frame_idx] = m[j]

    focus_ids = [int(x) for x in args.focus_ids.split(',')]
    fi = [obj_ids_all.index(i) for i in focus_ids if i in obj_ids_all]
    focus = masks[fi].astype(np.float32).max(0)
    subject = masks.astype(np.float32).max(0)
    np.savez_compressed(args.out, focus=focus.astype(np.float16), subject=subject.astype(np.float16),
                        per_object=masks, ids=np.array(obj_ids_all))
    print('saved', args.out, focus.shape)


if __name__ == '__main__':
    main()
