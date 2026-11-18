import onnxruntime as ort, numpy as np, cv2, sys, os, time, glob
S=sys.argv[1]; H,W=int(sys.argv[2]),int(sys.argv[3]); out=sys.argv[4]; lim=int(sys.argv[5]) if len(sys.argv)>5 else 10**9
so=ort.SessionOptions(); so.intra_op_num_threads=4
s=ort.InferenceSession('/home/user/video-depth-anything/checkpoints/dav2l.onnx',so,providers=['CPUExecutionProvider'])
os.makedirs(out,exist_ok=True)
m=np.array([0.485,0.456,0.406]);sd=np.array([0.229,0.224,0.225])
fs=sorted(glob.glob(S+'/*.png'))[:lim]
for f in fs:
    o=os.path.join(out,os.path.basename(f)[:-4]+'.npy')
    if os.path.exists(o): continue
    t=time.time()
    im=cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2RGB).astype(np.float32)/255
    x=cv2.resize(im,(W,H),interpolation=cv2.INTER_CUBIC)
    x=((x-m)/sd).transpose(2,0,1)[None].astype(np.float32)
    d=s.run(None,{'pixel_values':x})[0][0]
    # horizontal flip TTA for cleaner edges
    d2=s.run(None,{'pixel_values':np.ascontiguousarray(x[...,::-1])})[0][0][:,::-1]
    np.save(o,((d+d2)/2).astype(np.float32)); print(f,time.time()-t,flush=True)
