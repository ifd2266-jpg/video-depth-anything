import sys, numpy as np, cv2, onnxruntime as ort, os
S=sys.argv[1]; vid=sys.argv[2]
cap=cv2.VideoCapture(vid); frames=[]
while True:
    ok,f=cap.read()
    if not ok: break
    frames.append(f)
print(len(frames), frames[0].shape, flush=True)
so=ort.SessionOptions(); so.intra_op_num_threads=4
so.enable_cpu_mem_arena=False
which=sys.argv[3]
mean=np.array([0.485,0.456,0.406],np.float32); std=np.array([0.229,0.224,0.225],np.float32)
def prep(f,sz):
    x=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2RGB),(sz,sz),interpolation=cv2.INTER_CUBIC).astype(np.float32)/255
    return ((x-mean)/std).transpose(2,0,1)[None]

if which=='depth':
    dep=ort.InferenceSession(f'{S}/models/depth_anything_v2_vitl.onnx',so)
    D=[dep.run(None,{'l_x_':prep(f,518)})[0][0] for f in frames]
    np.save(f'{S}/work/depth.npy',np.array(D))
else:
    mat=ort.InferenceSession(f'{S}/models/birefnet_portrait.onnx',so); M=[]
    for i,f in enumerate(frames):
        m=mat.run(None,{'input_image':prep(f,1024)})[0][0,0]
        M.append((1/(1+np.exp(-m))).astype(np.float16)); print(i,flush=True)
    np.save(f'{S}/work/matte.npy',np.array(M))
print('done',flush=True)
