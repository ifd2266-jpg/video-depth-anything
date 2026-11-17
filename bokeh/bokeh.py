import sys, numpy as np, cv2, subprocess
S=sys.argv[1]; vid=sys.argv[2]; out=sys.argv[3]
RMAX=float(sys.argv[4]) if len(sys.argv)>4 else 20.0   # medium strength (px @1080w)
cap=cv2.VideoCapture(vid); frames=[]
while True:
    ok,f=cap.read()
    if not ok: break
    frames.append(f)
N=len(frames); H,W=frames[0].shape[:2]
D=np.load(f'{S}/work/depth.npy').astype(np.float32)   # disparity, higher = nearer
M=np.load(f'{S}/work/matte.npy').astype(np.float32)
# temporal smoothing of depth (5-frame weighted)
k=np.array([1,2,3,2,1],np.float32); k/=k.sum()
Dp=np.pad(D,((2,2),(0,0),(0,0)),mode='edge'); D=sum(k[i]*Dp[i:i+N] for i in range(5))
lo,hi=np.percentile(D,1),np.percentile(D,99.5); D=np.clip((D-lo)/(hi-lo),0,1)

def up(x,interp=cv2.INTER_CUBIC): return cv2.resize(x,(W,H),interpolation=interp)
# focus = disparity of the main person(s)
foc=[]
for i in range(N):
    m=cv2.resize(M[i],(518,518)); d=D[i]
    sel=d[m>0.6]
    foc.append(np.percentile(sel,50) if sel.size>500 else (foc[-1] if foc else 0.6))
foc=np.array(foc); fp=np.pad(foc,(4,4),mode='edge'); foc=np.convolve(fp,np.ones(9)/9,'valid')
print('focus',foc.round(3)[::10])

def disc(r):
    r=max(r,0.5); n=int(np.ceil(r))+1; y,x=np.mgrid[-n:n+1,-n:n+1]
    k=np.clip(r+0.5-np.sqrt(x*x+y*y),0,1).astype(np.float32); return k/k.sum()
LEVELS=np.linspace(0,RMAX,9)
KER=[None]+[disc(r) for r in LEVELS[1:]]
def blur(x,i): return x if KER[i] is None else cv2.filter2D(x,-1,KER[i],borderType=cv2.BORDER_REFLECT)

TOL=0.06; RAMP=0.30
rng=np.random.default_rng(0)
p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','30','-i','-',
   '-i',vid,'-map','0:v','-map','1:a?','-c:v','libx264','-preset','slow','-crf','14','-pix_fmt','yuv420p',
   '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-c:a','copy','-movflags','+faststart',out],stdin=subprocess.PIPE)
for i in range(N):
    img=frames[i].astype(np.float32)/255; lin=img**2.2
    d=up(D[i]); a=np.clip(up(M[i]),0,1)
    f=foc[i]; dd=d-f
    # circle of confusion (signed: + foreground, - background)
    rel=np.abs(dd)/max(f,1e-3)
    coc=np.clip((rel-0.12)/0.40,0,1); coc=coc*coc*(3-2*coc)
    coc_bg=np.where(dd<0,coc,0)*(1-a)            # people stay sharp
    coc_fg=np.where(dd>0,np.clip((rel-0.5)/1.5,0,1),0)
    coc_bg=cv2.GaussianBlur(coc_bg,(0,0),3); r_bg=coc_bg*RMAX
    # highlight boost -> bokeh balls in linear light
    lum=lin@np.array([0.114,0.587,0.299],np.float32)
    boost=1+4*np.clip(lum-0.75,0,None)[...,None]
    src=lin*boost
    # background weight: exclude subject so no colour bleeding / halos
    wb=np.clip(1-a*1.15,0,1)*(d<f*0.95)+1e-4; wb=wb.astype(np.float32)
    acc=np.zeros_like(lin); idx=r_bg/(RMAX/(len(LEVELS)-1))
    prev=None
    for li in range(len(LEVELS)):
        if li==0: lay=src
        else:
            num=blur(src*wb[...,None],li); den=blur(wb,li)[...,None]
            lay=num/np.maximum(den,1e-4)
        w=np.clip(1-np.abs(idx-li),0,1)[...,None]
        acc+=lay*w
    bg=acc/(1+4*np.clip((acc@np.array([0.114,0.587,0.299],np.float32))-0.75,0,None)[...,None]/1.0)
    bg=np.where((r_bg>0.3)[...,None],bg,lin)
    # restore film/sensor grain in blurred areas
    noise_sigma=0.010
    g=rng.normal(0,noise_sigma,(H,W,1)).astype(np.float32)
    outl=bg+g*np.clip(r_bg/4,0,1)[...,None]*np.sqrt(np.clip(bg,1e-4,1))
    # extra foreground defocus (already soft in source), premultiplied spill over subject
    if coc_fg.max()>0.05:
        wf=(coc_fg>0.05).astype(np.float32)*coc_fg
        li=4
        Cf=blur(outl*wf[...,None],li); Af=blur(wf,li)[...,None]
        outl=outl*(1-Af)+Cf
    o=np.clip(outl,0,1)**(1/2.2)
    p.stdin.write((o*255+rng.random(o.shape,dtype=np.float32)).clip(0,255).astype(np.uint8).tobytes())
    if i%10==0: print('frame',i,flush=True)
p.stdin.close(); p.wait(); print('saved',out)
