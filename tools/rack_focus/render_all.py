import sys,os,glob,json,numpy as np,cv2
sys.path.insert(0,'work')
from common import guided
from dof import render
from mask import subject_mask
P=json.load(open('work/params.json'))
S=np.load('depth_aligned.npy'); n=len(S)
fd=np.array(P['focus']); bf=np.array(P['band_front']); bb=np.array(P['band_back'])
K=P['K']; os.makedirs('out',exist_ok=True)
rng=np.random.default_rng(7)
idx=[int(a) for a in sys.argv[1:]] or range(n)
# subject (phone+hands) depth per frame, smoothed -> the phone is one solid object at one focus distance
meds=np.array([subject_mask(i,None,True)[1] or np.nan for i in range(n)],float)
v=~np.isnan(meds); meds=np.interp(np.arange(n),np.where(v)[0],meds[v])
k=4; meds_s=np.convolve(np.pad(meds,k,mode='edge'),np.ones(2*k+1)/(2*k+1),'valid')
ax0=np.array(P['anchors']); s_rack=np.array(P['rack'])
for i in idx:
    im=cv2.imread('frames/%04d.png'%(i+1))[...,::-1].astype(np.float32)/255
    d=cv2.resize(S[i],(1080,1920),interpolation=cv2.INTER_CUBIC)
    M,_=subject_mask(i,im)
    lo,hi=P['dlo'],P['dhi']; dn=((d-lo)/(hi-lo)).astype(np.float32)
    # edge snapping: coarse then fine guided filter on the RGB frame
    dn=guided(im,dn,12,2e-3); dn=guided(im,dn,4,2e-4)
    d=dn*(hi-lo)+lo
    if M is not None: d=M*meds_s[i]+(1-M)*d
    ax=np.array(P['anchors'])  # DA relative disparity -> inverse metric depth (diopters), piecewise linear
    sl0=(ax[1,1]-ax[0,1])/(ax[1,0]-ax[0,0]); sl1=(ax[-1,1]-ax[-2,1])/(ax[-1,0]-ax[-2,0])
    u=np.interp(d,ax[:,0],ax[:,1]); u=np.where(d<ax[0,0],ax[0,1]+(d-ax[0,0])*sl0,u); u=np.where(d>ax[-1,0],ax[-1,1]+(d-ax[-1,0])*sl1,u)
    u=np.maximum(u,0.05)
    fu=(1-s_rack[i])*fd[i]+s_rack[i]*np.interp(meds_s[i],ax[:,0],ax[:,1]) if s_rack[i]>0 else fd[i]
    e=u-fu
    e=np.where(e>0,np.maximum(e-bf[i],0),np.minimum(e+bb[i],0))  # depth-of-field band
    r=K*e; r=np.sign(r)*np.maximum(np.abs(r)-0.5,0)                 # permissible CoC
    r=np.clip(r,-P['rmax'],P['rmax'])
    o=render(im,r,hl_boost=0.6)
    # grain re-injection on defocused areas (blur kills sensor noise -> looks pasted otherwise)
    amt=np.clip(np.abs(r)/3,0,1)[...,None]
    g=rng.normal(0,1,im.shape[:2]).astype(np.float32); g=cv2.GaussianBlur(g,(0,0),0.7)
    gc=rng.normal(0,1,im.shape).astype(np.float32); gc=cv2.GaussianBlur(gc,(0,0),1.2)
    o=o+amt*(P['grain']*g[...,None]+P['grain']*0.35*gc)
    cv2.imwrite('out/%04d.png'%(i+1),np.clip(o[...,::-1]*255+0.5,0,255).astype(np.uint8))
    np.save('out/r%04d.npy'%(i+1),r.astype(np.float16)) if os.environ.get('SAVE_R') else None
    print(i,flush=True)
