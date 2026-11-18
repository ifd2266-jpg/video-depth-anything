import numpy as np, cv2
from common import disk, to_lin, to_srgb
RADII=[0,1,2,3,4.5,6,8,10.5,13.5,17,21,26,32,39,47,56,66]
def conv(x,r):
    if r<=0.25: return x
    if r>=9:   # blur at half-res for big kernels (speed), result is smooth anyway
        h,w=x.shape[:2]; s=cv2.resize(x,(w//2,h//2),interpolation=cv2.INTER_AREA)
        s=cv2.filter2D(s,-1,disk(r/2),borderType=cv2.BORDER_REFLECT)
        return cv2.resize(s,(w,h),interpolation=cv2.INTER_LINEAR)
    return cv2.filter2D(x,-1,disk(r),borderType=cv2.BORDER_REFLECT)
def layer_weights(r):
    """r: signed CoC radius map. returns dict (sign,idx)->weight with linear split between radii"""
    a=np.abs(r); R=np.array(RADII,np.float32); a=np.clip(a,0,R[-1])
    i=np.clip(np.searchsorted(R,a,side='right')-1,0,len(R)-2)
    t=(a-R[i])/(R[i+1]-R[i]); out={}
    for k in range(len(R)):
        w=np.where(i==k,1-t,0)+np.where(i+1==k,t,0)
        for sgn in (-1,1):
            if k==0 and sgn==-1: continue
            m=w*((r>=0) if (sgn==1) else (r<0)) if k>0 else w
            if m.max()>1e-4: out[(sgn,k)]=m.astype(np.float32)
    return out
def render(img_srgb,r,hl_boost=0.0):
    """img_srgb HxWx3 float [0,1]; r signed CoC radius (+ = in front of focus)."""
    lin=to_lin(img_srgb)
    if hl_boost>0:   # recover a bit of highlight energy clipped in the 8-bit source so bokeh blooms like real glass
        L=lin.max(-1,keepdims=True); lin=lin*(1+hl_boost*np.clip((L-0.8)/0.2,0,1)**2)
    W=layer_weights(r)
    # --- background group: normalized gather (no foreground contamination) ---
    acc=np.zeros(lin.shape[:2]+(4,),np.float32)
    for (s,k),w in W.items():
        if s!=-1: continue
        acc+=conv(np.dstack([lin*w[...,None],w]),RADII[k])
    # hole fill under the foreground: wide normalized blur of background pixels
    wb=(r<0).astype(np.float32)
    fill=conv(np.dstack([lin*wb[...,None],wb]),40)
    fill=fill[...,:3]/np.maximum(fill[...,3:],1e-4)
    a=acc[...,3:]
    bg=np.where(a>1e-3,acc[...,:3]/np.maximum(a,1e-6),fill)
    bg=bg*np.clip(a/0.5,0,1)+fill*(1-np.clip(a/0.5,0,1))
    # --- focus + foreground group: back-to-front premultiplied over ---
    out=bg
    for k in range(len(RADII)):
        if (1,k) not in W: continue
        w=W[(1,k)]; p=conv(np.dstack([lin*w[...,None],w]),RADII[k])
        out=p[...,:3]+out*(1-np.clip(p[...,3:],0,1))
    return to_srgb(out)
