import numpy as np, cv2
def box(x,r): return cv2.boxFilter(x,-1,(2*r+1,2*r+1),borderType=cv2.BORDER_REFLECT)
def guided(I,p,r,eps):
    """Color guided filter (He et al.). I: HxWx3 float32 [0,1], p: HxW float32"""
    mI=box(I,r); mp=box(p,r)
    cov=box(I*p[...,None],r)-mI*mp[...,None]
    II=np.einsum('hwi,hwj->hwij',I,I)
    v=np.zeros(I.shape[:2]+(3,3),np.float32)
    for i in range(3):
        for j in range(i,3):
            c=box(np.ascontiguousarray(II[...,i,j]),r)-mI[...,i]*mI[...,j]; v[...,i,j]=c; v[...,j,i]=c
    v+=np.eye(3,dtype=np.float32)*eps
    a=np.linalg.solve(v,cov[...,None])[...,0]
    b=mp-(a*mI).sum(-1)
    return (box(a,r)*I).sum(-1)+box(b,r)
def disk(r):
    """anti-aliased disk kernel of radius r (px), normalized"""
    R=int(np.ceil(r))+1; y,x=np.mgrid[-R:R+1,-R:R+1].astype(np.float32)
    k=np.clip(r+0.5-np.sqrt(x*x+y*y),0,1)
    return k/k.sum()
def to_lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4).astype(np.float32)
def to_srgb(x):
    x=np.clip(x,0,1); return np.where(x<=0.0031308,x*12.92,1.055*x**(1/2.4)-0.055).astype(np.float32)
