# Align depth across time + compute focus targets. Writes aligned depth (low-res) and stats.
import sys,glob,numpy as np,cv2
sys.path.insert(0,'work')
fs=sorted(glob.glob('depth/*.npy')); D=[np.load(f) for f in fs]; n=len(D)
h,w=D[0].shape
def fit(a,b):  # robust affine a*s+t ~ b
    m=np.ones(a.shape,bool)
    for _ in range(4):
        A=np.stack([a[m],np.ones(m.sum())],1); s,t=np.linalg.lstsq(A,b[m],rcond=None)[0]
        r=np.abs(a*s+t-b); m=r<np.quantile(r,0.6)
    return s,t
# static region (curtain top + sides + couch) never touched by the hands -> stable reference
mask=np.zeros((h,w),bool); mask[:int(h*.27)]=True; mask[:,:int(w*.12)]=True; mask[:,int(w*.88):]=True
A=[D[0]]
for i in range(1,n):
    s,t=fit(D[i][mask],A[i-1][mask]); A.append(D[i]*s+t)
A=np.stack(A)
# edge-safe temporal smoothing (only where neighbours agree -> no ghosting on moving hands)
S=A.copy()
for i in range(n):
    acc=A[i].copy(); wsum=np.ones_like(acc)
    for j in (i-2,i-1,i+1,i+2):
        if 0<=j<n:
            wj=np.exp(-((A[j]-A[i])/6.0)**2)*(0.5 if abs(j-i)==2 else 1)
            acc+=wj*A[j]; wsum+=wj
    S[i]=acc/wsum
np.save('depth_aligned.npy',S.astype(np.float32))
sy,sx=h/1920,w/1080
def box(a,y0,y1,x0,x1): return a[int(y0*sy):int(y1*sy),int(x0*sx):int(x1*sx)]
for i in range(n):
    f=np.median(box(S[i],620,820,480,680)); c=np.median(box(S[i],100,400,100,1000))
    b=box(S[i],350,1750,330,750); p=np.quantile(b,0.97)
    print(i,round(float(c),1),round(float(f),1),round(float(p),1))
