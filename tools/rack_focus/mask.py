import sys,numpy as np,cv2
sys.path.insert(0,'work')
from common import guided
S=np.load('depth_aligned.npy'); n,h,w=S.shape
def subject_mask(i,im,lowres_only=False):
    d=S[i]; T=400.0
    m=(d>T).astype(np.uint8)
    k,lab,st,_=cv2.connectedComponentsWithStats(m,8)
    if k<=1: return None,None
    # component overlapping the phone/hands zone
    zone=np.zeros_like(m); zone[int(h*.25):int(h*.8),int(w*.3):int(w*.7)]=1
    best=max(range(1,k),key=lambda j:((lab==j)&(zone>0)).sum())
    if ((lab==best)&(zone>0)).sum()<200: return None,None
    m=(lab==best).astype(np.float32)
    med=float(np.median(d[m>0]))
    if lowres_only: return None,med
    M=cv2.resize(m,(1080,1920),interpolation=cv2.INTER_LINEAR)
    M=guided(im,M.astype(np.float32),10,1e-3); M=guided(im,np.clip(M,0,1).astype(np.float32),3,1e-4)
    M=np.clip((M-0.5)*1.6+0.5,0,1)  # tighten transition, keep sub-pixel AA
    return M.astype(np.float32),med
if __name__=='__main__':
    out=[]
    for i in map(int,sys.argv[1:]):
        im=cv2.imread('frames/%04d.png'%(i+1))[...,::-1].astype(np.float32)/255
        M,med=subject_mask(i,im); print(i,med)
        v=(im*0.35+np.dstack([M*0,M*0.65,M*0])) if M is not None else im
        out.append(cv2.resize((v[...,::-1]*255).astype(np.uint8)[350:1450,180:900],None,fx=.5,fy=.5))
    cv2.imwrite('masks.jpg',np.concatenate(out,1))
