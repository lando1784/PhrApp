import numpy as np
from scipy import misc
import scipy as sp
from numpy import fft
import math
from PIL import Image as im
import matplotlib.pyplot as plt
import sys
import QPM_algorithm as qpm
import bestfocus as bf

imgTypes = {8:np.uint8,16:np.uint16,32:np.uint32,12:np.uint16}


def gaussian(x,y,cx,cy,b):
        return np.exp(-1*(b**2)*((x-cx)**2+(y-cy)**2))


def createIMGs(pxX,pxY,dx,dZ,b,imgN,l):
    
    R = np.arange(pxY)
    C = np.arange(pxX)
    
    k = 2*np.pi/l
    
    dFx = 1/(dx*pxX)
    dFy = 1/(dx*pxY)
    
    kx = np.arange(-pxX/2,pxX/2)*dFx
    ky = np.arange(-pxY/2,pxY/2)*dFy
    
#	kx = kx.T*np.matrix(np.ones(shape=[1,pxY]))
#	kx = np.float64(kx.T)
#	ky = np.float64(ky.T*np.matrix(np.ones(shape=[1,pxX])))
    #kpq = np.zeros((pxY,pxX))
    kx,ky = qpm.kCoordsPrev(pxY,pxX,dx)
    
    kpq = np.square(kx) + np.square(ky)

    r1 = pxY*3/5
    c1 = pxX*2/5
    
    r2 = pxY*2/5
    c2 = pxX*3/5
    
    I = np.zeros((pxY,pxX))
    
    for r in R:
        for c in C:
            #kpq[r,c] = kx[c]**2+ky[r]**2
            I[r,c]= 1.0-0.9*( gaussian(r,c,r1,c1,b) + gaussian(r,c,r2,c2,b)) 
            #I[r,c]= 1.0-0.9*(np.exp(-1*(b**2)*((c-c1)**2+(r-r1)**2))+np.exp(-1*(b**2)*((c-c2)**2+(r-r2)**2)))
    print np.min(I)
    I -= np.min(I)
    phi = 0.95*((2*np.pi/0.9)*(1.0-I)-np.pi)
    #I = np.ones((pxY,pxX))*128
    print np.shape(I)
    csi = np.sqrt(I)*(np.cos(phi)+1j*np.sin(phi))
    
    deltas = dZ*(np.arange(imgN)-(imgN-1)/2)
    imgs = []
    
    print deltas
    
    for d in deltas:
        tempCsi=qpm.propagateI(csi,kpq,d,k,True)
        tempImg = np.square(tempCsi.real)+np.square(tempCsi.imag)
        imgs.append(I if d==0 else tempImg)
        #imgs.append(tempImg)
        
    return imgs, phi


def createNstore(n,px,dX,dZ,b,imgNum,l,bPp,dir = ''):
    
    images, phi = createIMGs(px*n,px*n,dX/n,dZ,b/n,imgNum,l)
    count = 0
    bits = bPp
    
    for i in images:
        i = bf.adjustImgRange(i,2**(bits)-200,bits)+150
        print i.dtype
        img = im.fromarray(i,'I;'+str(bits))
        img.save(dir+str(count)+'.tif')
        count += 1
        
    phi = bf.adjustImgRange(phi,255,8)
    misc.imsave(dir+'phi'+'.tif',phi.astype(np.uint8))
	

if __name__== '__main__':
    
    n = float(sys.argv[1]) if len(sys.argv)>1 else 2
    createNstore(n,193,5.182e-7,100000000000e-7,0.045,5,632.8*10**-9,16)

