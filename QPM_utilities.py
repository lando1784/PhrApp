import numpy as np
from numpy import fft
import platform #di sistema
import sys
import cmath
import math
from numpy import uint16, uint8, uint32 #ok
from scipy.interpolate import interp1d

imgTypes = {8:uint8,16:uint16,32:uint32,12:uint16}

def atan2_2D(X,Y):
    
    res = np.zeros((np.shape(X)[0],np.shape(X)[1]))
    
    for i in range(0,np.shape(X)[0]):
        for j in range(0,np.shape(X)[1]):
            res[i][j] = math.atan2(Y[i][j], X[i][j])
            
    return res


def adjustImgRange(img,newMax,bits = None):
        
        currImgMax = np.max(img)
        currImgMin = np.min(img) if np.min(img) < currImgMax else 0   
        
        newImg = (img-float(currImgMin))/((float(currImgMax)-float(currImgMin)))*(newMax if np.min(img) < currImgMax else 0.5*newMax) 
        
        if bits == None:
            return newImg
        else:
            newImg2 = newImg.astype(imgTypes[bits])
            return newImg2


def myFFT2(signal,center = True):
    
    result = fft.fftshift(fft.fft2(signal)) if center else fft.fft2(signal)
    
    return result


def myIFFT2(Fsignal,center = True):
    
    result = fft.ifft2(fft.ifftshift(Fsignal)) if center else fft.ifft2(Fsignal)
    
    return result


def kCoords(R,C,dx):
    
    dFx = 1.0/(dx*C)
    dFy = 1.0/(dx*R)
    
    kxP = (np.arange(C/2+C%2)+1)*dFx
    kxM = np.sort(-1*kxP)[C%2:]
    kx = np.matrix(np.concatenate((kxM,kxP)))
    
    kyP = (np.arange(R/2+R%2)+1)*dFy
    kyM = np.sort(-1*kyP)[R%2:]
    ky = np.matrix(np.concatenate((kyM,kyP)))
    
    kx = kx.T*np.matrix(np.ones(shape=[1,R]))
    kx = np.float64(kx.T)
    ky = np.float64(ky.T*np.matrix(np.ones(shape=[1,C])))
    
    return kx,ky


def kCoordsPrev(R,C,dx):
    
    dFx = 1.0/(dx*C)
    dFy = 1.0/(dx*R)
    
    kx = np.matrix(np.arange(-C/2,C/2))*dFx
    ky = np.matrix(np.arange(-R/2,R/2))*dFy
    
    kx = kx.T*np.matrix(np.ones(shape=[1,R]))
    kx = np.float64(kx.T)
    ky = np.float64(ky.T*np.matrix(np.ones(shape=[1,C])))
    
    return kx,ky


def WxyFilterDenCorr(k,alpha,fselector):
    
    if len(alpha)==1:
        corrF = np.square(k)*alpha[0]
    else:
        if fselector == 1:
            corrF = np.square(k)*alpha[0] + k*alpha[1]
        if fselector == 2:
            corrF = alpha[0]*np.exp(-1*k*alpha[1])
        if fselector == 3:
            corrF = alpha[0]*k + alpha[1]
        if fselector == 4:
            corrF = np.square(k)*alpha[0] + alpha[1] 
     
    
    return corrF


def ZaxisDerive(imgs, bestFocusInd):
    N,R,C = np.shape(imgs)
    
    g= 3 if len(imgs) >3 else 2
    gc=g+1
    
    Q=np.zeros([gc,gc])
    
    for r in range(0,gc):
        for c in range(0,gc):
            for k in range(0,N):
                Q[r,c]+=(k-bestFocusInd)**(r+c)
    
    Q=np.matrix(Q)
    
    T = np.zeros([gc,N])
    for r in range(0,gc):
        for c in range(0,N):
            T[r,c] = (c-bestFocusInd)**(r)

    T=np.matrix(T)

    
    W=Q.I*T

    
    F = np.array(W[1,:])
    
    
    V=np.zeros([N])
    Derivata=np.zeros([R,C])
    
    for r in range(0,R):
        for c in range(0,C):
            for k in range(0,N):
                V[k]=imgs[k][r,c]
                
            val=np.sum(F*V)
            Derivata[r,c]=val
    
    return Derivata


def ZaxisDerive_v2(imgs, zStep):
    
    N,R,C = np.shape(imgs)
    
    der = np.zeros([R,C])
    
    for r in range(0,R):
        for c in range(0,C):
            der[r,c] = (imgs[2][r][c]-imgs[0][r][c])/(2*zStep)
            
    return der


def ZaxisDerive_v3(imgs,bestFocusInd,degree,z=None):
    
    N,R,C = np.shape(imgs)
    imgs = np.array(imgs)
    
    if z is not None:
        zAxis = (np.arange(0,N) - bestFocusInd)*z
    else:
        zAxis = (np.arange(0,N) - bestFocusInd)
    V = np.zeros(N)
    g = degree
    Derivata = np.zeros([R,C])
    
    for r in range(0,R):
        for c in range(0,C):
            V = imgs[:,r,c]
            p = np.polyfit(zAxis,V,g)
            Derivata[r][c] = p[g-1]
            
    return Derivata


def propagateI(csiK, kpq, delta, k, ctr):
    
    FcsiK = myFFT2(csiK,ctr)
    coeffExp = np.exp((-1j*delta/(2*k))*kpq)# if delta!=0 else np.ones(np.shape(kpq))
    FcsiKp1 = np.multiply(coeffExp,FcsiK)
    csiKp1 = myIFFT2(FcsiKp1,ctr)
    
    return csiKp1


def propagateBack(csiKp1c, kpq, delta, k,ctr):
    
    coeffExp = np.exp((-1j*delta/(2*k))*kpq)
    FcsiKp1c = myFFT2(csiKp1c,ctr)
    FcsiKc = np.divide(FcsiKp1c,coeffExp)
    csiKc = myIFFT2(FcsiKc,ctr)
    phiGuess = np.arctan2(csiKc.imag,csiKc.real)
    
    return phiGuess

def make2dList(rows, cols):
    a=[]
    for row in xrange(rows): a += [[0]*cols]
    return a

def interpImgs(images,dz,zero = None,k ='linear'):
    
    N,R,C = np.shape(images)
    imgs = np.array(images)
    middle = zero if zero != None else (N-N%2)/2
    Z = (np.arange(N)-middle)*dz
    print Z
    Itp = make2dList(R,C)
    
    
    for r in range(R):
        for c in range(C):
            Itp[r][c] = interp1d(Z, imgs[:,r,c], kind=k)
            
    return Itp
    
    
if __name__=='__main__':
    
    print 'Not for Standalone use'
    
    
    