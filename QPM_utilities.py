import bestfocus as bf # 
from numpy import fft
import platform #di sistema
import sys
import cmath
import math



def myFFT2(signal,center = True):
    
    result = fft.fftshift(fft.fft2(signal)) if center else fft.fft2(signal)
    
    return result


def myIFFT2(Fsignal,center = True):
    
    result = fft.ifft2(fft.ifftshift(Fsignal)) if center else fft.ifftshift(Fsignal)
    
    return result


def kCoords(R,C,dx):
    
    dFx = 1.0/(dx*C)
    dFy = 1.0/(dx*R)
    
    kxP = (bf.np.arange(C/2+C%2)+1)*dFx
    kxM = bf.np.sort(-1*kxP)[C%2:]
    kx = bf.np.matrix(bf.np.concatenate((kxM,kxP)))
    
    kyP = (bf.np.arange(R/2+R%2)+1)*dFy
    kyM = bf.np.sort(-1*kyP)[R%2:]
    ky = bf.np.matrix(bf.np.concatenate((kyM,kyP)))
    
    kx = kx.T*bf.np.matrix(bf.np.ones(shape=[1,R]))
    kx = bf.np.float64(kx.T)
    ky = bf.np.float64(ky.T*bf.np.matrix(bf.np.ones(shape=[1,C])))
    
    return kx,ky


def kCoordsPrev(R,C,dx):
    
    dFx = 1.0/(dx*C)
    dFy = 1.0/(dx*R)
    
    kx = bf.np.matrix(bf.np.arange(-C/2,C/2))*dFx
    ky = bf.np.matrix(bf.np.arange(-R/2,R/2))*dFy
    
    kx = kx.T*bf.np.matrix(bf.np.ones(shape=[1,R]))
    kx = bf.np.float64(kx.T)
    ky = bf.np.float64(ky.T*bf.np.matrix(bf.np.ones(shape=[1,C])))
    
    return kx,ky


def WxyFilterDenCorr(k,alpha,fselector):
    
    if len(alpha)==1:
        corrF = bf.np.square(k)*alpha[0]
    else:
        if fselector == 1:
            corrF = bf.np.square(k)*alpha[0] + k*alpha[1]
        if fselector == 2:
            corrF = alpha[0]*bf.np.exp(-1*k*alpha[1])
        if fselector == 3:
            corrF = alpha[0]*k + alpha[1]
        if fselector == 4:
            corrF = bf.np.square(k)*alpha[0] + alpha[1] 
     
    
    return corrF


def ZaxisDerive(imgs, bestFocusInd):
    N,R,C = bf.np.shape(imgs)
    
    g= 3 if len(imgs) >3 else 2
    gc=g+1
    
    Q=bf.np.zeros([gc,gc])
    
    for r in range(0,gc):
        for c in range(0,gc):
            for k in range(0,N):
                Q[r,c]+=(k-bestFocusInd)**(r+c)
    
    Q=bf.np.matrix(Q)
    
    T = bf.np.zeros([gc,N])
    for r in range(0,gc):
        for c in range(0,N):
            T[r,c] = (c-bestFocusInd)**(r)

    T=bf.np.matrix(T)

    
    W=Q.I*T

    
    F = bf.np.array(W[1,:])
    
    
    V=bf.np.zeros([N])
    Derivata=bf.np.zeros([R,C])
    
    for r in range(0,R):
        for c in range(0,C):
            for k in range(0,N):
                V[k]=imgs[k][r,c]
                
            val=bf.np.sum(F*V)
            Derivata[r,c]=val
    
    return Derivata


def ZaxisDerive_v2(imgs, zStep):
    
    N,R,C = bf.np.shape(imgs)
    
    der = bf.np.zeros([R,C])
    
    for r in range(0,R):
        for c in range(0,C):
            der[r,c] = (imgs[2][r][c]-imgs[0][r][c])/(2*zStep)
            
    return der


def ZaxisDerive_v3(imgs,bestFocusInd,degree,z=None):
    
    N,R,C = bf.np.shape(imgs)
    imgs = bf.np.array(imgs)
    
    if z is not None:
        zAxis = (bf.np.arange(0,N) - bestFocusInd)*z
    else:
        zAxis = (bf.np.arange(0,N) - bestFocusInd)
    V = bf.np.zeros(N)
    g = degree
    Derivata = bf.np.zeros([R,C])
    
    for r in range(0,R):
        for c in range(0,C):
            V = imgs[:,r,c]
            p = bf.np.polyfit(zAxis,V,g)
            Derivata[r][c] = p[g-1]
            
    return Derivata


def propagateI(csiK, kpq, delta, k):
    
    FcsiK = myFFT2(csiK,CTR)
    coeffExp = bf.np.exp((-1j*delta/(2*k))*kpq)
    FcsiKp1 = bf.np.multiply(coeffExp,FcsiK)
    csiKp1 = myIFFT2(FcsiKp1,CTR)
    
    return csiKp1


def propagateBack(csiKp1c, kpq, delta, k):
    
    coeffExp = bf.np.exp((-1j*delta/(2*k))*kpq)
    FcsiKp1c = myFFT2(csiKp1c,CTR)
    FcsiKc = bf.np.divide(FcsiKp1c,coeffExp)
    csiKc = myIFFT2(FcsiKc,CTR)
    phiGuess = bf.np.arctan2(csiKc.imag,csiKc.real)
    
    return phiGuess