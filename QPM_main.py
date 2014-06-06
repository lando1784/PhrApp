import bestfocus as bf # 
from numpy import fft
import platform #di sistema
import sys
import cmath
import math

lamD = 533.0 * (10**-9)
kD = (2 * bf.np.pi) / lamD


zD = 500 * (10**-9)
alphaCorrD = [1e-4]
dxD = 7.98 * (10**-8)

zeroSubst = 10**-9

CTR = True

def myFFT2(signal,center = True):
    
    result = fft.fftshift(fft.fft2(signal)) if center else fft.fft2(signal)
    
    return result


def myIFFT2(Fsignal,center = True):
    
    result = fft.ifft2(fft.ifftshift(Fsignal)) if center else fft.ifftshift(Fsignal)
    
    return result


def kCoords(R,C,dx):
    
    dFx = 1.0/(dx*C)
    dFy = 1.0/(dx*R)
    
    #kx = bf.np.matrix(bf.np.arange(-C/2,C/2))*dFx
    #ky = bf.np.matrix(bf.np.arange(-R/2,R/2))*dFy
    
    kxP = (bf.np.arange(C/2)+1)*dFx
    kxM = bf.np.sort(-1*kxP)
    kx = bf.np.matrix(bf.np.concatenate((kxM,kxP)))
    
    kyP = (bf.np.arange(R/2)+1)*dFy
    kyM = bf.np.sort(-1*kyP)
    ky = bf.np.matrix(bf.np.concatenate((kyM,kyP)))
    
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
        

def phaseReconstr(ZaxisDer,R,C,Ifuoco,fselect,k=kD,z=zD,dx=dxD,alphaCorr=alphaCorrD,imgBitsPerPixel=8):
    
    deltax = dx
    ZderFFT=myFFT2(ZaxisDer,CTR)
    
    kx,ky = kCoords(R,C,1)
    
    kx[bf.np.where(kx == 0)] = zeroSubst
    ky[bf.np.where(ky == 0)] = zeroSubst
    fireInd=bf.np.where(Ifuoco == 0)
    Ifuoco=bf.np.float64(Ifuoco)
    
    V = bf.np.square(kx)+bf.np.square(ky)
    
    alpha = bf.np.max(V)*alphaCorr
    
    kxCorr = WxyFilterDenCorr(kx, alpha, fselect)
    kyCorr = WxyFilterDenCorr(ky, alpha, fselect)
    
    Wx = bf.np.divide(V,(bf.np.square(V)+kxCorr))
    Wy = bf.np.divide(V,(bf.np.square(V)+kyCorr))
    
    FilterX = bf.np.multiply(bf.np.multiply(kx,Wx),ZderFFT)
    IfftFiltX = myIFFT2(FilterX,CTR)
    divX = bf.np.divide(IfftFiltX,Ifuoco)
    divX[fireInd] = bf.np.max(divX[bf.np.where(Ifuoco != 0)])
    FFTdivX = myFFT2(divX,CTR)
    mulX = bf.np.multiply(bf.np.multiply(Wx,kx),FFTdivX)
    IFFTmulX = myIFFT2(mulX,CTR)
    IFFTmulX = -k*(1/z)*(1/((C*deltax)**2))*IFFTmulX
    
    FilterY = bf.np.multiply(bf.np.multiply(ky,Wy),ZderFFT)
    IfftFiltY = myIFFT2(FilterY,CTR)
    divY = bf.np.divide(IfftFiltY,Ifuoco)
    divY[fireInd] = bf.np.max(divY[bf.np.where(Ifuoco != 0)])
    FFTdivY = myFFT2(divY,CTR)
    mulY = bf.np.multiply(bf.np.multiply(Wy,ky),FFTdivY)
    IFFTmulY = myIFFT2(mulY,CTR)
    IFFTmulY = -k*(1/z)*(1/((R*deltax)**2))*IFFTmulY
    
    sumMul = IFFTmulX + IFFTmulY
    realSum = bf.np.real(sumMul)
    rSmax = bf.np.max(realSum)
    rSmin = bf.np.min(realSum)

    
    rangeCorrector = 2**imgBitsPerPixel-1
    
    pixelToRad = (rSmax-rSmin)/rangeCorrector
    
    rSnorm = ((realSum-rSmin)/(rSmax-rSmin)*float(rangeCorrector)).astype(int)
    
    return rSnorm, pixelToRad


def phaseReconstr_v2(ZaxisDer,R,C,Ifuoco,fselect,k=kD,z=zD,dx=dxD,alphaCorr=alphaCorrD,imgBitsPerPixel=8, onlyAguess = False):
    
    if z is not None:
        ZderFFT=myFFT2(ZaxisDer*k/z,CTR)
    else:
        ZderFFT=myFFT2(ZaxisDer*k,CTR)
    
    kx,ky = kCoords(R,C,dx)
    
    kx[bf.np.where(kx == 0)] = zeroSubst
    ky[bf.np.where(ky == 0)] = zeroSubst
    fireInd=bf.np.where(Ifuoco == 0)
    Ifuoco=bf.np.float64(Ifuoco)
    
    V = bf.np.square(kx)+bf.np.square(ky)
    
    print alphaCorr
    
    alpha = bf.np.max(V)*alphaCorr
    
    print alpha
    
    print bf.np.max(kx)
    print bf.np.min(kx)
    print bf.np.max(ky)
    print bf.np.min(ky)
    
    kxCorr = WxyFilterDenCorr(kx, alpha, fselect)
    kyCorr = WxyFilterDenCorr(ky, alpha, fselect)
    
    if len(alpha) == 1 and alpha[0] == 0.0:
        oneone = bf.np.matrix(bf.np.ones((R,C)))
        Wx = Wy = bf.np.divide(oneone,V)
        print 'ciao'
    else:
        Wx = bf.np.divide(V,(bf.np.square(V)+kxCorr))
        Wy = bf.np.divide(V,(bf.np.square(V)+kyCorr))
    
    FilterX = bf.np.multiply(bf.np.multiply(kx,Wx),ZderFFT)
    IfftFiltX = myIFFT2(FilterX,CTR)
    divX = bf.np.divide(IfftFiltX,Ifuoco)
    #divX[fireInd] = bf.np.max(divX[bf.np.where(Ifuoco != 0)])
    FFTdivX = myFFT2(divX,CTR)
    mulX = bf.np.multiply(bf.np.multiply(Wx,kx),FFTdivX)
    IFFTmulX = -1*myIFFT2(mulX,CTR)
    
    FilterY = bf.np.multiply(bf.np.multiply(ky,Wy),ZderFFT)
    IfftFiltY = myIFFT2(FilterY,CTR)
    divY = bf.np.divide(IfftFiltY,Ifuoco)
    #divY[fireInd] = bf.np.max(divY[bf.np.where(Ifuoco != 0)])
    FFTdivY = myFFT2(divY,CTR)
    mulY = bf.np.multiply(bf.np.multiply(Wy,ky),FFTdivY)
    IFFTmulY = -1*myIFFT2(mulY,CTR)
    
    sumMul = IFFTmulX + IFFTmulY
    realSum = bf.np.real(sumMul)

    #realSum = bf.np.unwrap(realSum)

    if not onlyAguess:
        rSmax = bf.np.max(realSum)
        rSmin = bf.np.min(realSum)
    
        rangeCorrector = 2**imgBitsPerPixel-1
    
        pixelToRad = (rSmax-rSmin)/rangeCorrector
    
        rSnorm = ((realSum-rSmin)/(rSmax-rSmin)*float(rangeCorrector)).astype(int)
    
        return rSnorm, pixelToRad
    
    else:
        return realSum
        

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


def AI(images, dz = zD, dx = dxD, k=kD, initPhase = None, errLim = 10**-6, iterLim = 20):
    
    N,R,C = bf.np.shape(images)
    
    kx,ky = kCoords(R,C,dx)
    
    kpq = bf.np.square(kx) + bf.np.square(ky) 
    
#    if N%2 == 0:
#        return []
    
    sqrtImgs = bf.np.sqrt(images)
    
    if initPhase == None:
        phiGuess = bf.np.zeros((R,C))
    else: 
        phiGuess = initPhase
        
    currIter = 0
    
    x = range(N)
    
    propList = x[(len(x)-1)/2:len(x)]+x[-2:-1*(len(x)+1):-1]+x[1:(len(x)-1)/2+1]
    
    print propList
    
    deltas = [(x-(N-1)/2)*dz for x in propList]
    
    err = bf.np.sum(images[(N-1)/2])**2
    #csiK = bf.np.multiply(sqrtImgs[(N-1)/2],(bf.np.cos(phiGuess) + bf.np.sin(phiGuess)*1j))
    errList = []
    
    while err > errLim and currIter < iterLim:
        
        for ind in range(len(propList)-1):
            csiK = bf.np.multiply(sqrtImgs[(N-1)/2],(bf.np.cos(phiGuess) + bf.np.sin(phiGuess)*1j))
            delta = deltas[ind+1]
            #delta = dz if propList[ind] < propList[ind+1] else -1*dz
            csiKp1 = propagateI(csiK,kpq,delta,k)
            csiKp1I = bf.np.square(csiKp1.real)+bf.np.square(csiKp1.imag)
            csiKp1P = bf.np.arctan2(csiKp1.imag,csiKp1.real)
            #if bf.np.equal(csiKp1I,images[propList[ind+1]]).all is False:
            #    csiK = bf.np.multiply(sqrtImgs[propList[ind+1]],(bf.np.cos(csiKp1P) + bf.np.sin(csiKp1P)*1j))
            #else:
            #    csiK = csiKp1
            err = (1.0/(R*C))*float(bf.np.sum(bf.np.square(bf.np.square(csiK.real)+bf.np.square(csiK.imag)-images[propList[ind+1]])))
            
            if err > errLim:
                csiKp1cR = sqrtImgs[propList[ind+1]]
                csiKp1c = bf.np.multiply(csiKp1cR,(bf.np.cos(csiKp1P) + bf.np.sin(csiKp1P)*1j))
                phiGuess = propagateBack(csiKp1c, kpq, delta, k)
                
            else:
                pass
        
        #err = (1.0/(R*C))*float(bf.np.sum(bf.np.square(bf.np.square(csiK.real)+bf.np.square(csiK.imag)-images[propList[ind+1]])))
        errList.append(err)
        currIter += 1
        print currIter
        print err
    
    phiGuess = bf.np.arctan2(csiK.imag,csiK.real)
    
    return phiGuess, errList


if __name__== '__main__':
    
    print 'Not for standalone use'
    
    sys.exit(0)
    
    
      
    