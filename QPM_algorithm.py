from QPM_utilities import *

lamD = 533.0 * (10**-9)
kD = (2 * bf.np.pi) / lamD


zD = 500 * (10**-9)
alphaCorrD = [1e-4]
dxD = 7.98 * (10**-8)

zeroSubst = 10**-9

CTR = True
        

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
    
    #kx,ky = kCoordsPrev(R,C,dx)
    kx,ky = kCoords(R,C,dx)
    
    kx[bf.np.where(kx == 0)] = zeroSubst
    ky[bf.np.where(ky == 0)] = zeroSubst
    fireInd=bf.np.where(Ifuoco == 0)
    Ifuoco=bf.np.float64(Ifuoco)
    
    V = bf.np.square(kx)+bf.np.square(ky)
    
    alpha = bf.np.max(V)*alphaCorr
    
    kxCorr = WxyFilterDenCorr(kx, alpha, fselect)
    kyCorr = WxyFilterDenCorr(ky, alpha, fselect)
    
    if len(alpha) == 1 and alpha[0] == 0.0:
        oneone = bf.np.matrix(bf.np.ones((R,C)))
        Wx = Wy = bf.np.divide(oneone,V)
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


def AI(images, dz = zD, dx = dxD, k=kD, initPhase = 'Test', errLim = 10**-6, iterLim = 20):
    
    N,R,C = bf.np.shape(images)
    
    kx,ky = kCoords(R,C,dx)
    
    kpq = bf.np.square(kx) + bf.np.square(ky) 
    
#    if N%2 == 0:
#        return []
    
    sqrtImgs = bf.np.sqrt(images)
    
    if initPhase == None:
        phiGuess = bf.np.zeros((R,C))
    elif initPhase == 'Test':
        phiGuess = phi = 0.95*((2*bf.np.pi/0.9)*(1.0-images[(N-1)/2])-bf.np.pi)
    else: 
        phiGuess = initPhase
        
    currIter = 0
    
    x = range(N)
    
    propList = x[(len(x)-1)/2:len(x)]+x[-2:-1*(len(x)+1):-1]+x[1:(len(x)-1)/2+1]
    
    deltas = [(x-(N-N%2)/2)*dz for x in propList]
    
    err = bf.np.sum(images[(N-N%2)/2])**2
    csiK = bf.np.multiply(sqrtImgs[(N-N%2)/2],(bf.np.cos(phiGuess) + bf.np.sin(phiGuess)*1j))
    errList = []
    
    while err > errLim and currIter < iterLim:
        
        for ind in range(len(propList)-1):
            #csiK = bf.np.multiply(sqrtImgs[(N-1)/2],(bf.np.cos(phiGuess) + bf.np.sin(phiGuess)*1j))
            #delta = deltas[ind+1]
            delta = dz if propList[ind] < propList[ind+1] else -1*dz
            csiKp1 = propagateI(csiK,kpq,delta,k)
            csiKp1I = bf.np.square(csiKp1.real)+bf.np.square(csiKp1.imag)
            csiKp1P = bf.np.arctan2(csiKp1.imag,csiKp1.real)
            
            #################################
            bf.sp.misc.imsave('{0}'.format(currIter)+'_{0}.jpg'.format(ind),bf.adjustImgRange(csiKp1I,255))
            #################################
            
            if bf.np.equal(csiKp1I,images[propList[ind+1]]).all is False:
                csiK = bf.np.multiply(sqrtImgs[propList[ind+1]],(bf.np.cos(csiKp1P) + bf.np.sin(csiKp1P)*1j))
            else:
                csiK = csiKp1
            #err = (1.0/(R*C))*float(bf.np.sum(bf.np.square(bf.np.square(csiK.real)+bf.np.square(csiK.imag)-images[propList[ind+1]])))
            
            #if err > errLim:
            #    csiKp1cR = sqrtImgs[propList[ind+1]]
            #    csiKp1c = bf.np.multiply(csiKp1cR,(bf.np.cos(csiKp1P) + bf.np.sin(csiKp1P)*1j))
            #    phiGuess = propagateBack(csiKp1c, kpq, delta, k)
                
            #else:
            #    pass
        
        err = (1.0/(R*C))*float(bf.np.sum(bf.np.square(bf.np.square(csiK.real)+bf.np.square(csiK.imag)-images[propList[ind+1]])))
        errList.append(err)
        currIter += 1
        print currIter
    
    phiGuess = bf.np.arctan2(csiK.imag,csiK.real)
    
    return phiGuess, errList


if __name__== '__main__':
    
    print 'Not for standalone use'
    
    sys.exit(0)
    
    
      
    