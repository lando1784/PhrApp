from QPM_utilities import *
from scipy import rand
from PIL import Image as im

lamD = 533.0 * (10**-9)
kD = (2 * np.pi) / lamD


zD = 500 * (10**-9)
alphaCorrD = [1e-4]
dxD = 7.98 * (10**-8)

zeroSubst = 10**-9

CTR = True
        

def phaseReconstr(ZaxisDer,R,C,Ifuoco,fselect,k=kD,z=zD,dx=dxD,alphaCorr=alphaCorrD,imgBitsPerPixel=8):
    
    deltax = dx
    ZderFFT=myFFT2(ZaxisDer,CTR)
    
    kx,ky = kCoords(R,C,1)
    
    kx[np.where(kx == 0)] = zeroSubst
    ky[np.where(ky == 0)] = zeroSubst
    fireInd=np.where(Ifuoco == 0)
    Ifuoco=np.float64(Ifuoco)
    
    V = np.square(kx)+np.square(ky)
    
    alpha = np.max(V)*alphaCorr
    
    kxCorr = WxyFilterDenCorr(kx, alpha, fselect)
    kyCorr = WxyFilterDenCorr(ky, alpha, fselect)
    
    Wx = np.divide(V,(np.square(V)+kxCorr))
    Wy = np.divide(V,(np.square(V)+kyCorr))
    
    FilterX = np.multiply(np.multiply(kx,Wx),ZderFFT)
    IfftFiltX = myIFFT2(FilterX,CTR)
    divX = np.divide(IfftFiltX,Ifuoco)
    divX[fireInd] = np.max(divX[np.where(Ifuoco != 0)])
    FFTdivX = myFFT2(divX,CTR)
    mulX = np.multiply(np.multiply(Wx,kx),FFTdivX)
    IFFTmulX = myIFFT2(mulX,CTR)
    IFFTmulX = -k*(1/z)*(1/((C*deltax)**2))*IFFTmulX
    
    FilterY = np.multiply(np.multiply(ky,Wy),ZderFFT)
    IfftFiltY = myIFFT2(FilterY,CTR)
    divY = np.divide(IfftFiltY,Ifuoco)
    divY[fireInd] = np.max(divY[np.where(Ifuoco != 0)])
    FFTdivY = myFFT2(divY,CTR)
    mulY = np.multiply(np.multiply(Wy,ky),FFTdivY)
    IFFTmulY = myIFFT2(mulY,CTR)
    IFFTmulY = -k*(1/z)*(1/((R*deltax)**2))*IFFTmulY
    
    sumMul = IFFTmulX + IFFTmulY
    realSum = np.real(sumMul)
    rSmax = np.max(realSum)
    rSmin = np.min(realSum)

    
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
    
    kx[np.where(kx == 0)] = zeroSubst
    ky[np.where(ky == 0)] = zeroSubst
    fireInd=np.where(Ifuoco == 0)
    Ifuoco=np.float64(Ifuoco)
    
    V = np.square(kx)+np.square(ky)
    
    alpha = np.max(V)*alphaCorr
    
    kxCorr = WxyFilterDenCorr(kx, alpha, fselect)
    kyCorr = WxyFilterDenCorr(ky, alpha, fselect)
    
    if len(alpha) == 1 and alpha[0] == 0.0:
        oneone = np.matrix(np.ones((R,C)))
        Wx = Wy = np.divide(oneone,V)
    else:
        Wx = np.divide(V,(np.square(V)+kxCorr))
        Wy = np.divide(V,(np.square(V)+kyCorr))
    
    FilterX = np.multiply(np.multiply(kx,Wx),ZderFFT)
    IfftFiltX = myIFFT2(FilterX,CTR)
    divX = np.divide(IfftFiltX,Ifuoco)
    divX[fireInd] = np.max(divX[np.where(Ifuoco != 0)])
    FFTdivX = myFFT2(divX,CTR)
    mulX = np.multiply(np.multiply(Wx,kx),FFTdivX)
    IFFTmulX = -1*myIFFT2(mulX,CTR)
    
    FilterY = np.multiply(np.multiply(ky,Wy),ZderFFT)
    IfftFiltY = myIFFT2(FilterY,CTR)
    divY = np.divide(IfftFiltY,Ifuoco)
    divY[fireInd] = np.max(divY[np.where(Ifuoco != 0)])
    FFTdivY = myFFT2(divY,CTR)
    mulY = np.multiply(np.multiply(Wy,ky),FFTdivY)
    IFFTmulY = -1*myIFFT2(mulY,CTR)
    
    sumMul = IFFTmulX + IFFTmulY
    realSum = np.real(sumMul)

    if not onlyAguess:
        rSmax = np.max(realSum)
        rSmin = np.min(realSum)
    
        rangeCorrector = 2**imgBitsPerPixel-1
    
        pixelToRad = (rSmax-rSmin)/rangeCorrector
    
        #rSnorm = ((realSum-rSmin)/(rSmax-rSmin)*float(rangeCorrector)).astype(int)
        rSnorm =adjustImgRange(realSum,rangeCorrector,imgBitsPerPixel)
    
        return rSnorm, pixelToRad
    
    else:
        return realSum


def AI(images, dz = zD, dx = dxD, k=kD, initPhase = 'Test', errLim = 10**-6, iterLim = 20):
    
    N,R,C = np.shape(images)
    
    kx,ky = kCoords(R,C,dx)
    
    kpq = np.square(kx) + np.square(ky) 

#######################################################
    
    I = adjustImgRange(images[(N-1)/2],1)
    
#######################################################
        
    sqrtImgs = np.sqrt(images)
    
    if initPhase == None:
        phiGuess = np.zeros((R,C))
        #phiGuess = rand(R,C)*np.pi*2
        #phiGuess = np.ones((R,C))*np.pi*2
        
    elif initPhase == 'Test':
        phiGuess = phi = 0.95*((2*np.pi/0.9)*(1.0-I)-np.pi)
    else: 
        phiGuess = initPhase
        
    currIter = 0
    
    x = range(N)
    
    propList = x[(len(x)-1)/2:len(x)]+x[-2:-1*(len(x)+1):-1]+x[1:(len(x)-1)/2+1]
    
    errList = []
    
    deltas = [(x-(N-N%2)/2)*dz*-1 for x in propList] # Mio
    
    err = np.sum(images[(N-N%2)/2])**2
    #csiK = np.multiply(sqrtImgs[(N-N%2)/2],(np.cos(phiGuess) + np.sin(phiGuess)*1j)) # Normale
    
    while err > errLim and currIter < iterLim:
        
        print currIter
        
        for ind in range(len(propList)-1):
            csiK = np.multiply(sqrtImgs[(N-1)/2],(np.cos(phiGuess) + np.sin(phiGuess)*1j)) # Mio
            delta = deltas[ind+1] # Mio
            #delta = dz if propList[ind] < propList[ind+1] else -1*dz # Normale
            csiKp1 = propagateI(csiK,kpq,delta,k,CTR)
            csiKp1I = np.square(csiKp1.real)+np.square(csiKp1.imag)
            csiKp1P = np.arctan2(csiKp1.imag,csiKp1.real)
            
            
            ###################################################################################
            
            #phi = adjustImgRange(csiKp1P,2**(16)-1,16)
            #imgP = im.fromarray(phi,'I;'+str(16))
            #imgP.save('phi_'+str(currIter)+'_'+str(ind)+'.tif')
            
            ###################################################################################
            
            #if np.equal(csiKp1I,images[propList[ind+1]]).all() == False: # Normale
            #    csiK = np.multiply(sqrtImgs[propList[ind+1]],(np.cos(csiKp1P) + np.sin(csiKp1P)*1j)) # Normale
            #else: # Normale
            #    csiK = csiKp1 # Normale
            err = (1.0/(R*C))*float(np.sum(np.square(np.square(csiKp1.real)+np.square(csiKp1.imag)-images[propList[ind+1]]))) # Mio
            
            #print err # Mio
            
            if err > errLim: # Mio
                csiKp1cR = sqrtImgs[propList[ind+1]] # Mio
                csiKp1c = np.multiply(csiKp1cR,(np.cos(csiKp1P) + np.sin(csiKp1P)*1j)) # Mio
                #phiGuess = propagateBack(csiKp1c, kpq, delta, k, CTR) # Mio
                csiKc = propagateI(csiKp1c,kpq,-1*delta,k,CTR) # Mio ma boh...
                normCsiKc = np.divide(csiKc,sqrtImgs[(N-1)/2]+0.0000001) # Mio ma boh...
                phiGuess = np.arctan2(normCsiKc.imag,normCsiKc.real) # Mio ma boh...
                
            else: # Mio
                pass # Mio
        
        err = (1.0/(R*C))*float(np.sum(np.square(np.square(csiK.real)+np.square(csiK.imag)-images[propList[ind+1]])))
        
        errList.append(err)
        
        print err
        currIter += 1
    
    phiGuess = np.arctan2(csiK.imag,csiK.real)
    
    return phiGuess,errList


if __name__== '__main__':
    
    print 'Not for standalone use'
    
    sys.exit(0)
    
    
      
    