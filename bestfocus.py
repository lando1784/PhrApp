import numpy as np
import platform

if platform.system()=='Linux':
    from PIL import Image
else:
    import Image
    
from PIL import ImageFilter as filt
    
#import cv2 # ok
import os # di sistema
from os import listdir
from os import walk
from os.path import isfile, join
import sys # di sistema
from scipy import ndimage, signal
from numpy import uint16, uint8, uint32 #ok
import scipy as sp
import math

imgTypes = {8:uint8,16:uint16,32:uint32,12:uint16}

def findBestFocus(imageFiles):

    sumSarray=[]
    
    for imgF in imageFiles:
        img = Image.open(imgF)
        img = np.array(img.getdata()).reshape(img.size[::-1])
        res = mySobel(img,-1,mode='xy')
        imgS = res[2]
        sumS=np.sum(imgS)
        sumSarray.append(sumS)
    
    return sumSarray.index(max(sumSarray))


def findBestFocus_histNsobel(imageFiles):
    
    sumSarray = []
    
    for imgF in imageFiles:
        img = Image.open(imgF)
        img = np.array(img.getdata()).reshape(img.size[::-1])
        res = mySobel(img,-1,mode='xy')
        imgS = res[2]
        histo = np.histogram(imgS)
        print np.shape(histo)
        print histo[0]
        print histo[1]
        hStd = np.std(histo[0])
        sumSarray.append(hStd)
        
    return sumSarray.index(min(sumSarray))
        


def findBestFocus_diff(imageFiles):
    
    sumSarray = []
    cont = 0
    for f in imageFiles:
        img = Image.open(f)
        try:
            imgBlur = img.filter(filt.GaussianBlur(4))
        except:
            try:
                tempData = np.array(img.getdata()).reshape(img.size[::-1])
                tempData = adjustImgRange(tempData,255,8)
                img = Image.fromarray(tempData)
                imgBlur = img.filter(filt.GaussianBlur(4))
            except Exception, e:
                print e.message
                imgBlur = img
        imgSharp = img.filter
        imgSharp = img.filter(filt.SHARPEN)
        blurData = np.array(imgBlur.getdata()).reshape(imgBlur.size[::-1])
        sharpData = np.array(imgSharp.getdata()).reshape(imgSharp.size[::-1])
        diffData = sharpData - blurData
        sumDiff = np.sum(diffData)
        sumSarray.append(sumDiff)
        diffData2 = adjustImgRange(diffData,255,8)
        sp.misc.imsave('diff'+str(cont)+'.jpg',diffData2)
        cont += 1
    
    return sumSarray.index(max(sumSarray))


def findBestFocus_diffNsobel(imageFiles):
    
    sumSarray = []
    cont = 0
    for f in imageFiles:
        img = Image.open(f)
        try:
            imgBlur = img.filter(filt.GaussianBlur(4))
        except:
            try:
                tempData = np.array(img.getdata()).reshape(img.size[::-1])
                tempData = adjustImgRange(tempData,255,8)
                img = Image.fromarray(tempData)
                imgBlur = img.filter(filt.GaussianBlur(4))
            except Exception, e:
                print e.message
                imgBlur = img
        imgSharp = img.filter(filt.SHARPEN)
        blurData = np.array(imgBlur.getdata()).reshape(imgBlur.size[::-1])
        sharpData = np.array(imgSharp.getdata()).reshape(imgSharp.size[::-1])
        diffData = sharpData - blurData
        res = mySobel(diffData,-1,mode='xy')
        imgS = res[2]
        sumDiff = np.sum(imgS)
        sumSarray.append(sumDiff)
        diffData2 = adjustImgRange(imgS,255,8)
        sp.misc.imsave('diff'+str(cont)+'.jpg',diffData2)
        cont += 1
    
    return sumSarray.index(max(sumSarray))


def findBestFocus_golayNsobel(imageFiles):
    
    scale = 1
    delta = 0
    kernelS = 3
    sigX=0
    sigY=0
    ddepth = -1

    sumSarray=[]
    cont = 0
    
    for imgF in imageFiles:
        img = Image.open(imgF)
        img = np.array(img.getdata()).reshape(img.size[::-1])
        img = sgolay2d(img,7)
        res = mySobel(img,-1,mode='xy')
        imgS = res[2]
        sumS=np.sum(imgS)
        sumSarray.append(sumS)
        img2 = adjustImgRange(img,255,8)
        sp.misc.imsave('sg'+str(cont)+'.jpg',img2)
        cont += 1
    
    return sumSarray.index(max(sumSarray))
    
    
def findBestFocus_guobao(imageFiles):

    sumSarray=[]
    
    for imgF in imageFiles:
        img = Image.open(imgF)
        img = np.array(img.getdata()).reshape(img.size[::-1])
        res = mySobel(img,5,mode='xy')
        imgS = np.square(res[2])
        
        maxList = myGuoBaoMax(myGuoBaoDiv(8,8,imgS))
        
        sumS=np.sum(maxList)
        sumSarray.append(sumS)
    
    return sumSarray.index(max(sumSarray))
    
    

def atan2_2D(X,Y):
    
    res = np.zeros((np.shape(X)[0],np.shape(X)[1]))
    
    for i in range(0,np.shape(X)[0]):
        for j in range(0,np.shape(X)[1]):
            res[i][j] = math.atan2(Y[i][j], X[i][j])
            
    return res


def myGuoBaoDiv(mb,nb,M):
    
    m,n = np.shape(M)
    
    MstList = []
    
    for s in range(0,m/mb):
        for t in range(0,n/nb):
            MstTemp = np.zeros([mb,nb])
            for i in range(0,mb):
                for j in range(0,nb):
                    MstTemp[i,j] = M[i+s*mb,j+t*nb]
            MstList.append(MstTemp)
            
    return MstList

def myGuoBaoMax(MstList):
    
    maxList = []
    
    for mst in MstList:
        maxList.append(np.max(mst))
    
    return maxList
            

def adjustImgRange(img,newMax,bits = None):
        
        currImgMax = np.max(img)
        currImgMin = np.min(img)
        
        newImg = ((img-float(currImgMin))/(float(currImgMax)-float(currImgMin)))*newMax
        
        if bits == None:
            return newImg
        else:
            newImg2 = newImg.astype(imgTypes[bits])
            return newImg2
    
    
def mySobel(img,ksize,mode):
    
    xSobel3 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    xSobel5 = np.array([[-2,-1,0,1,2],[-3,-2,0,2,3],[-4,-3,0,3,4],[-3,-2,0,2,3],[-2,-1,0,1,2]])
    xSobel7 = np.array([[-3,-2,-1,0,1,2,3],[-4,-3,-2,0,2,3,4],[-5,-4,-3,0,3,4,5],[-6,-5,-4,0,4,5,6],[-5,-4,-3,0,3,4,5],[-4,-3,-2,0,2,3,4],[-3,-2,-1,0,1,2,3]])
    xScharr = np.array([[+3,0,-3],[+10,0,-10],[3,0,-3]])
    
    xSobels = {3:xSobel3,5:xSobel5,7:xSobel7,-1:xScharr}
    
    ySobel3 = np.transpose(xSobel3)
    ySobel5 = np.transpose(xSobel5)
    ySobel7 = np.transpose(xSobel7)
    yScharr = np.transpose(xScharr)
    
    ySobels = {3:ySobel3,5:ySobel5,7:ySobel7,-1:yScharr}
    
    if mode == 'x':
        dx = np.abs(signal.convolve2d(img,xSobels[ksize],mode='same'))
        dy = None
        tot = dx
        
    elif mode == 'y':
        dx = None
        dy = np.abs(signal.convolve2d(img,ySobels[ksize],mode='same'))
        tot = dy
        
    elif mode == 'xy':
        dx = np.abs(signal.convolve2d(img,xSobels[ksize],mode='same'))
        dy = np.abs(signal.convolve2d(img,ySobels[ksize],mode='same'))
        tot = np.hypot(dx,dy)
    else:
        dx = dy = tot = None
        
    return dx,dy,tot



def sgolay2d ( z, window_size, order=2, derivative=None):
    """
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return signal.fftconvolve(Z, -r, mode='valid'), signal.fftconvolve(Z, -c, mode='valid')



if __name__=='__main__':
    
    print 'Not for Standalone use'
    
    sys.exit(0)
    
    mypath='/home/lando/Documenti/Lavoro/Octave/neuroi_15/'
    mypath2='/home/lando/Documenti/Lavoro/Octave/diaframma-aperto-5-um/'
    mypath3='/home/lando/Documenti/Lavoro/Octave/EGbone60x2p5umAperto/'
    mypath4='/home/lando/Documenti/Lavoro/Octave/3_20x/'
    mypath5='/home/lando/Documenti/Lavoro/Octave/sandro/a/'
    
    usedpath=mypath5
    
    
    onlyfiles = [ f for f in listdir(usedpath) if isfile(join(usedpath,f)) ]
    
    imgPaths=[]
    
    for name in onlyfiles:
        imgPaths.append(usedpath+name)

    imgPaths.sort()
    testFocusInd = findBestFocus(imgPaths)
    testFocusInd2 = findBestFocusHist(imgPaths)
    testFocusInd3 = findBestFocusGradientX(imgPaths)
    
    print testFocusInd, testFocusInd2, testFocusInd3
    
    testFocusImg = cv2.imread(imgPaths[testFocusInd])
    
    print imgPaths[testFocusInd2]
    
    