import QPM_utilities as qu
from scipy import misc
import Image

class QPM_Volume(object):
    
    def __init__(self,images,bits,dz,dx,dy = None,zeroInd = None,itpMapYet = True,k='linear'):
        
        self.planes = images
        self.levels,self.rows,self.columns = qu.np.shape(images)
        self.dz = dz
        self.dx = dx
        self.dy = dy if dy != None else dx
        self.middle = zeroInd if zeroInd != None else (self.levels-self.levels%2)/2
        if itpMapYet: 
            self.itpMap = qu.interpImgs(self.planes,self.dz,self.middle,k)
        
    
    def __iter__(self):
        return self.plane.__iter__()
    
    
    def __getitem__(self, index):
        return self.planes[index]
    
        
    def getNewPlane(self,z):
        
        if z > (self.levels-self.middle)*self.dz and z < (self.levels-self.middle)*dz*-1:
            raise ValueError("Out of range")
            return None
        
        newPlane = qu.np.zeros([self.rows,self.columns])
        
        for r in range(self.rows):
            for c in range(self.columns):
                newPlane[r,c] = self.itpMap[r][c](z)
                
        return newPlane
    
    
    def generateItpMap(self,k = 'linear'):
        
        self.itpMap = qu.interpImgs(self.planes,self.dz,self.middle,k)
    

if __name__=='__main__':
    
    less = True
    
    paths = []
    
    for i in range(10):
        paths.append('g:\\Immagini\\neuroi_15\\neuroi 15000{0}.tif'.format(i))
        #paths.append('0{0}.tif'.format(i))
    for i in range(11):
        paths.append('g:\\Immagini\\neuroi_15\\neuroi 1500{0}.tif'.format(i+10))
        #paths.append(str(i+10)+'.tif')
        
    if less:
        
        paths = ['g:\\Immagini\\neuroi_15\\neuroi 150000.tif',
                 'g:\\Immagini\\neuroi_15\\neuroi 150010.tif',
                 'g:\\Immagini\\neuroi_15\\neuroi 150020.tif']
        
    img = []
    
    for p in paths:
        tempImg = Image.open(p)
        imgPreData = qu.np.array(tempImg.getdata())
        if len(qu.np.shape(imgPreData)) > 1:
            imgPreData = imgPreData[:,1]
        data = imgPreData.reshape(tempImg.size[::-1])
        img.append(data)
    
    l = len(img)
    
    z = (qu.np.arange(l)-(l-l%2)/2)*5e-7
    
    p = []
    
    for h in range(l-1):
        g = (z[h+1]+z[h])/2
        p.append(g)
        
    
    img = qu.np.array(img)
    
    #img = img[::-1]
    
    dz = 5e-7*(10**less)
    
    if less:
        p = qu.np.array([-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9])*(dz/10)
    
    vol = QPM_Volume(img,16,5e-6,0.789e-7)
    
    ciao = 0
    
    for i in p:
        
        prova = qu.adjustImgRange(vol.getNewPlane(i),255,8)
    
        saveMe = Image.fromarray(prova)#,'I;'+str(16))
    
        saveMe.save('g:\\Immagini\\neuroi_15\\p 15000{0}.tif'.format(ciao))
        
        ciao += 1
    
    print 'Not for Standalone use'
    
    
    
    