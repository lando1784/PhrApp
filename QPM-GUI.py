
import os #di sistema
#import matplotlib #ok
#from matplotlib.figure import Figure #ok
import QPM_main as qpm #da progetto
import bestfocus as bf # da progetto
import sys # di sistema
import tiffClass as tc #da progetto
from mayavi import mlab #ok
import wx # ok
import subprocess as subp # di sistema
if qpm.platform.system() == 'Windows':
    try:
        import pywinauto #da scaricare
        pilot = True
    except:
        pilot = False
else:
    pilot = False
import time # di sistema
from subprocess import Popen #di sistema
from wx import Point

#matplotlib.interactive( True )
#matplotlib.use('WXAgg')
#from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
#from matplotlib.backends.backend_wx import NavigationToolbar2Wx

polyfitDer = False
zCorr = True

if not polyfitDer:
    zCorr = False
    
dimRet = True

class MainFrame(wx.Frame):

    def onOpen(self, event=None, altDir = None):

        if altDir is None:
        
            openFileDialog = wx.FileDialog(self, "Open", "", "", 
                                           "TIFF files (*.tif)|*.tif|TIFF files (*.tiff)|*.tiff",
                                           wx.MULTIPLE | wx.FD_FILE_MUST_EXIST)
            openFileDialog.ShowModal()
            
        try:
            if altDir is None:
                self.imagePaths = openFileDialog.GetPaths()
            else:
                
                onlyfiles = [ f for f in bf.listdir(altDir) if bf.isfile(bf.join(altDir,f)) ]
                
                self.imagePaths = []
                
                for name in onlyfiles:
                    self.imagePaths.append(altDir+'\\'+name)
                           
            self.imagePaths.sort()
            self.images = []
            self.currImgCbBox.Clear()
            cont=0
            self.currDirTxt.ChangeValue(os.path.dirname((self.imagePaths[0])))
            self.resImgDirTxt.ChangeValue(os.path.dirname((self.imagePaths[0])))
            self.imgInfo = tc.TiffInfo()
            self.imgInfo.tiff = self.imagePaths[0]
            self.currImgCbBox.SetSelection(0)
            for path in self.imagePaths:
                tempImg = bf.Image.open(path)
                imgPreData = bf.np.array(tempImg.getdata())
                if len(bf.np.shape(imgPreData)) > 1:
                    imgPreData = imgPreData[:,1]
                data = imgPreData.reshape(tempImg.size[::-1])
                if self.imgInfo.BitsPerSample == -1 or self.imgInfo.BitsPerSample == 0 or self.imgInfo.BitsPerSample > 32:
                    #imgDataType = str(imgPreData.dtype)
                    #self.BitsPerSample = int(imgDataType[-2]+imgDataType[-1])
                    self.BitsPerSample = 16
                else:
                    self.BitsPerSample = self.imgInfo.BitsPerSample
                data = data.astype(bf.imgTypes[self.BitsPerSample])
                self.images.append(data)
                self.currImgCbBox.Insert(path,cont)
                cont+=1
            self.currImgCbBox.SetSelection(0)
            self.imgSclBar.SetScrollbar(orientation = 0, position = 0, range = 1, thumbSize = len(self.imagePaths)-1)
            self.currImgNum.ChangeValue('1')
            self.alphaFuncCbBox.SetSelection(0)
            self.algCbBox.SetSelection(0)
            self.alphaNum2.ChangeValue('0')
            self.alphaNum2.SetEditable(False)
            self.bestFocusIndex = (len(self.imagePaths)+len(self.imagePaths)%2)/2-1
            self.bfIndexNum.SetValue(str(self.bestFocusIndex+1))
            self.radio1.SetValue(False)
            self.radio2.SetValue(True)
            self.res3Dimage = None
            
            self.drawMe(0)
            
            self.Bind(wx.EVT_SCROLL_CHANGED, self.onScroll, self.imgSclBar)
            self.Bind(wx.EVT_COMBOBOX, self.onCombo, self.currImgCbBox)
            self.Bind(wx.EVT_COMBOBOX, self.onCombo, self.alphaFuncCbBox)
            self.Bind(wx.EVT_RADIOBUTTON, self.onRadio, self.radio1)
            self.Bind(wx.EVT_RADIOBUTTON, self.onRadio, self.radio2)
            self.Bind(wx.EVT_CHECKBOX, self.onCheckFocus, self.bfImgCkBox)
            self.Bind(wx.EVT_BUTTON, self.onCreate3D, self.createImgBtn)
            self.Bind(wx.EVT_BUTTON, self.onView3D, self.ViewImgBtn)
            self.Bind(wx.EVT_BUTTON, self.onSave3D, self.saveImgBtn)
            self.Bind(wx.EVT_BUTTON, self.onDirSelect, self.selDirBtn)
        
        
        except Exception, e:

            wx.MessageBox("Error: " + str(e) + "\nError message: " + e.message)
        
        if altDir is None:    
            openFileDialog.Destroy()
        
        
    def onOpenNd2(self, event):
        
        openFileDialog = wx.FileDialog(self, "Open nd2", "", "", 
                                       "Nd2 files (*.nd2)|*.nd2",
                                       wx.FD_FILE_MUST_EXIST)
        openFileDialog.ShowModal()
        
        path = openFileDialog.GetPath()
        
        dirDialog = wx.DirDialog(self,"Select a directory for the exported files")
        dirDialog.ShowModal()
        dir_path = dirDialog.GetPath()
        
        
        myCursor= wx.StockCursor(wx.CURSOR_WAIT)
        self.SetCursor(myCursor)

        Popen(['C:\\Program Files (x86)\\NIS-Elements Viewer\\Viewer.exe', path])
        
        listBefore = pywinauto.findwindows.find_windows()
        
        time.sleep(10)
        
        listAfter = pywinauto.findwindows.find_windows()
        newWindow = 0
        controlNum = 0
        
        for ela in listAfter:
            controlNum = 0
            for elb in listBefore:
                if ela==elb:
                    controlNum += 1
            if controlNum == 0:
                newWindow = ela
                break
        
        w_handle = newWindow
        
        pwa_app = pywinauto.application.Application()
        
        pwa_app.connect_(handle = w_handle)
        
        pwa_app.top_window_().TypeKeys("%FE")
        pwa_app.top_window_().TypeKeys(dir_path)
        pwa_app.top_window_().TypeKeys("{TAB 6}")
        pwa_app.top_window_().TypeKeys("{UP 2}")
        pwa_app.top_window_().TypeKeys("{TAB 10}")
        pwa_app.top_window_().TypeKeys("{ENTER}")
        
        time.sleep(5)
        
        pwa_app.top_window_().TypeKeys("%{F4}")
        pwa_app.top_window_().TypeKeys("{ENTER}")
        
        myCursor= wx.StockCursor(wx.CURSOR_ARROW)
        self.SetCursor(myCursor)
        
        self.onOpen(altDir = dir_path)
        
                

    def onScroll(self, event):
        
        culprit = event.GetEventObject()
        self.currImgNum.ChangeValue(str(culprit.GetThumbPosition()+1))
        self.currImgCbBox.SetSelection(culprit.GetThumbPosition())
        if self.bestFocusIndex == culprit.GetThumbPosition():
            self.bfImgCkBox.SetValue(True)
        if self.bestFocusIndex != culprit.GetThumbPosition() and self.bfImgCkBox.IsChecked():
             self.bfImgCkBox.SetValue(False)
        self.drawMe(culprit.GetThumbPosition())

        
    def onCombo(self, event):
        
        culprit = event.GetEventObject()
        if culprit is self.currImgCbBox:
            self.currImgNum.ChangeValue(str(culprit.GetSelection()+1))
            self.imgSclBar.SetThumbPosition(culprit.GetSelection())
            if self.bestFocusIndex == culprit.GetSelection():
                self.bfImgCkBox.SetValue(True)
            if self.bestFocusIndex != culprit.GetSelection() and self.bfImgCkBox.IsChecked():
                self.bfImgCkBox.SetValue(False)
            self.drawMe(culprit.GetSelection())
        elif culprit is self.alphaFuncCbBox:
            if culprit.GetSelection() == 0:
                self.alphaNum2.ChangeValue('0')
                self.alphaNum2.SetEditable(False)
            else:
                self.alphaNum2.SetEditable(True)
                self.alphaNum2.ChangeValue('0.0001')
                
        
    
    def onDirSelect(self, event):
        
        res3DdirDialog = wx.DirDialog(self,"Select a directory")
        res3DdirDialog.ShowModal()
        self.resImgDirTxt.SetValue(res3DdirDialog.GetPath())


    def onTextinNumF(self, event):
        """
        check for numeric entry and limit to 3 decimals
        accepted result is in self.value
        """
        culprit = event.GetEventObject()
        
        raw_value = culprit.GetValue().strip()
        # numeric check
        if all(x in '0123456789.+-' for x in raw_value):
            # convert to float and limit to 2 decimals
            #self.value = round(float(raw_value), 6)
            #culprit.ChangeValue(str(self.value))
            pass
        else:
            culprit.ChangeValue("0.001")


    def onTextinNumI(self, event):
        """
        check for numeric entry and limit to 2 decimals
        accepted result is in self.value
        """
        culprit = event.GetEventObject()
        
        raw_value = culprit.GetValue().strip()
        # numeric check
        if all(x in '0123456789.+-' for x in raw_value):
            # convert to float and limit to 2 decimals
            value = int(raw_value)
            culprit.ChangeValue(str(value))
        else:
            culprit.ChangeValue("1")
        
        if culprit is self.currImgNum:
            culprit = event.GetEventObject()
            if int(culprit.GetValue()) > len(self.images):
                culprit.SetValue(str(len(self.images)))
            elif int(culprit.GetValue()) <= 0:
                culprit.SetValue('1')
            self.imgSclBar.SetThumbPosition(int(culprit.GetValue())-1)
            self.currImgCbBox.SetSelection(int(culprit.GetValue())-1)
            if self.bestFocusIndex == int(culprit.GetValue())-1:
                self.bfImgCkBox.SetValue(True)
            if self.bestFocusIndex != int(culprit.GetValue())-1 and self.bfImgCkBox.IsChecked():
                 self.bfImgCkBox.SetValue(False)
            self.drawMe(int(culprit.GetValue())-1)
            


    def onRadio(self, event):
        
        culprit = event.GetEventObject()
        
        if culprit is self.radio1:
            
            #self.bestFocusIndex = bf.findBestFocus_diffNsobel(self.imagePaths)
            #self.bestFocusIndex = bf.findBestFocus_diff(self.imagePaths)
            #self.bestFocusIndex = bf.findBestFocus_golayNsobel(self.imagePaths)
            #self.bestFocusIndex = bf.findBestFocus_guobao(self.imagePaths)
            self.bestFocusIndex = bf.findBestFocus(self.imagePaths)
            #self.bestFocusIndex = bf.findBestFocus_histNsobel(self.imagePaths)
            
            self.bfIndexNum.SetValue(str(self.bestFocusIndex+1))
            if self.bfImgCkBox.IsChecked() and self.bestFocusIndex != self.currImgCbBox.GetSelection():
                self.bfImgCkBox.SetValue(False)
            elif not self.bfImgCkBox.IsChecked() and self.bestFocusIndex == self.currImgCbBox.GetSelection():
                self.bfImgCkBox.SetValue(True)
    
    
    def onCheckFocus(self, event):
        
        if self.radio1.GetValue():
            self.radio1.SetValue(False)
            self.radio2.SetValue(True)
        
        self.bestFocusIndex = self.currImgCbBox.GetSelection()
        self.bfIndexNum.SetValue(self.currImgNum.GetValue())
        
        
    def onCreate3D(self, event):
        
        kN = (2 * bf.np.pi) / (float(self.lambdaNum.GetValue())*10**(-9))
        myCursor= wx.StockCursor(wx.CURSOR_WAIT)
        self.SetCursor(myCursor)
        self.degree = None
        path3d = self.resImgDirTxt.GetValue()+'\\'+self.resImgFileNameTxt.GetValue()+self.fileExtCbBox.GetValue()
        if polyfitDer and self.algCbBox.GetSelection() is not 1:
            degDef = 3 if len(self.images)>3 else len(self.images)-1
            degreeDialog = wx.TextEntryDialog(self,'Enter the polynomial fit degree:\n',caption='Degree Dialog', defaultValue=str(degDef))
            degreeDialog.ShowModal()
            self.degree = int(degreeDialog.GetValue())
            if zCorr:
                zder = qpm.ZaxisDerive_v3(self.images, self.bestFocusIndex,self.degree,z=float(self.zStepNum.GetValue())*(10**(-9)))
                zN = None
            else:
                zder = qpm.ZaxisDerive_v3(self.images, self.bestFocusIndex,self.degree)
                zN = float(self.zStepNum.GetValue())*(10**(-9))
        elif not polyfitDer and self.algCbBox.GetSelection() is not 1:
            zder = qpm.ZaxisDerive(self.images, self.bestFocusIndex)
            zN = float(self.zStepNum.GetValue())*(10**(-9))
        else:
            zN = float(self.zStepNum.GetValue())*(10**(-9))
        R,C = bf.np.shape(self.images[self.bestFocusIndex])
        deltaX = (float(self.xStepNum.GetValue())*(10**(-9)))
        if self.alphaFuncCbBox.GetSelection() is not 0:
            alphaPar = bf.np.array([float(self.alphaNum.GetValue()),float(self.alphaNum2.GetValue())])
        else:
            alphaPar = bf.np.array([float(self.alphaNum.GetValue())])
        
        if self.algCbBox.GetSelection() == 0:
            if dimRet:
                self.res3Dimage, pixelToRad = qpm.phaseReconstr_v2(zder,R,C,self.images[self.bestFocusIndex],fselect = self.alphaFuncCbBox.GetSelection(),k=kN,z=zN,dx=deltaX,alphaCorr=alphaPar,imgBitsPerPixel=self.BitsPerSample)
            else:
                self.res3Dimage, pixelToRad = qpm.phaseReconstr(zder,R,C,self.images[self.bestFocusIndex],fselect = self.alphaFuncCbBox.GetSelection(),k=kN,z=zN,dx=deltaX,alphaCorr=alphaPar,imgBitsPerPixel=self.BitsPerSample)
        else:
            if self.algCbBox.GetSelection() == 2:
                if dimRet:
                    phaseGuess = qpm.phaseReconstr_v2(zder,R,C,self.images[self.bestFocusIndex],fselect = self.alphaFuncCbBox.GetSelection(),k=kN,z=zN,dx=deltaX,alphaCorr=alphaPar,imgBitsPerPixel=self.BitsPerSample, onlyAguess = True)
                else:
                    phaseGuess = qpm.phaseReconstr(zder,R,C,self.images[self.bestFocusIndex],fselect = self.alphaFuncCbBox.GetSelection(),k=kN,z=zN,dx=deltaX,alphaCorr=alphaPar,imgBitsPerPixel=self.BitsPerSample, onlyAguess = True)
                self.res3Dimage = bf.adjustImgRange(qpm.AI(self.images,zN,deltaX,kN,phaseGuess,float(self.errLimNum.GetValue()),int(self.iterLimNum.GetValue())),2**self.BitsPerSample-1).astype(bf.imgTypes[self.BitsPerSample])
                pixelToRad = 1
            else:
                self.res3Dimage = bf.adjustImgRange(qpm.AI(self.images,zN,deltaX,kN,None,float(self.errLimNum.GetValue()),int(self.iterLimNum.GetValue())),(2**self.BitsPerSample-1)).astype(bf.imgTypes[self.BitsPerSample])
                pixelToRad = 1
        
        print pixelToRad
        nS = float(self.nSampleNum.GetValue())
        nM = float(self.nMedNum.GetValue())
        #self.radToHeight = qpm.lamD*(pixelToRad/(2*bf.np.pi))*(nS-nM)
        self.radToHeight = pixelToRad/(qpm.kD*(nS-nM))
        print self.radToHeight
        myCursor= wx.StockCursor(wx.CURSOR_ARROW)
        self.SetCursor(myCursor)
        
    
    def onSave3D(self, event):
      
        if qpm.platform.system() == "Windows":  
            path3D = self.resImgDirTxt.GetValue()+'\\'+self.resImgFileNameTxt.GetValue()+self.fileExtCbBox.GetValue()
        else:
            path3D = self.resImgDirTxt.GetValue()+'/'+self.resImgFileNameTxt.GetValue()+self.fileExtCbBox.GetValue()
            
        comment = str('Pixel to nm: ' + str(self.radToHeight) + '\ndX: ' + self.xStepNum.GetValue() + '\ndZ: ' + self.zStepNum.GetValue() + '\nnSample: ' + self.nSampleNum.GetValue() + '\nnMedium: ' + 
                   self.nMedNum.GetValue() + '\nAlpha function: ' + self.alphaFuncCbBox.GetValue() + '\nAlpha 1: ' + self.alphaNum.GetValue() + '\nAlpha 2: ' + self.alphaNum2.GetValue() + '\nImg num: ' + str(len(self.images)) + '\nFocus index: ' + str(self.bestFocusIndex) + '\nPolyFit Der: ' +
                   str(polyfitDer) + '\nWavelength: ' + self.lambdaNum.GetValue() + '\nPolynom deg: ' + str(self.degree) + '\nReal Z axis: ' + str(zCorr) + '\nCorrected dimensions: ' + str(dimRet))
        paramsSet = ([[(bf.adjustImgRange(self.res3Dimage,255)).astype(bf.uint8)],
                     [(bf.adjustImgRange(self.res3Dimage,255)).astype(bf.uint8)],
                     [(bf.adjustImgRange(self.res3Dimage,2**(self.BitsPerSample)-1)).astype(bf.imgTypes[self.BitsPerSample]),'I;'+str(self.BitsPerSample)],
                     [path3D,''],
                     [path3D,''],
                     [path3D,comment]
                     ]
                     if self.BitsPerSample == 16 else
                     [[(bf.adjustImgRange(self.res3Dimage,255)).astype(bf.uint8)],
                     [(bf.adjustImgRange(self.res3Dimage,255)).astype(bf.uint8)],
                     [(bf.adjustImgRange(self.res3Dimage,2**(self.BitsPerSample)-1)).astype(bf.imgTypes[self.BitsPerSample])],
                     [path3D,''],
                     [path3D,''],
                     [path3D,comment]
                     ])
        
        
        
        #bf.sp.misc.imsave(*paramsSet[self.fileExtCbBox.GetSelection()])
        img = bf.Image.fromarray(*paramsSet[self.fileExtCbBox.GetSelection()])
        img.save(paramsSet[self.fileExtCbBox.GetSelection()+3][0],description = paramsSet[self.fileExtCbBox.GetSelection()+3][1])
        
        
    def onView3D(self, event):
        
        bf.sp.misc.imsave('temp.jpg',bf.adjustImgRange(self.res3Dimage,255))
        showME = bf.Image.open('temp.jpg')
        showME.show()
        showME = bf.np.array(showME.getdata()).reshape(showME.size[::-1])
        showME = bf.adjustImgRange(showME,255,8)
        #bf.cv2.imshow('3D image',showME)
        x = bf.np.arange(bf.np.shape(showME)[0])
        y = bf.np.arange(bf.np.shape(showME)[1])
        mlab.surf(x,y,bf.np.array(showME[:,:]))

    
    def drawMe(self,ind):
        
        self.imgViewer.SetBitmap(wx.EmptyBitmap(bf.np.shape(self.images[ind])[0],bf.np.shape(self.images[ind])[1]))
        
        
        if self.BitsPerSample == 16:
            preTemp = bf.adjustImgRange(self.images[ind], 255.0)
        else:
            preTemp = self.images[ind]
            
        preTemp = self.reSize(preTemp,512)

        bf.sp.misc.imsave('temp.jpg',preTemp)
        img2 = wx.Image('temp.jpg',wx.BITMAP_TYPE_ANY)
        img2 = img2.ConvertToBitmap()
        
        try:
            self.imgViewer.SetBitmap(img2)
            self.p06.Layout()
        except Exception, e:
            print e
    
    
    def reSize(self, img, maxDim):
        
        Height = bf.np.shape(img)[0]
        Width = bf.np.shape(img)[1]
        
        convFact = float(maxDim)/bf.np.max(bf.np.array([Height,Width]))
        
        retImg = bf.sp.misc.imresize(img,(int(Height*convFact),int(Width*convFact)))
        
        return retImg
                
                
    def enhcBright(self,image,maxIntensity=255.0): 

        # Parameters for manipulating image data
        phi = 1
        theta = 1

        # Increase intensity such that
        # dark pixels become much brighter, 
        # bright pixels become slightly bright
        newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.5
        newImage0 = bf.np.array(newImage0,dtype=bf.np.uint8)
        
        return newImage0
                
    def createGrid(self):
        
        sizer = wx.GridBagSizer()
        self.p00 = wx.Panel(self) # pannello da (0,0) a (0,2) text box input
        self.p00s1 = wx.Panel(self)
        self.p00s2 = wx.Panel(self)
        self.p00s3 = wx.Panel(self)
        self.p01 = wx.Panel(self) # pannello da (1,0) a (2,0) radio button e radio button
        self.p01s1 = wx.Panel(self) # pannello da (2,0) a (2,3) num output
        self.p01s2 = wx.Panel(self)
        self.p01s3 = wx.Panel(self)
        self.p02 = wx.Panel(self) # pannello da (3,0) a (3,1) numerical input 
        self.p02a1 = wx.Panel(self) # pannello da (4,0) a (4,1) numerical input
        self.p02a2 = wx.Panel(self) # pannello da (5,0) a (5,1) numerical input
        self.p02a3 = wx.Panel(self) # pannello da (5,0) a (5,1) numerical input
        self.p02a4 = wx.Panel(self) # pannello da (5,0) a (5,1) numerical input
        self.p02a5 = wx.Panel(self) # pannello da (5,0) a (5,1) numerical input
        self.p03 = wx.Panel(self) # pannello da (6,0) a (6,1) text box input dropdown menu
        self.p04 = wx.Panel(self) # pannello da (7,0) a (7,1) text box input e button
        self.p05 = wx.Panel(self) # pannello da (8,0) a (8,1) button
        self.p05s1 = wx.Panel(self) # pannello da (9,0) a (9,1) button e button
        self.p06 = wx.Panel(self) # pannello da (0,3) a (6,6) immagine
        self.p07 = wx.Panel(self) # pannello da (8,3) a (8,5) dropdown menu
        self.p08 = wx.Panel(self) # pannello da (9,3) a (9,4) scrollbar e numerical input
        self.p09 = wx.Panel(self) # pannello da (8,6) a (9,6) check box
        
        

        sizer.Add(self.p00, pos=(0, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p00s1, pos=(1, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p00s2, pos=(2, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p00s3, pos=(3, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p01, pos=(4, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p01s1, pos=(5, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p01s2, pos=(6, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p01s3, pos=(7, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p02, pos=(8, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p02a1, pos=(9, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p02a2, pos=(10, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p02a3, pos=(11, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p02a4, pos=(12, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p02a5, pos=(13, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p03, pos=(14, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p04, pos=(15, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p05, pos=(16, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p05s1, pos=(17, 0), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p06, pos=(0, 3), span=(16,4), flag=wx.EXPAND)
        sizer.Add(self.p07, pos=(16, 3), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p08, pos=(17, 3), span=(1,3), flag=wx.EXPAND)
        sizer.Add(self.p09, pos=(16, 6), span=(2,1), flag=wx.EXPAND)
        
        sizer.AddGrowableCol(3)
        sizer.AddGrowableCol(4)
        sizer.AddGrowableCol(5)
        sizer.AddGrowableCol(6)
        sizer.AddGrowableRow(0)
        sizer.AddGrowableRow(1)
        sizer.AddGrowableRow(2)
        sizer.AddGrowableRow(3)
        sizer.AddGrowableRow(4)
        sizer.AddGrowableRow(5)
        sizer.AddGrowableRow(6)
        sizer.AddGrowableRow(7)
        sizer.AddGrowableRow(8)
        sizer.AddGrowableRow(9)
        sizer.AddGrowableRow(10)
        sizer.AddGrowableRow(11)
        sizer.AddGrowableRow(12)
        sizer.AddGrowableRow(13)
        sizer.AddGrowableRow(14)
        sizer.AddGrowableRow(15)
        
        return sizer
    
    
    def createNum(self,panel,labVal,size,mulNum,mulLab,defVal):
        numSizer = wx.BoxSizer(wx.HORIZONTAL)
        num = wx.TextCtrl(panel,size = tuple(size*[mulNum,1]))
        label = wx.StaticText(panel,size = tuple(size*[mulLab,1]),label=labVal)
        numSizer.Add(num,mulNum,wx.ALIGN_CENTER_VERTICAL)
        numSizer.Add(label,mulLab,wx.ALIGN_CENTER_VERTICAL)
        num.ChangeValue(defVal)
        
        return num,label,numSizer
    
    
    def createMenu(self):

        menuBar = wx.MenuBar()
        menuFile = wx.Menu()
        menu1 = wx.Menu()
        self.openFiles = menu1.Append(wx.ID_ANY, "&Open", "This the text in the Statusbar")
        self.Bind(wx.EVT_MENU, self.onOpen, self.openFiles)
        if pilot:
            self.openNd2Files = menu1.Append(wx.ID_ANY, "Open &Nd2", "This the text in the Statusbar")
            self.Bind(wx.EVT_MENU, self.onOpenNd2, self.openNd2Files)
        menuBar.Append(menu1, "&File")
        self.SetMenuBar(menuBar)
        

    def __init__(self):
        wx.Frame.__init__(self, None, title = "QPM User Interface" , size=(1000, 830), style = wx.DEFAULT_FRAME_STYLE)# ^ wx.VSCROLL)#, style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        
        sizeUnit = bf.np.array([80,42])
        
        self.createMenu()
        s = self.createGrid()

        text00Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.currDirTxt = wx.TextCtrl(self.p00,size = tuple(sizeUnit*[2,1]),style = wx.TE_MULTILINE)
        self.currDirLabel = wx.StaticText(self.p00,size = tuple(sizeUnit*[1,1]),label="Current folder")
        text00Sizer.Add(self.currDirTxt,2,wx.EXPAND)
        text00Sizer.Add(self.currDirLabel,1,wx.EXPAND)

        cbBox00s1Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.algCbBox = wx.ComboBox(self.p00s1,size = tuple(sizeUnit*[3,1]),choices=['Standard QPM','Iterative Algorithm', 'IA + Std QPM'])
        cbBox00s1Sizer.Add(self.algCbBox,1,wx.ALIGN_CENTER_VERTICAL)
        
        self.errLimNum,self.errLimLabel,num00s2Sizer = self.createNum(self.p00s2,'Max Error', sizeUnit, 1, 2, '0.00001')
        self.iterLimNum,self.iterLimLabel,num00s3Sizer = self.createNum(self.p00s3,'Max Iterations num', sizeUnit, 1, 2, '20')

        radio01Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.radio1 = wx.RadioButton(self.p01,size = tuple((sizeUnit*[3,1])/[2,1]), label="Autoforcus", style=wx.RB_GROUP)
        self.radio2 = wx.RadioButton(self.p01,size = tuple((sizeUnit*[3,1])/[2,1]), label="Manual")
        radio01Sizer.Add(self.radio1,1,wx.ALIGN_CENTER_VERTICAL)
        radio01Sizer.Add(self.radio2,1,wx.ALIGN_CENTER_VERTICAL)
        
        self.bfIndexNum,self.bfIndexLabel,num01s1Sizer = self.createNum(self.p01s1,'Focus image index', sizeUnit, 1, 2, '0')
        
        cbBox01s2Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.alphaFuncCbBox = wx.ComboBox(self.p01s2,size = tuple(sizeUnit*[3,1]),choices=['alpha*k^2','alpha*k^2 + alpha2*k','alpha*exp[-alpha2*k]'])
        cbBox01s2Sizer.Add(self.alphaFuncCbBox,1,wx.ALIGN_CENTER_VERTICAL)
        
        self.alphaNum,self.alphaLabel,num01s3Sizer = self.createNum(self.p01s3,'Alpha factor', sizeUnit, 1, 2, '0.0001')        
        self.alphaNum2,self.alphaLabel2,num02Sizer = self.createNum(self.p02,'Alpha factor 2', sizeUnit, 1, 2, '0.0001')
        self.zStepNum,self.zStepLabel,num02a1Sizer = self.createNum(self.p02a1,'Z step [nm]', sizeUnit, 1, 2, '500')        
        self.xStepNum,self.xStepLabel,num02a2Sizer = self.createNum(self.p02a2,'Image resolution [nm/px]', sizeUnit, 1, 2, '79.8')
        self.nSampleNum,self.nSampleLabel,num02a3Sizer = self.createNum(self.p02a3,'Sample refractive index', sizeUnit, 1, 2, '1.367')        
        self.nMedNum,self.nMedLabel,num02a4Sizer = self.createNum(self.p02a4,'Medium refractive index', sizeUnit, 1, 2, '1.333')        
        self.lambdaNum,self.lambdaLabel,num02a5Sizer = self.createNum(self.p02a5,'Wavelength [nm]', sizeUnit, 1, 2, '530')
                
        text03Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.resImgFileNameTxt = wx.TextCtrl(self.p03,size = tuple(sizeUnit*[2,1]))
        self.fileExtCbBox = wx.ComboBox(self.p03,size = tuple(sizeUnit*[1,1]),choices=['.jpg','.png','.tif'])
        text03Sizer.Add(self.resImgFileNameTxt,2,wx.ALIGN_CENTER_VERTICAL)
        text03Sizer.Add(self.fileExtCbBox,1,wx.ALIGN_CENTER_VERTICAL)
        self.resImgFileNameTxt.ChangeValue('filename')
        self.fileExtCbBox.SetSelection(0)
        
        text04Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.resImgDirTxt = wx.TextCtrl(self.p04,size = tuple(sizeUnit*[2,1]))
        self.selDirBtn = wx.Button(self.p04,size = tuple(sizeUnit*[1,1]), label="...")
        text04Sizer.Add(self.resImgDirTxt,2,wx.ALIGN_CENTER_VERTICAL)
        text04Sizer.Add(self.selDirBtn,1,wx.ALIGN_CENTER_VERTICAL)

        button05Sizer = wx.BoxSizer(wx.VERTICAL)
        self.createImgBtn = wx.Button(self.p05,size = tuple(sizeUnit*[3,1]), label="Create 3D Image")
        button05Sizer.Add(self.createImgBtn,1,wx.ALIGN_CENTER_VERTICAL)
        
        button05s1Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.ViewImgBtn = wx.Button(self.p05s1,size = tuple((sizeUnit*[3,1])/[2,1]), label="View 3D Image")
        self.saveImgBtn = wx.Button(self.p05s1,size = tuple((sizeUnit*[3,1])/[2,1]), label="Save 3D Image")
        button05s1Sizer.Add(self.ViewImgBtn,1,wx.ALIGN_CENTER_VERTICAL)
        button05s1Sizer.Add(self.saveImgBtn,1,wx.ALIGN_CENTER_VERTICAL)
        
        img06Sizer = wx.BoxSizer(wx.VERTICAL)
        self.imgViewer = wx.StaticBitmap(self.p06)
        img06Sizer.Add(self.imgViewer, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL | wx.ADJUST_MINSIZE, 10)
        
        cbBox07Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.currImgCbBox = wx.ComboBox(self.p07)
        cbBox07Sizer.Add(self.currImgCbBox,1,wx.ALIGN_CENTER_VERTICAL)
        
        sclBar08Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.imgSclBar = wx.ScrollBar(self.p08)
        self.currImgNum = wx.TextCtrl(self.p08) 
        sclBar08Sizer.Add(self.imgSclBar,2,wx.ALIGN_CENTER_VERTICAL)
        sclBar08Sizer.Add(self.currImgNum,1,wx.ALIGN_CENTER_VERTICAL)
        
        
        ckBox09Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.bfImgCkBox = wx.CheckBox(self.p09,label="Best Focused image")
        ckBox09Sizer.Add(self.bfImgCkBox,1,wx.ALIGN_CENTER)
        
        
        self.p00.SetSizer(text00Sizer)
        self.p00.Layout()
        self.p00s1.SetSizer(cbBox00s1Sizer)
        self.p00s1.Layout()
        self.p00s2.SetSizer(num00s2Sizer)
        self.p00s2.Layout()
        self.p00s3.SetSizer(num00s3Sizer)
        self.p00s3.Layout()
        self.p01.SetSizer(radio01Sizer)
        self.p01.Layout()
        self.p01s1.SetSizer(num01s1Sizer)
        self.p01s1.Layout()
        self.p01s2.SetSizer(cbBox01s2Sizer)
        self.p01s2.Layout()
        self.p01s3.SetSizer(num01s3Sizer)
        self.p01s3.Layout()
        self.p02.SetSizer(num02Sizer)
        self.p02.Layout()
        self.p02a1.SetSizer(num02a1Sizer)
        self.p02a1.Layout()
        self.p02a2.SetSizer(num02a2Sizer)
        self.p02a2.Layout()
        self.p02a3.SetSizer(num02a3Sizer)
        self.p02a3.Layout()
        self.p02a4.SetSizer(num02a4Sizer)
        self.p02a4.Layout()
        self.p02a5.SetSizer(num02a5Sizer)
        self.p02a5.Layout()
        self.p03.SetSizer(text03Sizer)
        self.p03.Layout()
        self.p04.SetSizer(text04Sizer)
        self.p04.Layout()
        self.p05.SetSizer(button05Sizer)
        self.p05.Layout()
        self.p05s1.SetSizer(button05s1Sizer)
        self.p05s1.Layout()
        self.p06.SetSizer(img06Sizer)
        self.p06.Layout()
        self.p07.SetSizer(cbBox07Sizer)
        self.p07.Layout()
        self.p08.SetSizer(sclBar08Sizer)
        self.p08.Layout()
        self.p09.SetSizer(ckBox09Sizer)
        self.p09.Layout()
        
        self.Bind(wx.EVT_TEXT, self.onTextinNumF, self.alphaNum)
        self.Bind(wx.EVT_TEXT, self.onTextinNumF, self.alphaNum2)
        self.Bind(wx.EVT_TEXT, self.onTextinNumI, self.iterLimNum)
        self.Bind(wx.EVT_TEXT, self.onTextinNumF, self.errLimNum)
        self.Bind(wx.EVT_TEXT, self.onTextinNumF, self.xStepNum)
        self.Bind(wx.EVT_TEXT, self.onTextinNumI, self.currImgNum)
        self.Bind(wx.EVT_TEXT, self.onTextinNumI, self.zStepNum)
        self.Bind(wx.EVT_TEXT, self.onTextinNumF, self.nSampleNum)
        self.Bind(wx.EVT_TEXT, self.onTextinNumF, self.nMedNum)
        self.Bind(wx.EVT_TEXT, self.onTextinNumI, self.lambdaNum)
            
        
        self.SetAutoLayout(True)
        self.SetSizer(s)
        self.Layout()
        
        
if __name__ == '__main__':
    wx.SetDefaultPyEncoding('utf8') 
    myapp = wx.App(redirect=False)
    
    mf = MainFrame()
    mf.Show()
    mf.CenterOnScreen()
    
    myapp.MainLoop()