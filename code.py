import sys
from PySide import QtGui
from PySide import QtCore
from PIL import Image
from random import randint
import cv2
import numpy as np
from scipy import ndimage
import glob
from matplotlib import pyplot as plt
import time
class MainWindow(QtGui.QWidget):
    
    def __init__(self):
        super(MainWindow, self).__init__()
        
        self.initUI()
        
    def initUI(self):

        print "Initializing"
        
        # File Pick Layout
        filePick = QtGui.QHBoxLayout()

        # Create a label which displays the path to our chosen file
        self.fileLabel = QtGui.QLabel('No file selected')
        filePick.addWidget(self.fileLabel)

        # Create a push button labelled 'choose' and add it to our layout
        fileBtn = QtGui.QPushButton('Choose file', self)
        filePick.addWidget(fileBtn)
        
        # Connect the clicked signal to the get_fname handler
        self.connect(fileBtn, QtCore.SIGNAL('clicked()'), self.get_fname)
        
        #Set the image to be blank at first
        pixmap = QtGui.QPixmap()
        self.imageLabel = QtGui.QLabel()
        self.imageLabel.setPixmap(pixmap)
        self.imageString = ""
        self.sliderValue = 0
        
        #Define layout boxes
        bottomBox = QtGui.QHBoxLayout()
        topBox = QtGui.QHBoxLayout()
        sideBox = QtGui.QVBoxLayout()
        sideBoxPad = QtGui.QHBoxLayout()
        
        #File picker on top
        topBox.addStretch(1)
        topBox.addLayout(filePick)
        topBox.addStretch(1)
        
        #Image in middle/right
        bottomBox.addStretch(1)
        bottomBox.addWidget(self.imageLabel)
        bottomBox.addStretch(1)
        
        #Buttons on left
        normalButton = QtGui.QPushButton("WBC/Platelets")
        self.connect(normalButton, QtCore.SIGNAL('clicked()'), self.wbc_platelet)
        secondButton = QtGui.QPushButton("Malaria RBC Count")
        self.connect(secondButton, QtCore.SIGNAL('clicked()'), self.rbc)

        #Slider
        sld = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        sld.setFocusPolicy(QtCore.Qt.NoFocus)
        sld.setGeometry(30, 40, 100, 50)
        sld.valueChanged[int].connect(self.change_slider)
        
        sideBox.addStretch(1)
        sideBox.addWidget(normalButton)
        sideBox.addStretch(1)
        sideBox.addWidget(secondButton)
        # sideBox.addWidget(blurButton)
        # sideBox.addWidget(contrastButton)
        # sideBox.addWidget(edgeDetectButton)
        # sideBox.addWidget(invertButton)
        # sideBox.addWidget(multiplyButton)
        # sideBox.addWidget(sharpenButton)
        # sideBox.addWidget(twoToneButton)
        sideBox.addWidget(sld)
        sideBox.addStretch(1)
        sideBoxPad.addStretch(1)
        sideBoxPad.addLayout(sideBox)
        sideBoxPad.addStretch(1)
        
        #Set grid layout
        grid = QtGui.QGridLayout()
        grid.addLayout(topBox, 0, 1)
        grid.addLayout(sideBoxPad, 1, 0)
        grid.addLayout(bottomBox, 1, 1)
        self.setLayout(grid)  
        
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Image Processing Functions')
        
        self.show()
      
    #File Picker Function    
    def get_fname(self):
        """
        Handler called when 'choose file' is clicked
        """
        # When you call getOpenFileName, a file picker dialog is created
        # and if the user selects a file, it's path is returned, and if not
        # (ie, the user cancels the operation) None is returned
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Select file')
        print fname
        if fname:
            self.fileLabel.setText(fname[0])
            self.load_image(fname[0])
        else:
            self.fileLabel.setText("No file selected")
            
    #Load new image function        
    def load_image(self, filepath):
        #Load the image into the label
        print "Loading Image"
        pixmap = QtGui.QPixmap(filepath)
        self.imageLabel.setPixmap(pixmap)
        self.imageString = filepath
        
    #Load new image function        
    def set_image(self, image):
        #Load the image into the label
        self.imageLabel.setPixmap(image)
        
     #Load normal image       
    def normal_image(self):
        #Load the image into the label
        print "Loading Image"
        pixmap = QtGui.QPixmap(self.imageString)
        self.imageLabel.setPixmap(pixmap)
        
     #Slider Changed     
    def change_slider(self, value):
        self.sliderValue = value
        
    def wbc_platelet(self):
        # WBC
        image = cv2.imread(self.imageString)
        blur = cv2.GaussianBlur(image[:,:,1], (5,5), 0)
        _, otsu =cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        #cv2.imshow("otsu", otsu)

        kernel = np.ones((3,3),np.uint8)
        #kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
        erosion = cv2.erode(otsu,kernel,iterations = 5)

        '''
        cv2.imshow("orig", image)
        cv2.imshow("test", erosion)
        cv2.imwrite("test.jpg", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        #blue
        blue = image.copy()
        blue[:,:,1]=0
        blue[:,:,2]=0
        #cv2.imshow("blue", blue)

        #green
        green = image.copy()
        green[:,:,0]=0
        green[:,:,2]=0
        #cv2.imshow("green", green)

        #red
        red = image.copy()
        red[:,:,0]=0
        red[:,:,1]=0
        #cv2.imshow("red", red)


        #threshold
        ret,thresh1 = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
        ret,thresh3 = cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
        ret,thresh4 = cv2.threshold(image,127,255,cv2.THRESH_TOZERO)
        ret,thresh5 = cv2.threshold(image,127,255,cv2.THRESH_TOZERO_INV)
        thresh1_blue = thresh1.copy()
        thresh1_blue[:,:,0]=0
        thresh1_blue[:,:,2]=0
        ################WBC##############################################
        kernel = np.ones((3,3),np.uint8)
        thresh1 = cv2.dilate(thresh1,kernel,iterations = 5)

        #thresh1[:,:,1]=0
        #thresh1[:,:,2]=0
        thresh1 = cv2.GaussianBlur(thresh1, (5,5), 0)

        #extracting blue color
        lower = np.array([0,0,0], dtype="uint8")
        upper = np.array([255,0,0], dtype="uint8")
        mask = cv2.inRange(thresh1, lower, upper)
        output = cv2.bitwise_and(thresh1, thresh1, mask = mask)

        kernel = np.ones((7,7),np.uint8)
        output = cv2.erode(output ,kernel,iterations = 5)
        output = cv2.dilate(output ,kernel,iterations = 5)
        output = cv2.dilate(output ,kernel,iterations = 5)

        #Blob
        #ret,blob_bw = cv2.threshold(output,100,255,cv2.THRESH_TOZERO_INV)
        blob_bw = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        _, blob_thresh =cv2.threshold(blob_bw,10 , 255, cv2.THRESH_BINARY)
        #cv2.imshow("blob_thresh", blob_thresh)
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 255
        # Filter by Area.
        params.filterByArea = False
        params.minArea = 0
        params.maxArea = 10000
        # Filter by Circularity
        params.filterByCircularity = False
        #params.minCircularity = 0.1
        # Filter by Convexity
        params.filterByConvexity = False
        #params.minConvexity = 0.87
        # Filter by Inertia
        params.filterByInertia = False
        #params.minInertiaRatio = 0.01
        params.filterByColor = False
        params.blobColor = 255
        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else : 
            detector = cv2.SimpleBlobDetector_create(params)
        # Detect blobs.
        keypoints_wbc = detector.detect(blob_thresh)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob

        wbc_blob = cv2.drawKeypoints(image, keypoints_wbc, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show blobs
        #cv2.imshow("WBC_blob", wbc_blob)
        #cv2.imshow("images", np.hstack([output,image, wbc_blob]))

        
        # PLATELETS
        
        blur = cv2.GaussianBlur(image[:,:,1],(5,5),0)
        ret3,green = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #ret3,green = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\cv2.THRESH_BINARY_INV,11,2)
        ret3,blue = cv2.threshold(image[:,:,2],150,255,cv2.THRESH_BINARY_INV)
        total = green + blue
        kernel = np.ones((3,3), np.uint8)
        img_erosion = cv2.erode(image[:,:,1], kernel, iterations=3)
        eroded = cv2.erode(blue,kernel, iterations = 3)
        opened = cv2.dilate(eroded,kernel,iterations= 3)
        opened = cv2.subtract(opened, blob_thresh)
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 255   
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 1000    
        params.filterByCircularity = False
        params.filterByInertia = True
        params.filterByConvexity = False
        params.filterByColor = True
        params.blobColor = 255

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(opened)
        #f.write("{} platelets:{} wbc:{} \r\n".format(i,len(keypoints),len(keypoints_wbc)))
        im_with_keypoints = cv2.drawKeypoints(wbc_blob, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (0,30)
        fontScale              = 1
        fontColor              = (0,0,255)
        lineType               = 2

        cv2.putText(im_with_keypoints,"platelets:{}".format(len(keypoints)), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (0,60)
        fontScale              = 1
        fontColor              = (0,255,0)
        lineType               = 2

        cv2.putText(im_with_keypoints,"wbc:{}".format(len(keypoints)), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        #cv2.imwrite("total.jpg",total)
        cv2.imshow('IMAGE WITH KEYPOINTS', im_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def rbc(self):
        img = cv2.imread(self.imageString)
        I = img # this variable will be used later


        # In[208]:

        I2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # In[209]:

        ret, thresh1 = cv2.threshold(I2,50,255,cv2.THRESH_BINARY_INV)


        # In[210]:

        I3 = cv2.medianBlur(I2, 3)


        # In[211]:

        I31 = cv2.equalizeHist(I3)


        # In[212]:

        # In[213]:

        bluecells = I[:, :, 0] - 0.5*I[:, :, 2] - 0.5*I[:, :, 1]
        m, n = np.shape(bluecells)
        temp = np.zeros((m, n))
        for i in range(0,m):
            for j in range(0,m):
                if (bluecells[i,j] > 0):
                    temp[i,j] = 1


        # In[214]:

        # normalize bluecells
        bluecells = (bluecells - np.min(bluecells))/(np.max(bluecells) - np.min(bluecells))


        # In[215]:

        Blue = bluecells > 0.62
        Blue = (Blue*1*255).astype(np.uint8)
        #Blue = cv2.medianBlur(Blue*1, 3)
        #Blue = cv2.morphologyEx((Blue*1).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        kernel = np.ones((5,5),np.uint8)
        Blue = cv2.erode(Blue, kernel,iterations = 4)
        Blue = cv2.dilate(Blue,kernel,iterations = 4)

        # In[216]:

        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 255   
        params.filterByArea = True
        params.minArea = 1000
        params.maxArea = 100000   
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByColor = True
        params.blobColor = 255

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(Blue)
        I_copy = I
        im_with_keypoints = cv2.drawKeypoints(I_copy, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        infected = len(keypoints)
        print("Infected RBC = ", infected)


        # In[217]:

        #kernel = np.ones((3,3),np.uint8)
        #Blue = cv2.erode((Blue*1).astype(np.uint8),kernel,iterations = 5)
        #NRem = cv2.dilate(Blue,kernel,iterations = 5)
        #NRem = ndimage.binary_opening(Blue, iterations = 8)


        # In[218]:

        ## Add Intensity Adjustment here ("imadjust" is the function in matlab)


        # In[219]:

        #applied threshold on I31 here instead of I2
        ret2, I6 = cv2.threshold(I3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        # In[220]:

        #I7 = ndimage.binary_closing(I6, iterations = 15)
        kernel = np.ones((3,3),np.uint8)
        Blue = cv2.erode(I6, kernel,iterations = 5)
        I7 = cv2.dilate(Blue,kernel,iterations = 5)


        # In[221]:

        I7 = cv2.GaussianBlur(I7, (3,3), 0)

        # In[222]:




        # In[223]:

        des = cv2.bitwise_not(I7)
        im2,contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contour:
            cv2.drawContours(des,[cnt],0,255,-1)

        I7 = cv2.bitwise_not(des)


        # In[224]:

        I7 = 255-I7


        # In[225]:

        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 255   
        params.filterByArea = True
        params.minArea = 10000
        params.maxArea = 1000000    
        params.filterByCircularity = False
        params.filterByInertia = True
        params.filterByConvexity = False
        params.filterByColor = True
        params.blobColor = 255

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(I7)

        im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        rbc_count = len(keypoints)


        # In[227]:

        healthy_rbc = rbc_count - infected
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (0,80)
        fontScale              = 3
        fontColor              = (0,255,0)
        lineType               = 2

        cv2.putText(im_with_keypoints,"rbc count: {} infected: {} healthy_rbc:{}".format(rbc_count,infected,healthy_rbc), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        #cv2.imwrite("total.jpg",total)
        im_with_keypoints = cv2.resize(im_with_keypoints,(1280,720))
        cv2.imshow('IMAGE WITH KEYPOINTS', im_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()