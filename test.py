import cv2
import numpy as np
from scipy import ndimage
import glob
path = "A:\Work\Projects\Hackathon\BCCD_Dataset-master\BCCD\JPEGImages"
f = open('result.txt','ab')
for i in range(0,411):
	filename = path+"\BloodImage_%05d.jpg" %i
	print i
	image = cv2.imread(filename)
	if image is None:
		print 'Image Not Found'
		continue

	# WBC
	
	blur = cv2.GaussianBlur(image[:,:,1], (5,5), 0)
	_, otsu =cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	cv2.imshow("otsu", otsu)

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
	cv2.imshow("WBC_blob", wbc_blob)
	cv2.imwrite(path+'%05d_wbc.jpg' %i, wbc_blob)






	cv2.imshow("images", np.hstack([output,image, wbc_blob]))

	
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
	f.write("{} platelets:{} wbc:{} \r\n".format(i,len(keypoints),len(keypoints_wbc)))
	im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imwrite(path+'%05d_platelet.jpg' %i, im_with_keypoints)
	#cv2.imwrite("total.jpg",total)
	cv2.waitKey(10)
	cv2.destroyAllWindows()
	

f.close()