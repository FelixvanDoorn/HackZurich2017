import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb



def findObstacles(im1):
	# Read image
	#im1 = cv2.imread("depth.jpg", cv2.IMREAD_GRAYSCALE)
	#color1 = cv2.imread("rgb.jpg")

	#im1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# Adaptive Threshold
	thresh = cv2.adaptiveThreshold(im1, 255,
	                            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
	                            thresholdType=cv2.THRESH_BINARY_INV,
	                            blockSize=21,
	                            C=2)

	#pdb.set_trace()
	# Morphology to close gaps
	se = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
	out = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)

	# Find holes
	mask = np.zeros_like(im1)
	cv2.floodFill(out[1:-1,1:-1].copy(), mask, (0,0), 255)
	mask = (1 - mask).astype('bool')

	# Fill holes
	out[mask] = 255

	# Find contours
	#contours,_ = cv2.findContours(out.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	image, contours, _ = cv2.findContours(out.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	#pdb.set_trace()

	# Filter out contours with less than certain area
	area = 500
	filtered_contours = filter(lambda x: cv2.contourArea(x) > area,
	                           contours)

	# Draw final contours
	final = np.zeros_like(im1)
	#cv2.drawContours(final, filtered_contours, -1, 255, -1)

	#get mean depth for each contour
	# if std to large try to segment 2 blobs
	# if area too small, remove
	for h,cnt in enumerate(contours):
	    con_mask = np.zeros(im1.shape,np.uint8)
	    cv2.drawContours(con_mask,[cnt],0,255,-1)
	    mean = cv2.mean(im1,mask = con_mask)
	    cv2.drawContours(final, [cnt], -1, mean, -1)
	return final

#draw the contour in the mean depth color

"""
new_image = color1.copy();
new_image[:,:,0] = 0.5*new_image[:,:,0] +4*final
#new_image = im1 + (final.astype(im1.dtype))
imgplot = plt.imshow(np.concatenate((new_image, color1),axis=1))
plt.pause(0.04)
pdb.set_trace()

#cv2.imshow('Shapes', final)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite('train1_final.png', final)
"""


"""
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
#params.minThreshold = 10;
#params.maxThreshold = 200;
 
# Filter by Area.
#params.filterByArea = True
#params.minArea = 150
 
#params.filterByColor=1;
# Filter by Circularity
#params.filterByCircularity = True
#params.minCircularity = 0.1
 
# Filter by Convexity
#params.filterByConvexity = True
#params.minConvexity = 0.87
 
# Filter by Inertia
#params.filterByInertia = True
#params.minInertiaRatio = 0.01
imgplot = plt.imshow(im)
plt.pause(1)
im=im*100;
detector = cv2.SimpleBlobDetector_create(params)
#detector = cv2.SimpleBlobDetector(params)
# Create a detector with the parameters
#detector = cv2.SimpleBlobDetector(params)
# Detect blobs.
keypoints = detector.detect(im)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
pdb.set_trace()
imgplot = plt.imshow(im_with_keypoints)
plt.pause(0.04)
# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
"""