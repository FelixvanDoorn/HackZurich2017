
# import numpy as np
# import cv2
# import time


# WIDTH = 1280
# HEIGHT = 720

# cap = cv2.VideoCapture("../testimages/normal0.avi")


# while(cap.isOpened()):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     if not ret:
#         break

#     try:
#         # Display the resulting frame
#         cv2.imshow('frame',frame)
#     except:
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
YOFFSET = 300
BAND = 20

import numpy as np
import cv2

WIDTH = 1280
HEIGHT = 720
cap = cv2.VideoCapture("../testimages/normal0.avi")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    filterSize = 30
    kernel = np.ones((filterSize,filterSize),np.float32)/(filterSize*filterSize)
    img_filter = cv2.filter2D(img,-1,kernel)


    crop_img = img[YOFFSET:(YOFFSET + BAND), 0:WIDTH]
    crop_img = cv2.reduce(crop_img, 0, cv2.REDUCE_AVG)


    # Find tow maximum intensity positions

    max1 = -1
    max1pos = -1
    max2= -1
    max2pos = -1

    THRESHOLD = 200
    hasFirstPeak = False
    inFirstPeak = False

    # TODO PEAK WIDTh

    for i in range(0, WIDTH):
        val = crop_img[0,i]
        if not hasFirstPeak and not inFirstPeak:
            if val > THRESHOLD:
                inFirstPeak = True
        if not hasFirstPeak and inFirstPeak:
            if val < THRESHOLD:
                inFirstPeak = False
                hasFirstPeak = True
        
        if not hasFirstPeak:
            if val > max1:
                max1 = val
                max1pos = i
        else:
            if val > max2:
                max2 = val
                max2pos = i



    # for i in range(0, WIDTH):
    #     val = crop_img[0,i]
    #     print val
    #     # Love dirty code
    #     if val > max1:
    #         if max1 > max2:
    #             max2 = max1
    #             max2pos = max1pos
            
    #         max1 = val
    #         max1pos = i
    #     elif val > max2:
    #         if max2 > max1:
    #             max1 = max2
    #             max1pos = max2pos
    #         max2 = val
    #         max2pos = i

    #  thereshold


    THRESHOLD = 220
    nbPeaks = 0
    if max1 > THRESHOLD:
        nbPeaks += 1
    if max2 > THRESHOLD:
        nbPeaks += 1

    img_filter = cv2.cvtColor(img_filter, cv2.COLOR_GRAY2RGB)
    cv2.circle(img_filter,(max1pos, YOFFSET), 5, (255,0,0), -1)
    cv2.circle(img_filter,(max2pos, YOFFSET), 5, (255,0,0), -1)
    cv2.imshow('filter image',img_filter)


    print "FOUND ", nbPeaks, " PEAKS"

    print max1, max1pos, max2, max2pos



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()