import cv2
import os
import numpy as np

fname = 'samples/im0.jpg'
x_offset = 250
border = 4

im = cv2.imread(fname)
cap = cv2.VideoCapture('samples/vid0.avi')

while(cap.isOpened()):
    _, im = cap.read()
    if im is None:
        break
    h, w, _ = im.shape

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_crop = np.ones([h+2*border-x_offset,w+2*border], dtype=np.uint8)
    im_crop[border:-border,border:-border] = im_gray[x_offset:]
    h, w = im_crop.shape

    edges =  cv2.Canny(im_crop, 150, 150, apertureSize=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    dilated = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)

    filled = dilated.copy()
    delta = 20
    filled[:delta,:delta] = 0
    filled[-delta:,:delta] = 0
    filled[:delta,-delta:] = 0
    filled[-delta:,-delta:] = 0

    mask = np.zeros((h+2,w+2),np.uint8)
    cv2.floodFill(filled, mask, (0,0), 255);
    filled = cv2.bitwise_not(filled)
    # filled = dilated | filled

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(opened.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    blobs = []
    for i,c in enumerate(contours):
        if hierarchy[0][i][-1]  == -1:
            m = cv2.moments(c)
            if m['m00'] > 1000:
                c_x = m['m10']/m['m00']
                c_y = m['m01']/m['m00']
                mu = {}
                # mu['m11'] = m['m11']/m['m00'] - c_y*c_x
                # mu['m20'] = m['m20']/m['m00'] - np.power(c_x,2)
                # mu['m02'] = m['m02']/m['m00'] - np.power(c_y,2)
                mu['m11'] = m['m11'] - c_x*m['m01']
                mu['m20'] = m['m20'] - c_x*m['m10']
                mu['m02'] = m['m02'] - c_y*m['m01']
                ecc = abs((np.power(mu['m20']-mu['m02'],2)-4*np.power(mu['m11'],2))/np.power(mu['m20']+mu['m02'],2))
                blobs.append({'pts':c,'ecc':ecc})

    blobs = sorted(blobs, key=lambda b: b['ecc'], reverse=True)
    # print [b['ecc'] for b in blobs]
    blobs = blobs[:2]

    for b in blobs:
        cv2.drawContours(im, [b['pts']], 0, (0,0,255), -1, offset=(-1,x_offset-1))


    cv2.imshow('im', im)
    cv2.imshow('crop', im_crop)
    cv2.imshow('edges', edges)
    cv2.imshow('dilated', dilated*255)
    cv2.imshow('filled', filled)
    cv2.imshow('opened', opened)
    c = cv2.waitKey(0)
    if c == ord('q'):
        break
