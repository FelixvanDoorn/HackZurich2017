## setup logging
import logging
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import subplot, imshow, draw
#import matplotlib.imshow
import pdb
from blob_detection import findObstacles

logging.basicConfig(level = logging.INFO)

## import the package
import pyrealsense as pyrs

## start the service - also available as context manager
serv = pyrs.Service()

## create a device from device id and streams of interest
cam = serv.Device(device_id = 0, streams = [pyrs.stream.DepthStream(fps = 60), pyrs.stream.ColorStream(fps = 60)])
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#vidObjColor = cv2.VideoWriter("Color7.avi", fourcc, 25, (640,480))
#vidObjDepth = cv2.VideoWriter("Depth7.avi", fourcc, 25, (640,480))

## retrieve 60 frames of data
for _ in range(4000):
    cam.wait_for_frames()
    d = cam.depth * cam.depth_scale * 1000
    c = cam.color
    d = d.astype(np.float) / 20000 * 256
    d = d.astype(np.uint8) 
    dd = np.tile(d[:,:,np.newaxis], (1,1,3));
    final = findObstacles(d)
    final3 = np.tile(final[:,:,np.newaxis], (1,1,3));
    new_image = c.copy();
    new_image[:,:,0] = 0.5*new_image[:,:,0]+40*final
    subplot(221), plt.imshow(new_image, aspect='auto', interpolation='nearest')
    subplot(222), plt.imshow(dd*500, aspect='auto', interpolation='nearest')
    subplot(223), plt.imshow(final3*4, aspect='auto', interpolation='nearest')
   	#d3 = plt.imshow(final3*4, aspect='auto', interpolation='nearest')
    subplot(224), plt.imshow(c, aspect='auto', interpolation='nearest')
    #imgplot = plt.imshow(np.concatenate((new_image,dd*500, final3*4, c),axis=1))
    plt.pause(0.04)
    if 0xFF == ord('q'):
        break

## stop camera and service
cam.stop()
serv.stop()
#vidObjColor.release()
#vidObjDepth.release()
