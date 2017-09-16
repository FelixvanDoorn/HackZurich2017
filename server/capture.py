from multiprocessing import Process, Event, Lock, Queue
from Queue import Full
from time import sleep
import select
import cv2
import os
import logging
import numpy as np

class Capture(Process):

    def __init__(self, out_queue):
        Process.__init__(self)
        self._out_queue = out_queue
        self._stop = Event()
        self._stop.set()
        self._stream = None
        self._device_name = None

    def setDevice(self, device):
        self._device_name = device
    
    def openStream(self):
        logging.debug("Opening stream.")
	self._stream = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

    def run(self):
        self._stop.clear()
	if self._stream is None:
	    self.openStream()
        
        while True :
            if self._stop.is_set():
                break
                
	    ret, im = self._stream.read()

	    if im is None:
	        logging.warning("Grabbed frame is empty.")
		continue
	    im = cv2.resize(im, None, fx=0.25, fy=0.25)

	    while True:
	        try:
		    self._out_queue.put(cv2.imencode('.jpg', im, [int(cv2.IMWRITE_JPEG_QUALITY),70])[1].tobytes())
	        except Full:
		    self._out_queue.get()
	        else:
		    break

        if self._stream is not None:
            self._stream.release()
        logging.info("Thread stopped.")
    

    def stop(self):
        self._stop.set()


