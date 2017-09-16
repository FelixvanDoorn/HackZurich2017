import cv2
import numpy as np


img = cv2.imread("./testimages/im5.jpg")
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
crop_img = img[400:, :]
cv2.imshow(crop_img)

