import cv2
import numpy as np


img = cv2.imread("./testimages/im2345.jpg")
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
crop_img = img_grey[399:, :]
print(np.shape(crop_img))
cv2.imshow("cropped", crop_img)
border_size = 4

dims = np.shape(crop_img)
img = np.ones([(dims[0]+2*border_size)*10, (dims[1]+2*border_size)*10], dtype=np.uint8)

img[:dims[0], :dims[1]] = crop_img


imedge = cv2.Canny(img, 250, 700, 3)

size = border_size * 8

imedge[:size, :size] = 0
imedge[dims[0]-size:dims[1], :size] = 0
imedge[1:size, dims[0]-size:dims[1]] = 0
imedge[dims[0]-size:dims[0], dims[1]-size:dims[1]] = 0

cv2.imshow("Epic shit!", imedge)


cv2.waitKey(100000)