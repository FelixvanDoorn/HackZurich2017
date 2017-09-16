import cv2
import numpy as np

im = cv2.imread('samples/solidWhiteCurve.jpg')

def callback(x): return

def binary_thresh(channel, thresh = (0, 255), on = 1):
    binary = np.zeros_like(channel)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = on
    return binary

params = ['H','L','S','gradX','gradY','gradMag']
thresh = {p:{'min':0,'max':255} for p in params}
cv2.namedWindow('thresholds',0)
for p in params:
    for m in ['min','max']:
        name = p+'_'+m
        cv2.createTrackbar(name,'thresholds',0,255,callback)
        cv2.setTrackbarPos(name,'thresholds',thresh[p][m])

while True:
    for p in params:
        for m in ['min', 'max']:
            thresh[p][m] = cv2.getTrackbarPos(p+'_'+m,'thresholds')

    binaries = {}

    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    kernel = 3
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
    # calculate the scobel x gradient binary
    abs_sobelx = np.absolute(sobelx)
    binaries['gradX'] = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # calculate the scobel y gradient binary
    abs_sobely = np.absolute(sobely)
    binaries['gradY'] = np.uint8(255*abs_sobely/np.max(abs_sobely)).copy()
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    binaries['gradMag'] = gradmag/scale_factor

    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    # absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # dir_binary = binary_thresh(absgraddir, dir_thresh)

    hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    binaries['H'] = hls[:,:,0]
    binaries['L'] = hls[:,:,1]
    binaries['S'] = hls[:,:,2]

    # RGB colour
    # binaries['R'] = im[:,:,2]
    # binaries['G'] = im[:,:,1]
    # binaries['B'] = im[:,:,0]

    for p in binaries:
        cv2.imshow(p+'_raw', binaries[p])
        binaries[p] = 255*binary_thresh(binaries[p], (thresh[p]['min'], thresh[p]['max']))
        cv2.imshow(p+'_bin', binaries[p])

    cv2.imshow('im', im)

    c = cv2.waitKey(0)
    if c == ord('q'):
        break

print 'Thresholds: {t}'.format(t=thresh)
