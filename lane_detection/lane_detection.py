import numpy as np
import cv2
from scipy import signal

fname = 'samples/solidWhiteCurve.jpg'
# fname = 'samples/straight1.jpg'
# fname = 'samples/corners4.jpg'
im = cv2.imread(fname)
print im.shape


# def canny(img, low_threshold, high_threshold):
    # """Applies the Canny transform"""
    # return cv2.Canny(img, low_threshold, high_threshold)

# def auto_canny(image, sigma=0.33):
    # """Applies Canny transform and calculates the threshold"""
    # # compute the median of the single channel pixel intensities
    # v = np.median(image)

    # # apply automatic Canny edge detection using the computed median
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    # edged = cv2.Canny(image, lower, upper)
    # return edged

def binary_thresh(channel, thresh = (0, 255), on = 1):
    binary = np.zeros_like(channel)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = on
    return binary

def perspective_transforms(src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def perspective_warp(img, M):
    #img_size = (img.shape[1], img.shape[0])
    img_size = (img.shape[0], img.shape[1])

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped

def perspective_unwarp(img, Minv):
    #img_size = (img.shape[1], img.shape[0])
    img_size = (img.shape[0], img.shape[1])

    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)

    return unwarped

def calc_warp_points(img_height,img_width,x_center_adj=0):

    # calculator the vertices of the region of interest
    imshape = (img_height, img_width)
    xcenter=imshape[1]/2+x_center_adj
    xfd=34
    yf=320
    xoffset=90

    src = np.float32(
        [(xoffset,imshape[0]),
         (xcenter-xfd, yf),
         (xcenter+xfd,yf),
         (imshape[1]-xoffset,imshape[0])])
    dst = np.float32(
        [(xoffset,imshape[1]),
         (xoffset,0),
         (imshape[0]-xoffset, 0),
        (imshape[0]-xoffset,imshape[1])])
    return src, dst

# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# k_size = 5
# blur_gray = cv2.GaussianBlur(gray, (k_size, k_size), 0)
# edges = canny(blur_gray, 50, 150)
# cv2.imshow('edges',edges)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
kernel = 3
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
abs_sobelx = np.absolute(sobelx)
gradx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
abs_sobely = np.absolute(sobely)
grady = np.uint8(255*abs_sobely/np.max(abs_sobely)).copy()
gradx_bin = binary_thresh(gradx, (85,255))

hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
L = hls[:,:,1]
S = hls[:,:,2]
L_bin = binary_thresh(L, (170,255))
S_bin = binary_thresh(S, (140,255))

lanes_bin = gradx_bin | L_bin | S_bin

src, dst = calc_warp_points(im.shape[0],im.shape[1])
M, _ = perspective_transforms(src, dst)
im_warped = perspective_warp(im, M)
bin_warped = perspective_warp(lanes_bin, M)

h, w = bin_warped.shape
for row in range(h):
    slice = bin_warped[h-row-1]
    mask = np.concatenate(([0], np.equal(slice, 1).view(np.int8), [0]))
    absdiff = np.abs(np.diff(mask))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    # hist = np.sum(bin_warped[h-row-10:h-row], axis=0)
    # peaks = signal.find_peaks_cwt(hist, np.arange(1,150),min_length=150)
    if len(ranges) == 2:
        start_left, start_right = [(ranges[x][0]+ranges[x][1])/2 for x in (0,1)]
        start_row = h-row-1
        break

nb_points = 10
w_win = w/5
h_win = h/10
# rows, h_win = np.linspace(0, start_row, num=nb_points, endpoint=False, retstep=True)
rows = np.arange(h_win/2,start_row+h_win/2, step=h_win)
rows = rows[::-1]
pts = {'left':[start_left], 'right':[start_right]}
print 'Window [h,w]: [{h},{w}]'.format(h=h_win,w=w_win)
for l in pts:
    for r in rows:
        y = pts[l][-1]
        while True:
            print 'y: {y}'.format(y=y)
            mask = bin_warped[r-h_win/2:r+h_win/2,y-w_win/2:y+w_win/2]
            if not mask.sum(): # no lane bin
                y_center = y
            else:
                x_center, y_center = np.argwhere(mask==1).sum(0)/(mask == 1).sum()
                y_center += y - w_win/2
            print 'New center: {y}'.format(y=y_center)
            # raw_input('')
            if abs(y_center - y)<2:
                pts[l].append(y)
                print 'New y position: {y}'.format(y=y)
                break
            else:
                y = y_center
rows = np.insert(rows,0,start_row)

poly = {l:np.polyfit(rows, pts[l], 2) for l in pts}

disp_warped = np.zeros((h,w,3), dtype=np.uint8)
disp_warped[np.where(bin_warped)] = (255,255,255)
mask = 255*np.zeros_like(disp_warped)
for l in pts:
    for p in zip(pts[l],rows):
        cv2.circle(disp_warped, p, 3, (0,0,255), -1)
    x = np.arange(h)
    val = np.polyval(poly[l], x).astype(np.int)
    for px,py in zip(x,val):
        mask[px,py-10:py+10] = (0,255,0)
disp_warped = cv2.addWeighted(disp_warped, 0.7, mask, 0.3, 0)


cv2.imshow('im',im)
cv2.imshow('bin', lanes_bin*255)
cv2.imshow('im_warped', im_warped)
cv2.imshow('im_warped', bin_warped*255)
cv2.imshow('disp_warped',disp_warped)
cv2.waitKey(0)
