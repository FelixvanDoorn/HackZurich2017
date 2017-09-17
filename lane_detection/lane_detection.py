import numpy as np
import cv2
from scipy import signal
from sklearn.cluster import MeanShift, estimate_bandwidth

fname = 'samples/im0.jpg'
# fname = 'samples/solidWhiteCurve.jpg'
# fname = 'samples/straight1.jpg'
# fname = 'samples/corners4.jpg'
im = cv2.imread(fname)
cap = cv2.VideoCapture('samples/vid0.avi')
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
    # xfd=34
    # yf=320
    # xoffset=90
    xoffset = 0
    xfd = imshape[1]/2-xoffset
    yf = 350

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

def extract_lanes(im):

    disp_im = im.copy()

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
    L_bin = binary_thresh(L, (150,255))
    S_bin = binary_thresh(S, (140,255))

    lanes_bin = gradx_bin | L_bin | S_bin

    # src, dst = calc_warp_points(im.shape[0],im.shape[1])
    # M, _ = perspective_transforms(src, dst)
    # im_warped = perspective_warp(im, M)
    # bin_warped = perspective_warp(lanes_bin, M)
    roi_offset = 450
    bin_warped = lanes_bin[-roi_offset:,:].copy()

    # cv2.line(im, tuple(src[0]), tuple(src[1]), (0,0,255), thickness=3)
    # cv2.line(im, tuple(src[1]), tuple(src[2]), (0,0,255), thickness=3)
    # cv2.line(im, tuple(src[2]), tuple(src[3]), (0,0,255), thickness=3)
    # cv2.line(im, tuple(src[3]), tuple(src[0]), (0,0,255), thickness=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    bin_warped = cv2.morphologyEx(bin_warped, cv2.MORPH_OPEN, kernel)
    bin_warped = cv2.morphologyEx(bin_warped, cv2.MORPH_CLOSE, kernel)

    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray[-roi_offset:,:],150,300,apertureSize = 3)
    # edges = cv2.Canny(bin_warped*255,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,100)
    print lines

    if lines is None:
        cluster_centers = None
    elif len(lines[0]) == 1:
        cluster_centers = lines[0][0]
    else:
        # try:
            # bandwidth = estimate_bandwidth(lines[0], quantile=0.4, n_samples=500)
        # except:
            # print lines
        # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        # ms.fit(lines[0])
        # labels = ms.labels_
        # cluster_centers = ms.cluster_centers_

        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(disp_im,(x1,y1+roi_offset),(x2,y2+roi_offset),(0,0,255),2)
        # for rho,theta in cluster_centers:
            # a = np.cos(theta)
            # b = np.sin(theta)
            # x0 = a*rho
            # y0 = b*rho
            # x1 = int(x0 + 1000*(-b))
            # y1 = int(y0 + 1000*(a))
            # x2 = int(x0 - 1000*(-b))
            # y2 = int(y0 - 1000*(a))
            # cv2.line(disp_im,(x1,y1+roi_offset),(x2,y2+roi_offset),(0,255,0),2)
    cluster_centers = None

    return cluster_centers, disp_im, bin_warped, edges

    # h, w = bin_warped.shape
    # for row in range(h):
        # slice = bin_warped[h-row-1]
        # mask = np.concatenate(([0], np.equal(slice, 1).view(np.int8), [0]))
        # absdiff = np.abs(np.diff(mask))
        # ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        # # hist = np.sum(bin_warped[h-row-10:h-row], axis=0)
        # # peaks = signal.find_peaks_cwt(hist, np.arange(1,150),min_length=150)
        # if len(ranges) == 2:
            # start_left, start_right = [(ranges[x][0]+ranges[x][1])/2 for x in (0,1)]
            # start_row = h-row-1
            # break

    # nb_points = 10
    # w_win = w/5
    # h_win = h/10
    # rows = np.arange(h_win/2,start_row+h_win/2, step=h_win)
    # rows = rows[::-1]
    # pts = {'left':[start_left], 'right':[start_right]}
    # print 'Window [h,w]: [{h},{w}]'.format(h=h_win,w=w_win)
    # for l in pts:
        # for r in rows:
            # y = pts[l][-1]
            # while True:
                # print 'y: {y}'.format(y=y)
                # mask = bin_warped[r-h_win/2:r+h_win/2,y-w_win/2:y+w_win/2]
                # if not mask.sum(): # no lane bin
                    # y_center = y
                # else:
                    # x_center, y_center = np.argwhere(mask==1).sum(0)/(mask == 1).sum()
                    # y_center += y - w_win/2
                # print 'New center: {y}'.format(y=y_center)
                # if abs(y_center - y)<2:
                    # pts[l].append(y)
                    # print 'New y position: {y}'.format(y=y)
                    # break
                # else:
                    # y = y_center
    # rows = np.insert(rows,0,start_row)

    # poly = {l:np.polyfit(rows, pts[l], 2) for l in pts}

    # disp_warped = np.zeros((h,w,3), dtype=np.uint8)
    # disp_warped[np.where(bin_warped)] = (255,255,255)
    # mask = 255*np.zeros_like(disp_warped)
    # for l in pts:
        # for p in zip(pts[l],rows):
            # cv2.circle(disp_warped, p, 3, (0,0,255), -1)
        # x = np.arange(h)
        # val = np.polyval(poly[l], x).astype(np.int)
        # for px,py in zip(x,val):
            # mask[px,py-10:py+10] = (0,255,0)
    # disp_warped = cv2.addWeighted(disp_warped, 0.7, mask, 0.3, 0)
    # cv2.imshow('disp_warped',disp_warped)


while(cap.isOpened()):
    ret, im = cap.read()
    if im is None:
        break
    lanes, disp_im, bin_warped, edges = extract_lanes(im)
    # print lanes

    cv2.imshow('im',disp_im)
    cv2.imshow('e', edges)
    cv2.imshow('bin_warped', bin_warped*255)
    c = cv2.waitKey(1)
    if c == ord('q'):
        break
