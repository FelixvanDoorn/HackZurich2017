import numpy as np
import cv2


STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
k_min_number_of_features = 1500


lk_params = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def feature_tracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)
    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]
    return kp1, kp2


class PinholeCamera(object):
    def __init__(self, width, height, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry(object):
    def __init__(self, cam, annotations):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp(cam.cx, cam.cy)
        self.true_x = 0
        self.true_y = 0
        self.true_z = 0
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        with open(annotations) as f:
            self.annotations = f.readlines()

    def get_absolute_scale(self, frame_id):
        ss = self.annotations[frame_id-1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.true_x, self.true_y, self.true_z = x, y, z
        return np.sqrt((x-x_prev)*(x-x_prev)+(y-y_prev)*(y-y_prev)+(z-z_prev)*(z-z_prev))

    def process_first_frame(self):
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME

    def process_second_frame(self):
        self.px_ref, self.px_cur = feature_tracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp)
        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur

    def process_frame(self, frame_id):
        self.px_ref, self.px_cur = feature_tracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp)
        absolute_scale = self.get_absolute_scale(frame_id)
        if absolute_scale > 0.1:
            self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)
        if self.px_ref.shape[0] < k_min_number_of_features:
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
        self.px_ref = self.px_cur

        def update(self, img, frame_id):
            assert (img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[
                1] == self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
            self.new_frame = img
            if self.frame_stage == STAGE_DEFAULT_FRAME:
                self.processFrame(frame_id)
            elif self.frame_stage == STAGE_SECOND_FRAME:
                self.processSecondFrame()
            elif self.frame_stage == STAGE_FIRST_FRAME:
                self.processFirstFrame()
            self.last_frame = self.new_frame



