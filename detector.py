# detector.py
import cv2
import mediapipe as mp
import math
import config

class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=config.DETECTION_CONF,
            min_tracking_confidence=config.TRACKING_CONF,
        )

    @staticmethod
    def landmark_px(lm, w, h):
        return int(lm.x * w), int(lm.y * h)

    def fingers_up(self, lms, w, h):
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        state = []

        tx, _ = self.landmark_px(lms[4], w, h)
        ip_x, _ = self.landmark_px(lms[3], w, h)
        wrist_x, _ = self.landmark_px(lms[0], w, h)
        mid_mcp_x, _ = self.landmark_px(lms[9], w, h)

        if wrist_x < mid_mcp_x:
            state.append(1 if tx > ip_x else 0)
        else:
            state.append(1 if tx < ip_x else 0)

        for tip, pip in zip(tips[1:], pips[1:]):
            ty = self.landmark_px(lms[tip], w, h)[1]
            py = self.landmark_px(lms[pip], w, h)[1]
            state.append(1 if ty < py else 0)

        return state

    def is_closed_fist(self, lms, w, h):
        f = self.fingers_up(lms, w, h)
        return f[1] == 0 and f[2] == 0 and f[3] == 0 and f[4] == 0

    def pinch_distance(self, lms, w, h):
        tx, ty = self.landmark_px(lms[4], w, h)
        ix, iy = self.landmark_px(lms[8], w, h)
        return math.hypot(tx - ix, ty - iy)

    def classify_gesture(self, lms, w, h):
        f = self.fingers_up(lms, w, h)
        thumb, index, middle, ring, pinky = f

        if self.is_closed_fist(lms, w, h):
            return "closed_fist"
        if all(x == 1 for x in f):
            return "open_palm"
        if thumb == 1 and index == 0 and middle == 0 and ring == 0 and pinky == 0:
            return "thumbs_up"
        if index == 1 and middle == 1 and ring == 0 and pinky == 0 and thumb == 0:
            return "peace"
        if index == 1 and middle == 1 and ring == 1 and pinky == 0 and thumb == 0:
            return "three_fingers"
        if index == 1 and middle == 0 and ring == 0 and pinky == 0:
            return "index_only"
        if middle == 0 and ring == 0 and pinky == 0:
            return "pinch"

        return "unknown"