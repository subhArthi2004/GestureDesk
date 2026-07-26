# main.py
import cv2
import pyautogui
import time
from collections import deque
import config
from detector import GestureDetector

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05

class GestureDeskApp:
    def __init__(self):
        self.detector = GestureDetector()
        self.last_gesture_time = 0
        self.last_swipe_time = 0
        self.last_zoom_time = 0
        self.swipe_start_x = None
        self.swipe_start_time = None
        self.pinch_history = deque(maxlen=config.PINCH_HISTORY)
        self.last_pinch_state = None
        self.gesture_hold_name = None
        self.gesture_hold_count = 0
        self.action_log = deque(maxlen=3)
        self.frame_count = 0
        self.cached_result = None
        self.paused = False

    def update_hold(self, current_gesture):
        if current_gesture == self.gesture_hold_name:
            self.gesture_hold_count += 1
        else:
            self.gesture_hold_name = current_gesture
            self.gesture_hold_count = 1
        return self.gesture_hold_count == config.GESTURE_HOLD_FRAMES

    def execute_action(self, gesture, now, lms, w, h):
        hold_ready = self.update_hold(gesture)

        if gesture == "open_palm" and hold_ready and (now - self.last_gesture_time > config.GESTURE_COOLDOWN):
            pyautogui.press("f5")
            self.last_gesture_time = now
        elif gesture == "closed_fist" and hold_ready and (now - self.last_gesture_time > config.GESTURE_COOLDOWN):
            pyautogui.press("escape")
            self.last_gesture_time = now
        elif gesture == "thumbs_up" and hold_ready and (now - self.last_gesture_time > config.GESTURE_COOLDOWN):
            pyautogui.press("b")
            self.last_gesture_time = now
        elif gesture == "peace" and hold_ready and (now - self.last_gesture_time > config.GESTURE_COOLDOWN):
            pyautogui.hotkey("ctrl", "p")
            self.last_gesture_time = now
        elif gesture == "three_fingers" and hold_ready and (now - self.last_gesture_time > config.GESTURE_COOLDOWN):
            pyautogui.press("home")
            self.last_gesture_time = now
        elif gesture == "index_only":
            ix, _ = GestureDetector.landmark_px(lms[8], w, h)
            if self.swipe_start_x is None:
                self.swipe_start_x, self.swipe_start_time = ix, now
            else:
                elapsed = now - self.swipe_start_time
                delta = ix - self.swipe_start_x
                if elapsed < config.SWIPE_WINDOW:
                    if delta > config.SWIPE_THRESHOLD and (now - self.last_swipe_time > config.SWIPE_COOLDOWN):
                        pyautogui.press("right")
                        self.last_swipe_time = now
                        self.swipe_start_x = None
                    elif delta < -config.SWIPE_THRESHOLD and (now - self.last_swipe_time > config.SWIPE_COOLDOWN):
                        pyautogui.press("left")
                        self.last_swipe_time = now
                        self.swipe_start_x = None
                else:
                    self.swipe_start_x, self.swipe_start_time = ix, now

    def run(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            self.frame_count += 1

            if self.frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.cached_result = self.detector.hands.process(rgb)

            if not self.paused and self.cached_result and self.cached_result.multi_hand_landmarks:
                hand_lms = self.cached_result.multi_hand_landmarks[0]
                lms = hand_lms.landmark
                gesture = self.detector.classify_gesture(lms, w, h)
                self.execute_action(gesture, now, lms, w, h)

            cv2.imshow("GestureDesk HUD", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("p"):  # Pause/Unpause toggle
                self.paused = not self.paused

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = GestureDeskApp()
    app.run()