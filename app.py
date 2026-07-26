import os
import sys
import cv2
import mediapipe as mp
import pyautogui
import time
import math
from collections import deque

# ─── Fix for MediaPipe PyInstaller bundling ──────────────────────────────────
def get_mediapipe_path():
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, 'mediapipe')
    return os.path.dirname(mp.__file__)

# ─── Configuration ────────────────────────────────────────────────────────────
CAMERA_INDEX           = 0       
DETECTION_CONF         = 0.75    
TRACKING_CONF          = 0.75    

GESTURE_COOLDOWN       = 1.5     
SWIPE_COOLDOWN         = 0.8     
ZOOM_COOLDOWN          = 0.5     

GESTURE_HOLD_FRAMES    = 8       
SWIPE_THRESHOLD        = 90      
SWIPE_WINDOW           = 0.5     

PINCH_OPEN_DIST        = 80      
PINCH_CLOSE_DIST       = 30      
PINCH_HISTORY          = 5       

PROCESS_EVERY_N_FRAMES = 2

# ─── PyAutoGUI safety ─────────────────────────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.05

# ─── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=DETECTION_CONF,
    min_tracking_confidence=TRACKING_CONF,
)

# ─── State ────────────────────────────────────────────────────────────────────
last_gesture_time    = 0
last_swipe_time      = 0
last_zoom_time       = 0

swipe_start_x        = None
swipe_start_time     = None

pinch_history        = deque(maxlen=PINCH_HISTORY)
last_pinch_state     = None   

action_display       = ""
action_display_until = 0

gesture_hold_name    = None
gesture_hold_count   = 0

action_log           = deque(maxlen=3)
frame_count          = 0
cached_result        = None
fps_deque            = deque(maxlen=30)
blank_screen_on      = False


# ─── Helpers ──────────────────────────────────────────────────────────────────

def landmark_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def fingers_up(lms, w, h):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    state = []

    tx, _      = landmark_px(lms[4], w, h)   
    ip_x, _    = landmark_px(lms[3], w, h)   
    wrist_x, _ = landmark_px(lms[0], w, h)
    mid_mcp_x, _ = landmark_px(lms[9], w, h) 

    if wrist_x < mid_mcp_x:   
        state.append(1 if tx > ip_x else 0)
    else:                       
        state.append(1 if tx < ip_x else 0)

    for tip, pip in zip(tips[1:], pips[1:]):
        ty = landmark_px(lms[tip], w, h)[1]
        py = landmark_px(lms[pip], w, h)[1]
        state.append(1 if ty < py else 0)

    return state


def is_closed_fist(lms, w, h):
    f = fingers_up(lms, w, h)
    return f[1] == 0 and f[2] == 0 and f[3] == 0 and f[4] == 0


def pinch_distance(lms, w, h):
    tx, ty = landmark_px(lms[4], w, h)
    ix, iy = landmark_px(lms[8], w, h)
    return math.hypot(tx - ix, ty - iy)


def show_action(msg):
    global action_display, action_display_until
    action_display = msg
    action_display_until = time.time() + 1.5
    action_log.appendleft(f"{time.strftime('%H:%M:%S')}  {msg}")
    print(f"Action: {msg}")


def classify_gesture(lms, w, h):
    f = fingers_up(lms, w, h)
    thumb, index, middle, ring, pinky = f

    if is_closed_fist(lms, w, h):
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


def update_hold(current_gesture):
    global gesture_hold_name, gesture_hold_count

    if current_gesture == gesture_hold_name:
        gesture_hold_count += 1
    else:
        gesture_hold_name  = current_gesture
        gesture_hold_count = 1

    return gesture_hold_count == GESTURE_HOLD_FRAMES


# ─── HUD drawing helpers ───────────────────────────────────────────────────────

def draw_panel(img, pt1, pt2, color=(20, 20, 20), alpha=0.60, radius=8):
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    r = radius
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


GESTURE_COLORS = {
    "open_palm":     (0,   220, 100),
    "closed_fist":   (50,  50,  220),
    "thumbs_up":     (0,   200, 255),
    "peace":         (255, 180,   0),
    "three_fingers": (200,   0, 200),
    "index_only":    (255, 200,   0),
    "pinch":         (200, 100, 255),
    "none":          (80,   80,  80),
    "unknown":       (80,   80,  80),
}

HOLD_GESTURES = {"open_palm", "closed_fist", "thumbs_up", "peace", "three_fingers"}


# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    global last_gesture_time, last_swipe_time, last_zoom_time
    global swipe_start_x, swipe_start_time, last_pinch_state
    global frame_count, cached_result, blank_screen_on
    global GESTURE_HOLD_FRAMES, SWIPE_THRESHOLD

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_frame_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame       = cv2.flip(frame, 1)
        h, w        = frame.shape[:2]
        now         = time.time()
        frame_count += 1

        fps_deque.append(now - prev_frame_time)
        prev_frame_time = now
        fps = 1.0 / (sum(fps_deque) / len(fps_deque)) if fps_deque else 0.0

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cached_result = hands.process(rgb)

        result  = cached_result
        gesture = "none"

        if result and result.multi_hand_landmarks:
            hand_lms = result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            lms        = hand_lms.landmark
            gesture    = classify_gesture(lms, w, h)
            hold_ready = update_hold(gesture)

            if gesture == "open_palm":
                if hold_ready and now - last_gesture_time > GESTURE_COOLDOWN:
                    pyautogui.press("f5")
                    show_action("START PRESENTATION (F5)")
                    last_gesture_time = now
                swipe_start_x = swipe_start_time = None

            elif gesture == "closed_fist":
                if hold_ready and now - last_gesture_time > GESTURE_COOLDOWN:
                    pyautogui.press("escape")
                    show_action("END PRESENTATION (ESC)")
                    last_gesture_time = now
                swipe_start_x = swipe_start_time = None

            elif gesture == "thumbs_up":
                if hold_ready and now - last_gesture_time > GESTURE_COOLDOWN:
                    pyautogui.press("b")
                    blank_screen_on = not blank_screen_on
                    show_action("BLANK SCREEN (B)" if blank_screen_on else "UNBLANK SCREEN (B)")
                    last_gesture_time = now
                swipe_start_x = swipe_start_time = None

            elif gesture == "peace":
                if hold_ready and now - last_gesture_time > GESTURE_COOLDOWN:
                    pyautogui.hotkey("ctrl", "p")
                    show_action("TOGGLE PEN (Ctrl+P)")
                    last_gesture_time = now
                swipe_start_x = swipe_start_time = None

            elif gesture == "three_fingers":
                if hold_ready and now - last_gesture_time > GESTURE_COOLDOWN:
                    pyautogui.press("home")
                    show_action("FIRST SLIDE (Home)")
                    last_gesture_time = now
                swipe_start_x = swipe_start_time = None

            elif gesture == "index_only":
                ix, _ = landmark_px(lms[8], w, h)

                if swipe_start_x is None:
                    swipe_start_x    = ix
                    swipe_start_time = now
                else:
                    elapsed = now - swipe_start_time
                    delta   = ix - swipe_start_x

                    if elapsed < SWIPE_WINDOW:
                        if delta > SWIPE_THRESHOLD:
                            if now - last_swipe_time > SWIPE_COOLDOWN:
                                pyautogui.press("right")
                                show_action("NEXT SLIDE ->")
                                last_swipe_time = now
                            swipe_start_x = swipe_start_time = None

                        elif delta < -SWIPE_THRESHOLD:
                            if now - last_swipe_time > SWIPE_COOLDOWN:
                                pyautogui.press("left")
                                show_action("<- PREV SLIDE")
                                last_swipe_time = now
                            swipe_start_x = swipe_start_time = None
                    else:
                        swipe_start_x    = ix
                        swipe_start_time = now

            elif gesture == "pinch":
                dist = pinch_distance(lms, w, h)
                pinch_history.append(dist)
                avg_dist = sum(pinch_history) / len(pinch_history)

                if avg_dist > PINCH_OPEN_DIST:
                    state = "open"
                elif avg_dist < PINCH_CLOSE_DIST:
                    state = "closed"
                else:
                    state = "mid"

                if state != last_pinch_state and now - last_zoom_time > ZOOM_COOLDOWN:
                    if state == "open" and last_pinch_state == "closed":
                        pyautogui.hotkey("ctrl", "=")
                        show_action("ZOOM IN (+)")
                        last_zoom_time = now
                    elif state == "closed" and last_pinch_state == "open":
                        pyautogui.hotkey("ctrl", "-")
                        show_action("ZOOM OUT (-)")
                        last_zoom_time = now

                last_pinch_state = state
                swipe_start_x    = swipe_start_time = None

            else:
                swipe_start_x = swipe_start_time = None

        else:
            swipe_start_x = swipe_start_time = None

        # ── HUD Overlay ───────────────────────────────────────────────────────

        if now < action_display_until:
            banner_w = min(len(action_display) * 17 + 28, w)
            draw_panel(frame, (0, 4), (banner_w, 52), alpha=0.70)
            cv2.putText(frame, action_display, (12, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 120), 2)

        draw_panel(frame, (w - 95, 6), (w - 6, 28), alpha=0.50)
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 90, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

        hints = [
            "Q = Quit",
            "Palm  = F5",
            "Fist  = ESC",
            "Thumb = Blank",
            "Peace = Pen",
            "3-Fin = Home",
            "Swipe = Slide",
            "Pinch = Zoom",
        ]
        draw_panel(frame, (w - 155, 32), (w - 6, 32 + len(hints) * 20 + 6), alpha=0.45)
        for i, hint in enumerate(hints):
            cv2.putText(frame, hint, (w - 150, 50 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)

        sens_txt = (f"Hold: {GESTURE_HOLD_FRAMES} frames   "
                    f"Swipe: {SWIPE_THRESHOLD} px     "
                    f"[ ] = hold   , . = swipe")
        cv2.putText(frame, sens_txt, (12, h - 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1)

        if gesture in HOLD_GESTURES and gesture == gesture_hold_name:
            progress  = min(gesture_hold_count / GESTURE_HOLD_FRAMES, 1.0)
            bar_color = GESTURE_COLORS.get(gesture, (160, 160, 160))
            bar_w     = int(progress * 200)
            draw_panel(frame, (10, h - 108), (215, h - 88), alpha=0.60)
            if bar_w > 0:
                cv2.rectangle(frame, (12, h - 106), (12 + bar_w, h - 90), bar_color, -1)
            cv2.putText(frame, "Hold...", (218, h - 91),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        col = GESTURE_COLORS.get(gesture, (80, 80, 80))
        draw_panel(frame, (8, h - 40), (270, h - 8), alpha=0.55)
        cv2.putText(frame, f"Gesture: {gesture}", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, col, 2)

        if (swipe_start_x is not None
                and gesture == "index_only"
                and result and result.multi_hand_landmarks):
            ix, _ = landmark_px(result.multi_hand_landmarks[0].landmark[8], w, h)
            delta  = ix - swipe_start_x
            bar_x  = max(0, min(w, w // 2 + int(delta * 1.5)))
            cv2.line(frame, (w // 2, h - 60), (bar_x, h - 60), (255, 200, 0), 4)
            cv2.circle(frame, (w // 2, h - 60), 6, (255, 255, 255), -1)
            for sign in (1, -1):
                mx = w // 2 + sign * SWIPE_THRESHOLD
                cv2.line(frame, (mx, h - 70), (mx, h - 50), (100, 200, 100), 1)

        if gesture == "pinch" and result and result.multi_hand_landmarks:
            dist    = pinch_distance(result.multi_hand_landmarks[0].landmark, w, h)
            bar_len = int(min(dist, 200))
            draw_panel(frame, (8, h - 88), (225, h - 62), alpha=0.55)
            cv2.rectangle(frame, (12, h - 84), (12 + bar_len, h - 66), (200, 100, 255), -1)
            cv2.putText(frame, f"Pinch: {int(dist)} px", (12, h - 91),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 100, 255), 1)

        if action_log:
            log_panel_h = len(action_log) * 18 + 10
            draw_panel(frame, (w - 305, h - log_panel_h - 8), (w - 6, h - 6), alpha=0.40)
            for i, entry in enumerate(action_log):
                alpha_val = 180 - i * 50
                cv2.putText(frame, entry, (w - 300, h - 14 - i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                            (alpha_val, alpha_val, alpha_val), 1)

        cv2.imshow("Hand Gesture PPT Controller", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        elif key == ord("["):
            GESTURE_HOLD_FRAMES = max(2, GESTURE_HOLD_FRAMES - 1)
        elif key == ord("]"):
            GESTURE_HOLD_FRAMES = min(30, GESTURE_HOLD_FRAMES + 1)
        elif key == ord(","):
            SWIPE_THRESHOLD = max(30, SWIPE_THRESHOLD - 10)
        elif key == ord("."):
            SWIPE_THRESHOLD = min(200, SWIPE_THRESHOLD + 10)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()