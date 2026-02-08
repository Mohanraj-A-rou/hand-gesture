import cv2
import mediapipe as mp
import time
import math

# ==========================================
#        1. INITIALIZATION & SETUP
# ==========================================

# --- MediaPipe Solutions ---
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Detectors
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5, max_num_faces=1)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- Configuration & Thresholds ---
BODY_MISSING_THRESHOLD = 5.0     # Seconds body must be gone to trigger Fall Alert
FACE_ASYMMETRY_THRESHOLD = 0.03  # Sensitivity for Stroke detection

# *** UPDATED: Time to hold ANY gesture before it activates ***
GESTURE_HOLD_TIME = 3.0          

# --- State Variables ---
# Fall State
body_missing_start = None
fall_alert_active = False

# Gesture State
current_gesture = None
gesture_start_time = 0
confirmed_text = "Monitoring Active"

# --- Camera Setup ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640) # Width
cap.set(4, 480) # Height

# ==========================================
#        2. HELPER FUNCTIONS
# ==========================================

def get_fingers_status(hand, label):
    """Returns list of 1s (Open) and 0s (Closed) for [Thumb, Index, Middle, Ring, Pinky]"""
    lm = hand.landmark
    fingers = []
    TIP_IDS = [4, 8, 12, 16, 20]

    # Thumb Logic (Mirror Safe)
    if label == "Left": 
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else: 
        fingers.append(1 if lm[4].x > lm[3].x else 0)

    # 4 Fingers Logic
    for i in range(1, 5):
        fingers.append(1 if lm[TIP_IDS[i]].y < lm[TIP_IDS[i]-2].y else 0)
            
    return fingers

def identify_gesture(f):
    """Maps finger states to meaning."""
    # [Thumb, Index, Middle, Ring, Pinky]
    if f == [1, 0, 0, 0, 0]: return "THUMB_EMERGENCY" 
    if f == [1, 1, 0, 0, 0]: return "Need Water/Food"
    if f == [0, 1, 0, 0, 0]: return "Want Restroom"
    if f == [0, 0, 1, 1, 1]: return "Want Medicine"
    if f == [1, 0, 1, 1, 1]: return "Adjust Position"
    if f == [0, 1, 1, 0, 0]: return "Call Caregiver"
    if f == [1, 1, 1, 0, 0]: return "Uncomfortable"
    return None

def get_mouth_asymmetry(face_landmarks):
    left_mouth = face_landmarks[61]
    right_mouth = face_landmarks[291]
    nose = face_landmarks[1]

    left_rel = abs(left_mouth.y - nose.y)
    right_rel = abs(right_mouth.y - nose.y)
    return abs(left_rel - right_rel)

# ==========================================
#        3. MAIN LOOP
# ==========================================
print("System Starting... Press 'q' to exit.")

while True:
    success, frame = cap.read()
    if not success: continue

    # Flip & Convert
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    status_color = (0, 255, 0) # Green default
    top_alert_text = ""
    countdown_text = ""
    
    # ----------------------------------------------------
    # MODULE A: FALL DETECTION
    # ----------------------------------------------------
    pose_results = pose.process(rgb)
    
    if not pose_results.pose_landmarks:
        if body_missing_start is None:
            body_missing_start = time.time()
        
        elapsed = time.time() - body_missing_start
        if elapsed > BODY_MISSING_THRESHOLD:
            top_alert_text = "FALL DETECTED / PATIENT MISSING"
            status_color = (0, 0, 255) # Red
            fall_alert_active = True
        else:
            cv2.putText(frame, f"Searching Body: {int(BODY_MISSING_THRESHOLD - elapsed)}s", 
                       (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        body_missing_start = None
        fall_alert_active = False

    # ----------------------------------------------------
    # MODULE B: STROKE DETECTION
    # ----------------------------------------------------
    face_results = face_mesh.process(rgb)
    
    if face_results.multi_face_landmarks:
        face_lm = face_results.multi_face_landmarks[0].landmark
        asymmetry = get_mouth_asymmetry(face_lm)
        
        if asymmetry > FACE_ASYMMETRY_THRESHOLD:
            if top_alert_text == "": 
                top_alert_text = "POSSIBLE STROKE DETECTED"
                status_color = (0, 0, 255)
            
            mp_draw.draw_landmarks(frame, face_results.multi_face_landmarks[0], 
                                 mp_face.FACEMESH_TESSELATION,
                                 landmark_drawing_spec=None,
                                 connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

    # ----------------------------------------------------
    # MODULE C: GESTURE RECOGNITION (Unified 3s Timer)
    # ----------------------------------------------------
    hand_results = hands.process(rgb)
    
    if hand_results.multi_hand_landmarks:
        for idx, hand_lm in enumerate(hand_results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            
            label = hand_results.multi_handedness[idx].classification[0].label
            fingers = get_fingers_status(hand_lm, label)
            gesture = identify_gesture(fingers)

            if gesture:
                # If we are holding the SAME gesture
                if gesture == current_gesture:
                    hold_duration = time.time() - gesture_start_time
                    
                    # Calculate remaining seconds for UI (3...2...1)
                    seconds_left = math.ceil(GESTURE_HOLD_TIME - hold_duration)
                    
                    if hold_duration > GESTURE_HOLD_TIME:
                        # --- 3 SECONDS PASSED: CONFIRM GESTURE ---
                        if gesture == "THUMB_EMERGENCY":
                            confirmed_text = "EMERGENCY: THUMB TRIGGERED"
                            status_color = (0, 0, 255) # Red for emergency
                        else:
                            confirmed_text = gesture
                            # If emergency was previously set, revert color to green for normal requests
                            if "EMERGENCY" not in confirmed_text:
                                status_color = (0, 255, 0)
                    else:
                        # --- COUNTING DOWN ---
                        countdown_text = f"Holding: {seconds_left}s"
                        
                        # Draw countdown near the hand (using wrist coordinate)
                        h, w, c = frame.shape
                        cx, cy = int(hand_lm.landmark[0].x * w), int(hand_lm.landmark[0].y * h)
                        cv2.putText(frame, countdown_text, (cx + 20, cy), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    # New gesture detected, reset timer
                    current_gesture = gesture
                    gesture_start_time = time.time()
            else:
                # Hand is visible but gesture is unknown
                current_gesture = None
    else:
        # No hands visible
        current_gesture = None

    # ==========================================
    #        4. FINAL DISPLAY COMPOSITION
    # ==========================================
    
    # Priority 1: High Level Alerts
    if top_alert_text:
        cv2.putText(frame, f"ALERT: {top_alert_text}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    
    # Priority 2: Confirmed Status
    # Check if confirmed text is an Emergency to force RED color
    display_color = status_color
    if "EMERGENCY" in confirmed_text:
        display_color = (0, 0, 255)

    cv2.putText(frame, f"Status: {confirmed_text}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2)

    cv2.imshow("Patient Monitor - All In One", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
