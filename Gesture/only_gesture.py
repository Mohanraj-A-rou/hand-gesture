import cv2
import mediapipe as mp
import time
import math

# ==========================================
#        1. INITIALIZATION & SETUP
# ==========================================

# --- MediaPipe Solutions ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Detectors
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- Configuration ---
# Time to hold ANY gesture before it activates
GESTURE_HOLD_TIME = 3.0           

# --- State Variables ---
current_gesture = None
gesture_start_time = 0
confirmed_text = "Monitoring Active"
status_color = (0, 255, 0) # Green

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
    
    # --- MODIFIED MAPPINGS ---
    if f == [1, 0, 0, 0, 0]: return "Adjust Position"
    if f == [1, 1, 0, 0, 0]: return "Need Water/Food"
    if f == [0, 1, 0, 0, 0]: return "Want Restroom"
    if f == [0, 0, 1, 1, 1]: return "Want Medicine"
    if f == [0, 1, 1, 0, 0]: return "Call Caregiver"
    if f == [1, 1, 1, 0, 0]: return "Uncomfortable"
    
    return None

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

    countdown_text = ""
    
    # ----------------------------------------------------
    # GESTURE RECOGNITION (Unified 3s Timer)
    # ----------------------------------------------------
    hand_results = hands.process(rgb)
    
    if hand_results.multi_hand_landmarks:
        for idx, hand_lm in enumerate(hand_results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            
            # Get Hand Label (Left/Right) and Finger States
            if hand_results.multi_handedness:
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
                            confirmed_text = gesture
                            status_color = (0, 255, 0) # Green for confirmed
                        else:
                            # --- COUNTING DOWN ---
                            countdown_text = f"Holding: {seconds_left}s"
                            
                            # Draw countdown near the hand
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
    
    cv2.putText(frame, f"Status: {confirmed_text}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    cv2.imshow("Patient Monitor - Gestures Only", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()