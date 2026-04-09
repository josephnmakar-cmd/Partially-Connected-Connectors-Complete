import cv2
import numpy as np
import os
import time

# --- CONFIGURATION ---
CAMERA_INDEX = 1        
LOG_DIR = "defect_logs"  
# ---------------------

def nothing(x):
    pass

def run_ultimate_inspection():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    ret, setup_frame = cap.read()
    if not ret: return

    print("--- SETUP PHASE ---")
    
    # 1. The Tracking Anchor
    print("1. Draw a large box around the ENTIRE connector body to track it.")
    anchor_box = cv2.selectROI("Setup", setup_frame, fromCenter=False, showCrosshair=True)
    tx, ty, tw, th = anchor_box
    template_gray = cv2.cvtColor(setup_frame[ty:ty+th, tx:tx+tw], cv2.COLOR_BGR2GRAY)

    # 2. The Gap Window
    print("2. Draw the MAIN GAP Window.")
    gap_box = cv2.selectROI("Setup", setup_frame, fromCenter=False, showCrosshair=True)
    gx, gy, gw, gh = gap_box

    # 3. The Notch Window
    print("3. Draw the NOTCH Window.")
    notch_box = cv2.selectROI("Setup", setup_frame, fromCenter=False, showCrosshair=True)
    nx, ny, nw, nh = notch_box
    
    cv2.destroyWindow("Setup")

    # Calculate Offsets (Gluing the windows to the anchor)
    gap_offset_x = gx - tx
    gap_offset_y = gy - ty
    notch_offset_x = nx - tx
    notch_offset_y = ny - ty

    # Initialize UI and State Machine
    cv2.namedWindow("Inspection System")
    cv2.createTrackbar("Gap Thresh", "Inspection System", 20, 50, nothing)
    cv2.createTrackbar("Notch Thresh", "Inspection System", 700, 2000, nothing)
    
    current_mode = "CALIBRATION"
    is_currently_failing = False
    
    print("\n--- SYSTEM READY ---")
    print("Press 'm' to switch between Calibration and Production modes.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # UI & Mode Logic
        if current_mode == "CALIBRATION":
            live_gap_thresh = cv2.getTrackbarPos("Gap Thresh", "Inspection System")
            live_notch_thresh = cv2.getTrackbarPos("Notch Thresh", "Inspection System")
            cv2.putText(display_frame, "MODE: CALIBRATION (Logging Disabled)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "MODE: PRODUCTION (Logging ARMED)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- STEP A: TRACKING ---
        res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > 0.6:
            track_x, track_y = max_loc
            
            # Apply offsets to find the dynamic Gap and Notch boxes
            dyn_gx = track_x + gap_offset_x
            dyn_gy = track_y + gap_offset_y
            dyn_nx = track_x + notch_offset_x
            dyn_ny = track_y + notch_offset_y

            # Draw the boxes
            cv2.rectangle(display_frame, (dyn_gx, dyn_gy), (dyn_gx + gw, dyn_gy + gh), (255, 255, 255), 2)
            cv2.rectangle(display_frame, (dyn_nx, dyn_ny), (dyn_nx + nw, dyn_ny + nh), (255, 0, 255), 2)

           # --- STEP B: PROCESS GAP ---
            gap_roi = frame_gray[dyn_gy:dyn_gy+gh, dyn_gx:dyn_gx+gw]
            
            # THE SAFETY NET: Check if the math pushed the box off-screen
            if gap_roi.size == 0:
                cv2.putText(display_frame, "GAP ROI OFF SCREEN", (track_x, track_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                continue # Skip the rest of the math for this single frame
                
            blurred = cv2.bilateralFilter(gap_roi, 9, 75, 75)
            sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            abs_sobel_x = cv2.convertScaleAbs(sobel_x)
            _, thresh_edges = cv2.threshold(abs_sobel_x, 50, 255, cv2.THRESH_BINARY)
            
            column_sums = np.sum(thresh_edges, axis=0)
            peak_threshold = np.max(column_sums) * 0.5
            peaks = np.where(column_sums > peak_threshold)[0]
            
            gap_distance = None
            if len(peaks) > 1:
                left_edge = peaks[0]
                right_edge = peaks[-1]
                gap_distance = right_edge - left_edge
                cv2.line(display_frame, (dyn_gx + left_edge, dyn_gy), (dyn_gx + left_edge, dyn_gy + gh), (255, 0, 0), 2)
                cv2.line(display_frame, (dyn_gx + right_edge, dyn_gy), (dyn_gx + right_edge, dyn_gy + gh), (0, 0, 255), 2)

            # --- STEP C: PROCESS NOTCH ---
            notch_roi = frame_gray[dyn_ny:dyn_ny+nh, dyn_nx:dyn_nx+nw]
            
            # THE SAFETY NET: Check if the notch box went off-screen
            if notch_roi.size == 0:
                continue 
                
            internal_edges = cv2.Canny(notch_roi, 30, 100)
            notch_texture_score = cv2.countNonZero(internal_edges)
            # --- STEP D: LOGIC USING LIVE SLIDERS ---
            is_fail_state = False

            if notch_texture_score < 400:
                status = f"FAIL - DISCONNECTED (Notch: {notch_texture_score})"
                color = (0, 0, 255)
                is_fail_state = True
            elif 400 <= notch_texture_score < live_notch_thresh:
                status = f"FAIL - PARTIAL (Notch: {notch_texture_score})"
                color = (0, 165, 255)
                is_fail_state = True
            elif notch_texture_score >= live_notch_thresh:
                if gap_distance is not None and gap_distance <= live_gap_thresh:
                    status = f"PASS - FULLY SEATED (Gap: {gap_distance}px | Notch: {notch_texture_score})"
                    color = (0, 255, 0)
                    is_fail_state = False
                else:
                    gap_text = f"{gap_distance}px" if gap_distance is not None else "Lost"
                    status = f"FAIL - GAP ERROR (Gap: {gap_text})"
                    color = (0, 0, 255)
                    is_fail_state = True

            cv2.putText(display_frame, status, (track_x, track_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # --- STEP E: DEFECT LOGGING (Only in Production Mode) ---
            if current_mode == "PRODUCTION":
                if is_fail_state and not is_currently_failing:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"{LOG_DIR}/defect_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame) 
                    print(f"[{timestamp}] Defect Logged!")
                    is_currently_failing = True 
                elif not is_fail_state:
                    is_currently_failing = False
        else:
            cv2.putText(display_frame, "SEARCHING FOR CONNECTOR...", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Inspection System", display_frame)
        
        # Keyboard commands
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            current_mode = "PRODUCTION" if current_mode == "CALIBRATION" else "CALIBRATION"
            print(f"Switched to {current_mode} MODE")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_ultimate_inspection()