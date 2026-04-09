import cv2
import numpy as np
import os
import time

# --- CONFIGURATION ---
CAMERA_INDEX = 1        
MAX_ACCEPTABLE_GAP = 45
LOG_DIR = "defect_logs"  # The folder where bad connectors will be saved
# ---------------------

def run_dual_window_inspection():
    # 1. Initialize Camera & Folders
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Automatically create the logging folder if it doesn't exist yet
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print(f"Created logging directory: {LOG_DIR}/")

    ret, setup_frame = cap.read()
    if not ret: return

    print("--- STEP 1: DEFINE THE GAP WINDOW ---")
    gap_box = cv2.selectROI("1. Select MAIN GAP Window", setup_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("1. Select MAIN GAP Window")
    fx, fy, fw, fh = gap_box
    if fw == 0 or fh == 0: return

    print("\n--- STEP 2: DEFINE THE NOTCH WINDOW ---")
    notch_box = cv2.selectROI("2. Select NOTCH Window", setup_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("2. Select NOTCH Window")
    nx, ny, nw, nh = notch_box
    if nw == 0 or nh == 0: return
        
    print("\nWindows locked! Running inspection... Press 'q' to quit.")

    # State tracker to prevent taking 30 pictures a second
    is_currently_failing = False 

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Draw the fixed windows
        cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 255), 2) 
        cv2.putText(display_frame, "Gap ROI", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(display_frame, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2) 
        cv2.putText(display_frame, "Notch ROI", (nx, ny - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # --- STEP A: PROCESS THE GAP ---
        gap_roi = frame_gray[fy:fy+fh, fx:fx+fw]
        blurred = cv2.bilateralFilter(gap_roi, 9, 75, 75)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        _, thresh_edges = cv2.threshold(abs_sobel_x, 50, 255, cv2.THRESH_BINARY)
        
        column_sums = np.sum(thresh_edges, axis=0)
        peak_threshold = np.max(column_sums) * 0.5
        peaks = np.where(column_sums > peak_threshold)[0]
        
        # Determine gap distance
        gap_distance = None
        if len(peaks) > 1:
            left_edge = peaks[0]
            right_edge = peaks[-1]
            gap_distance = right_edge - left_edge
            cv2.line(display_frame, (fx + left_edge, fy), (fx + left_edge, fy + fh), (255, 0, 0), 2)
            cv2.line(display_frame, (fx + right_edge, fy), (fx + right_edge, fy + fh), (0, 0, 255), 2)

        # --- STEP B: PROCESS THE NOTCH ---
        notch_roi = frame_gray[ny:ny+nh, nx:nx+nw]
        internal_edges = cv2.Canny(notch_roi, 30, 100)
        notch_texture_score = cv2.countNonZero(internal_edges)

        # --- STEP C: THE LOGIC & LOGGING SYSTEM ---
        is_fail_state = False # Resets every frame

        if notch_texture_score < 300:
            status = f"FAIL - DISCONNECTED (Notch: {notch_texture_score})"
            color = (0, 0, 255) 
            is_fail_state = True

        elif 300 <= notch_texture_score < 600:
            status = f"FAIL - PARTIAL (Notch: {notch_texture_score})"
            color = (0, 165, 255) 
            is_fail_state = True

        elif notch_texture_score >= 600:
            if gap_distance is not None and gap_distance <= MAX_ACCEPTABLE_GAP:
                status = f"PASS - FULLY SEATED (Gap: {gap_distance}px | Notch: {notch_texture_score})"
                color = (0, 255, 0) 
                is_fail_state = False # This is a good part!
            else:
                gap_text = f"{gap_distance}px" if gap_distance is not None else "Lost"
                status = f"FAIL - GAP ERROR (Gap: {gap_text} | Notch: {notch_texture_score})"
                color = (0, 0, 255) 
                is_fail_state = True

        # Draw the final result text
        cv2.putText(display_frame, status, (fx - 50, fy - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ---------------------------------------------------------
        # NEW: THE DEFECT IMAGE SAVER
        # ---------------------------------------------------------
        if is_fail_state and not is_currently_failing:
            # We just transitioned from a PASS to a FAIL. Take a picture!
            timestamp = time.strftime("%Y%m%d-%H%M%S") # e.g., 20231024-153022
            filename = f"{LOG_DIR}/defect_{timestamp}.jpg"
            
            # Save the image with all the bounding boxes and text drawn on it
            cv2.imwrite(filename, display_frame) 
            print(f"[{timestamp}] Defect logged: {filename} | Reason: {status}")
            
            # Lock the trigger so it doesn't take 30 photos a second
            is_currently_failing = True 
            
        elif not is_fail_state:
            # The connector is fully seated (PASS). Reset the trigger!
            is_currently_failing = False
        # ---------------------------------------------------------

        cv2.imshow("Dual Window Inspection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_dual_window_inspection()