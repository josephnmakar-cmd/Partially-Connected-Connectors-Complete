import cv2
import numpy as np

# --- CONFIGURATION ---
CAMERA_INDEX = 1        
MAX_ACCEPTABLE_GAP = 50 # Maximum allowed gap in pixels
NOTCH_THRESHOLD = 10     # Minimum texture pixels needed to confirm the notch
# ---------------------

def run_live_inspection():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    print("--- INITIALIZATION ---")
    print("Draw a box around the connector (include the brown plastic and the gap).")
    print("Press ENTER to confirm.")
    
    roi_box = cv2.selectROI("Select Tracking Anchor", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Tracking Anchor")
    
    x, y, w, h = roi_box
    if w == 0 or h == 0:
        print("Selection canceled.")
        return
        
    template = first_frame[y:y+h, x:x+w]
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    print("Template saved! Starting live inspection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- STEP A: TRACKING ---
        res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > 0.6:
            track_x, track_y = max_loc
            
            # Draw tracking box
            cv2.rectangle(display_frame, (track_x, track_y), (track_x + w, track_y + h), (0, 255, 255), 2)

            # --- STEP B: EDGE DETECTION (The Gap) ---
            dynamic_roi = frame[track_y:track_y+h, track_x:track_x+w]
            roi_gray = cv2.cvtColor(dynamic_roi, cv2.COLOR_BGR2GRAY)
            
            blurred = cv2.bilateralFilter(roi_gray, 9, 75, 75)
            sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            abs_sobel_x = cv2.convertScaleAbs(sobel_x)
            _, thresh_edges = cv2.threshold(abs_sobel_x, 50, 255, cv2.THRESH_BINARY)
            
            column_sums = np.sum(thresh_edges, axis=0)
            peak_threshold = np.max(column_sums) * 0.5
            peaks = np.where(column_sums > peak_threshold)[0]
            
            # --- STEP C: SENSOR FUSION LOGIC (Gap + Notch) ---
            if len(peaks) > 1:
                left_edge = peaks[0]
                right_edge = peaks[-1]
                gap_distance = right_edge - left_edge
                
                # Draw the gap lines
                cv2.line(display_frame, (track_x + left_edge, track_y), (track_x + left_edge, track_y + h), (255, 0, 0), 2)
                cv2.line(display_frame, (track_x + right_edge, track_y), (track_x + right_edge, track_y + h), (0, 0, 255), 2)
                
                # 1. Define the tiny Sub-ROI for the Notch
                y_start = int(h * 0.25)
                y_end = int(h * 0.75)
                safe_left = max(0, left_edge)
                safe_right = min(w, right_edge)
                
                # 2. Extract Sub-ROI and run Canny
                if safe_right > safe_left:
                    gap_interior = roi_gray[y_start:y_end, safe_left:safe_right]
                    internal_edges = cv2.Canny(gap_interior, 30, 100)
                    notch_texture_score = cv2.countNonZero(internal_edges)
                else:
                    notch_texture_score = 0
                
                # Draw the purple box where the notch check is happening
                cv2.rectangle(display_frame, 
                              (track_x + safe_left, track_y + y_start), 
                              (track_x + safe_right, track_y + y_end), 
                              (255, 0, 255), 1)

                # 3. The Double-Validation Decision
                has_small_gap = gap_distance <= MAX_ACCEPTABLE_GAP
                has_notch = notch_texture_score > NOTCH_THRESHOLD

                if has_small_gap and has_notch:
                    status = f"PASS (Gap: {gap_distance}px | Notch: {notch_texture_score})"
                    color = (0, 255, 0)
                elif has_small_gap and not has_notch:
                    status = f"WARN (Gap OK, but NO NOTCH detected)"
                    color = (0, 165, 255) 
                else:
                    status = f"FAIL (Gap: {gap_distance}px | Plain Brown)"
                    color = (0, 0, 255) 
                    
                # Print results on screen
                cv2.putText(display_frame, status, (track_x - 10, track_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(display_frame, "CALCULATING...", (track_x, track_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "SEARCHING FOR CONNECTOR...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Production Line - Double Validation", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_inspection()