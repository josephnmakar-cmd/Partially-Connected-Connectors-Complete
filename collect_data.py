import cv2
import os
import time
import random

# --- CONFIGURATION ---
CAMERA_INDEX = 1
DATA_DIR = "connector_dataset"
SAVE_INTERVAL = 0.5  # Saves a photo every 0.5 seconds when active
# ---------------------

def setup_folders():
    categories = ['pass_seated', 'fail_partial', 'fail_disconnected']
    for split in ['train', 'val']:
        for cat in categories:
            os.makedirs(os.path.join(DATA_DIR, split, cat), exist_ok=True)
    print(f"Dataset folders successfully built in ./{DATA_DIR}/")

def run_data_harvester():
    setup_folders()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    ret, setup_frame = cap.read()
    if not ret: return

    print("\n--- SETUP ---")
    print("Draw a tight box around the connector's mating area (where the gap happens).")
    roi_box = cv2.selectROI("Select CNN Target", setup_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select CNN Target")
    tx, ty, tw, th = roi_box
    
    # Save the template for tracking
    template_gray = cv2.cvtColor(setup_frame[ty:ty+th, tx:tx+tw], cv2.COLOR_BGR2GRAY)

    current_class = None
    last_save_time = time.time()
    
    print("\n--- HARVESTING READY ---")
    print("Press '1' to start recording PASS")
    print("Press '2' to start recording PARTIAL")
    print("Press '3' to start recording DISCONNECTED")
    print("Press '0' to PAUSE recording")
    print("Press 'q' to QUIT")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. TRACK THE CONNECTOR
        res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.6:
            track_x, track_y = max_loc
            
            # Draw the bounding box
            cv2.rectangle(display_frame, (track_x, track_y), (track_x + tw, track_y + th), (255, 255, 0), 2)
            
            # Crop exactly what the AI will see
            cropped_roi = frame[track_y:track_y+th, track_x:track_x+tw]

            # 2. SAVE THE IMAGE (If recording is active)
            if current_class is not None:
                current_time = time.time()
                if current_time - last_save_time > SAVE_INTERVAL:
                    
                    # The 80/20 Split: 80% chance to train, 20% chance to val
                    target_folder = "train" if random.random() < 0.8 else "val"
                    
                    timestamp = time.strftime("%H%M%S") + str(int((current_time % 1) * 1000))
                    filename = f"{DATA_DIR}/{target_folder}/{current_class}/img_{timestamp}.jpg"
                    
                    # Save just the cropped image, not the whole screen!
                    cv2.imwrite(filename, cropped_roi)
                    
                    # Draw a flash on the screen so you know it took a picture
                    cv2.rectangle(display_frame, (0,0), (1280, 720), (255, 255, 255), 10)
                    last_save_time = current_time

        # 3. ON-SCREEN UI
        status_text = f"RECORDING: {current_class}" if current_class else "PAUSED"
        color = (0, 0, 255) if current_class else (0, 255, 255)
        cv2.putText(display_frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.imshow("Smart Data Harvester", display_frame)
        
        # Display the AI's view in a tiny separate window
        if max_val > 0.6 and cropped_roi.size > 0:
            cv2.imshow("AI Vision", cropped_roi)

        # 4. KEYBOARD CONTROLS
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('0'): current_class = None
        elif key == ord('1'): current_class = 'pass_seated'
        elif key == ord('2'): current_class = 'fail_partial'
        elif key == ord('3'): current_class = 'fail_disconnected'

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_data_harvester()