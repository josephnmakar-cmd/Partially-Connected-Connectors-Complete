import cv2
import os
import time
import random

# --- CONFIGURATION ---
CAMERA_INDEX = 1
DATA_DIR = "D:\AI and Automation- University West\Master Thesis\Project code\Thesis Project\Test_FullFrame"  # Notice the NEW folder name so we don't mix data!
SAVE_INTERVAL = 1
# ---------------------

def setup_folders():
    categories = ['pass_seated', 'fail_partial', 'fail_disconnected']
    for split in ['train', 'val']:
        for cat in categories:
            os.makedirs(os.path.join(DATA_DIR, split, cat), exist_ok=True)
    print(f"Dataset folders built in ./{DATA_DIR}/")

def run_full_harvester():
    setup_folders()
    cap = cv2.VideoCapture(CAMERA_INDEX)1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_class = None
    last_save_time = time.time()

    print("\n--- FULL IMAGE HARVESTING ---")
    print("Press '1' to record PASS")
    print("Press '2' to record PARTIAL")
    print("Press '3' to record DISCONNECTED")
    print("Press '0' to PAUSE")
    print("Press 'q' to QUIT")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()

        # Save the ENTIRE frame
        if current_class is not None:
            current_time = time.time()
            if current_time - last_save_time > SAVE_INTERVAL:
                target_folder = "train" if random.random() < 0.8 else "val"
                timestamp = time.strftime("%H%M%S") + str(int((current_time % 1) * 1000))
                filename = f"{DATA_DIR}/{target_folder}/{current_class}/img_{timestamp}.jpg"
                
                cv2.imwrite(filename, frame) # Saving the full picture!
                
                # Screen flash effect
                cv2.rectangle(display_frame, (0,0), (1280, 720), (255, 255, 255), 10)
                last_save_time = current_time

        # UI Overlay
        status_text = f"RECORDING: {current_class}" if current_class else "PAUSED"
        color = (0, 0, 255) if current_class else (0, 255, 255)
        cv2.putText(display_frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.imshow("Full Image Harvester", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('0'): current_class = None
        elif key == ord('1'): current_class = 'pass_seated'
        elif key == ord('2'): current_class = 'fail_partial'
        elif key == ord('3'): current_class = 'fail_disconnected'

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_full_harvester()