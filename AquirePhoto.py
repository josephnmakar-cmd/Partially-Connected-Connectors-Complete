import cv2
import os
import time
from tkinter import filedialog, Tk  # Used to browse files easily

# --- CONFIGURATION ---
CAMERA_INDEX = 1  
WIDTH, HEIGHT = 1280, 720
SAVE_PATH = "thesis_data"
for folder in ["full_frames", "roi_crops"]:
    os.makedirs(os.path.join(SAVE_PATH, folder), exist_ok=True)

def select_file():
    root = Tk()
    root.withdraw() # Hide the main tkinter window
    file_path = filedialog.askopenfilename(title="D:\AI and Automation- University West\Master Thesis\Project code\Thesis Project\connector_reference_roi.jpg" \
    "", 
                                          filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
    root.destroy()
    return file_path

def main():
    print("--- THESIS TOOL ---")
    print("Select Mode: [1] USB Camera  [2] Open Saved Image")
    mode = input("Choice: ")

    source = None
    is_live = False

    if mode == '1':
        source = cv2.VideoCapture(CAMERA_INDEX)
        source.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        source.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        is_live = True
        if not source.isOpened():
            print("Error: Could not open USB camera.")
            return
    else:
        file_path = select_file()
        if not file_path:
            print("No file selected.")
            return
        source = cv2.imread(file_path)
        is_live = False

    roi_coords = None 
    print("\nControls: 's'=Select ROI | 'c'=Capture/Save | 'q'=Quit")

    while True:
        if is_live:
            ret, frame = source.read()
            if not ret: break
        else:
            frame = source.copy() # Constant frame from file

        display_frame = frame.copy()
        if roi_coords:
            x, y, w, h = roi_coords
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Inspection Tool", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # --- SELECT ROI ---
        if key == ord('s'):
            roi_box = cv2.selectROI("Select Connector ROI", frame, fromCenter=False)
            if roi_box[2] > 0 and roi_box[3] > 0:
                roi_coords = [int(v) for v in roi_box]
            cv2.destroyWindow("Select Connector ROI")

        # --- CAPTURE & SAVE ---
        elif key == ord('c'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            full_name = f"full_{timestamp}.jpg"
            cv2.imwrite(os.path.join(SAVE_PATH, "full_frames", full_name), frame)
            
            if roi_coords:
                x, y, w, h = roi_coords
                roi_crop = frame[y:y+h, x:x+w]
                crop_name = f"roi_{timestamp}.jpg"
                cv2.imwrite(os.path.join(SAVE_PATH, "roi_crops", crop_name), roi_crop)
                print(f"Saved: {crop_name}")
            else:
                print("No ROI defined. Saved full frame only.")

        elif key == ord('q'):
            break

    if is_live: source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
