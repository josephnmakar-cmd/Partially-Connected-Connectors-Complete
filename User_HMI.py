import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

CAMERA_INDEX = 1

class InspectionGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Seat Connector Inspection")
        
        self.current_mode = "PRODUCTION" # Start in clean production mode
        self.is_calibrated = False       # Tracks if we have drawn our boxes yet

        # 1. MAIN LAYOUT
        self.sidebar = tk.Frame(window, width=300, bg="#2c3e50", padx=20, pady=20)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        self.video_frame = tk.Frame(window, bg="black")
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 2. STATUS LABEL
        self.status_label = tk.Label(self.sidebar, text="SYSTEM IDLE", fg="white", bg="#2c3e50", font=("Arial", 16, "bold"))
        self.status_label.pack(pady=20)

        # 3. THE TRIGGER BUTTON
        self.calib_btn = tk.Button(self.sidebar, text="START CALIBRATION", font=("Arial", 12, "bold"), 
                                  bg="#f39c12", fg="white", command=self.run_setup_phase)
        self.calib_btn.pack(pady=10, fill=tk.X, ipady=10)

        # 4. THE HIDDEN CALIBRATION TOOLS
        # We put all the text boxes inside this frame, but WE DO NOT PACK IT YET.
        self.tools_frame = tk.Frame(self.sidebar, bg="#2c3e50")

        tk.Label(self.tools_frame, text="Max Gap (pixels):", fg="white", bg="#2c3e50").pack(anchor="w", pady=(10, 0))
        self.gap_entry = tk.Entry(self.tools_frame, font=("Arial", 14))
        self.gap_entry.insert(0, "12")
        self.gap_entry.pack(fill=tk.X)

        tk.Label(self.tools_frame, text="Min Notch Score:", fg="white", bg="#2c3e50").pack(anchor="w", pady=(20, 0))
        self.notch_entry = tk.Entry(self.tools_frame, font=("Arial", 14))
        self.notch_entry.insert(0, "700")
        self.notch_entry.pack(fill=tk.X)

        self.finish_btn = tk.Button(self.tools_frame, text="LOCK & RUN PRODUCTION", font=("Arial", 12, "bold"), 
                                  bg="#2ecc71", fg="white", command=self.finish_calibration)
        self.finish_btn.pack(pady=40, fill=tk.X, ipady=10)

        # 5. VIDEO SETUP
        self.canvas = tk.Label(self.video_frame)
        self.canvas.pack()

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Keep at 1280x720 for now to ensure stability
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.update_video()

    def run_setup_phase(self):
        # 1. Grab a fresh frame from the camera
        ret, setup_frame = self.cap.read()
        if not ret: return

        self.status_label.config(text="CALIBRATING...", fg="#f1c40f")

        # 2. RUN THE OPENCV DRAWING WINDOWS
        # (This will pop up over your Tkinter window)
        anchor_box = cv2.selectROI("1. Draw Anchor", setup_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("1. Draw Anchor")
        self.tx, self.ty, self.tw, self.th = anchor_box
        self.template_gray = cv2.cvtColor(setup_frame[self.ty:self.ty+self.th, self.tx:self.tx+self.tw], cv2.COLOR_BGR2GRAY)

        gap_box = cv2.selectROI("2. Draw Gap", setup_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("2. Draw Gap")
        gx, gy, self.gw, self.gh = gap_box

        notch_box = cv2.selectROI("3. Draw Notch", setup_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("3. Draw Notch")
        nx, ny, self.nw, self.nh = notch_box

        # Calculate offsets and save them to 'self' so the rest of the app can use them
        self.gap_offset_x = gx - self.tx
        self.gap_offset_y = gy - self.ty
        self.notch_offset_x = nx - self.tx
        self.notch_offset_y = ny - self.ty

        self.is_calibrated = True

        # 3. SWAP THE UI
        self.calib_btn.pack_forget()           # Hide the "Start Calibration" button
        self.tools_frame.pack(fill=tk.X)       # Reveal the text boxes and "Finish" button!
        self.current_mode = "CALIBRATION"

    def finish_calibration(self):
        # Swap the UI back!
        self.tools_frame.pack_forget()
        self.calib_btn.pack(pady=10, fill=tk.X, ipady=10)
        
        self.status_label.config(text="SYSTEM ONLINE", fg="#2ecc71")
        self.current_mode = "PRODUCTION"
    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            display_frame = frame.copy()

            # 1. SAFELY GET THRESHOLDS (Prevents crashes in Production Mode)
            live_gap = 50
            live_notch = 600
            try:
                live_gap = int(self.gap_entry.get())
                live_notch = int(self.notch_entry.get())
            except ValueError:
                pass # If the user types a letter by accident, just use defaults

            # 2. THE MATH AND DRAWING (Only runs after calibration)
            if self.is_calibrated:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.putText(display_frame, f"MODE: {self.current_mode}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # --- THE TRACKING ENGINE ---
                res = cv2.matchTemplate(frame_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if max_val > 0.6:
                    track_x, track_y = max_loc

                    # --- CALCULATE DYNAMIC BOXES ---
                    dyn_gx = track_x + self.gap_offset_x
                    dyn_gy = track_y + self.gap_offset_y
                    dyn_nx = track_x + self.notch_offset_x
                    dyn_ny = track_y + self.notch_offset_y

                    # Draw the floating boxes
                    cv2.rectangle(display_frame, (dyn_gx, dyn_gy), (dyn_gx + self.gw, dyn_gy + self.gh), (255, 255, 255), 2)
                    cv2.rectangle(display_frame, (dyn_nx, dyn_ny), (dyn_nx + self.nw, dyn_ny + self.nh), (255, 0, 255), 2)

                    # --- PROCESS GAP ---
                    gap_roi = frame_gray[dyn_gy:dyn_gy+self.gh, dyn_gx:dyn_gx+self.gw]
                    gap_distance = None
                    
                    if gap_roi.size > 0: # Safety Net
                        blurred = cv2.bilateralFilter(gap_roi, 9, 75, 75)
                        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
                        _, thresh_edges = cv2.threshold(abs_sobel_x, 50, 255, cv2.THRESH_BINARY)
                        
                        column_sums = np.sum(thresh_edges, axis=0)
                        peak_threshold = np.max(column_sums) * 0.5
                        peaks = np.where(column_sums > peak_threshold)[0]
                        
                        if len(peaks) > 1:
                            left_edge = peaks[0]
                            right_edge = peaks[-1]
                            gap_distance = right_edge - left_edge
                            cv2.line(display_frame, (dyn_gx + left_edge, dyn_gy), (dyn_gx + left_edge, dyn_gy + self.gh), (255, 0, 0), 2)
                            cv2.line(display_frame, (dyn_gx + right_edge, dyn_gy), (dyn_gx + right_edge, dyn_gy + self.gh), (0, 0, 255), 2)

                    # --- PROCESS NOTCH ---
                    notch_roi = frame_gray[dyn_ny:dyn_ny+self.nh, dyn_nx:dyn_nx+self.nw]
                    notch_score = 0
                    
                    if notch_roi.size > 0: # Safety Net
                        internal_edges = cv2.Canny(notch_roi, 30, 100)
                        notch_score = cv2.countNonZero(internal_edges)

                    # --- THE UI LOGIC ---
                    if notch_score < (live_notch-200):
                        status = f"FAIL - DISCONNECTED (Notch: {notch_score})"
                        color = (0, 0, 255)
                    elif (live_notch-200) <= notch_score < live_notch:
                        status = f"FAIL - PARTIAL (Notch: {notch_score})"
                        color = (0, 165, 255)
                    elif notch_score >= live_notch:
                        if gap_distance is not None and gap_distance <= live_gap:
                            status = f"PASS - FULLY SEATED (Gap:{gap_distance} Notch:{notch_score})"
                            color = (0, 255, 0)
                        else:
                            gap_text = f"{gap_distance}px" if gap_distance is not None else "Lost"
                            status = f"FAIL - GAP ERROR (Gap: {gap_text})"
                            color = (0, 0, 255)

                    cv2.putText(display_frame, status, (track_x, track_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(display_frame, "SEARCHING FOR CONNECTOR...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # 3. RESIZE AND UPDATE TKINTER
            # THE FIX: We are resizing 'display_frame' (which has drawings), NOT 'frame'!
            display_frame = cv2.resize(display_frame, (1280, 960))
            cv2_im = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_im)
            imgtk = ImageTk.PhotoImage(image=img)

            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)

        self.window.after(15, self.update_video)

    def on_closing(self):
        self.cap.release()
        self.window.destroy()
    # ONLY run math/text if the user has drawn the boxes
          # ONLY run math/text if the user has drawn the boxes
        if self.is_calibrated:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.putText(display_frame, f"MODE: {self.current_mode}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- THE TRACKING ENGINE ---
            res = cv2.matchTemplate(frame_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > 0.6:
                track_x, track_y = max_loc

                # --- CALCULATE DYNAMIC BOXES ---
                dyn_gx = track_x + self.gap_offset_x
                dyn_gy = track_y + self.gap_offset_y
                dyn_nx = track_x + self.notch_offset_x
                dyn_ny = track_y + self.notch_offset_y

                # Draw the floating boxes
                cv2.rectangle(display_frame, (dyn_gx, dyn_gy), (dyn_gx + self.gw, dyn_gy + self.gh), (255, 255, 255), 2)
                cv2.rectangle(display_frame, (dyn_nx, dyn_ny), (dyn_nx + self.nw, dyn_ny + self.nh), (255, 0, 255), 2)

                # --- PROCESS GAP ---
                gap_roi = frame_gray[dyn_gy:dyn_gy+self.gh, dyn_gx:dyn_gx+self.gw]
                gap_distance = None

                if gap_roi.size > 0: # Safety Net!
                    blurred = cv2.bilateralFilter(gap_roi, 9, 75, 75)
                    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
                    _, thresh_edges = cv2.threshold(abs_sobel_x, 50, 255, cv2.THRESH_BINARY)

                    column_sums = np.sum(thresh_edges, axis=0)
                    peak_threshold = np.max(column_sums) * 0.5
                    peaks = np.where(column_sums > peak_threshold)[0]

                    if len(peaks) > 1:
                        left_edge = peaks[0]
                        right_edge = peaks[-1]
                        gap_distance = right_edge - left_edge
                        cv2.line(display_frame, (dyn_gx + left_edge, dyn_gy), (dyn_gx + left_edge, dyn_gy + self.gh), (255, 0, 0), 2)
                        cv2.line(display_frame, (dyn_gx + right_edge, dyn_gy), (dyn_gx + right_edge, dyn_gy + self.gh), (0, 0, 255), 2)

                # --- PROCESS NOTCH ---
                notch_roi = frame_gray[dyn_ny:dyn_ny+self.nh, dyn_nx:dyn_nx+self.nw]
                notch_score = 0

                if notch_roi.size > 0: # Safety Net!
                    internal_edges = cv2.Canny(notch_roi, 30, 100)
                    notch_score = cv2.countNonZero(internal_edges)

                # --- THE UI LOGIC ---
                if notch_score < 400:
                    status = f"FAIL - DISCONNECTED (Notch: {notch_score})"
                    color = (0, 0, 255)
                elif 400 <= notch_score < live_notch:
                    status = f"FAIL - PARTIAL (Notch: {notch_score})"
                    color = (0, 165, 255)
                elif notch_score >= live_notch:
                    if gap_distance is not None and gap_distance <= live_gap:
                        status = f"PASS - FULLY SEATED (Gap:{gap_distance} Notch:{notch_score})"
                        color = (0, 255, 0)
                    else:
                        gap_text = f"{gap_distance}px" if gap_distance is not None else "Lost"
                        status = f"FAIL - GAP ERROR (Gap: {gap_text})"
                        color = (0, 0, 255)

                cv2.putText(display_frame, status, (track_x, track_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(display_frame, "SEARCHING FOR CONNECTOR...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# Start the App
if __name__ == "__main__":
    root = tk.Tk()
    app = InspectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
