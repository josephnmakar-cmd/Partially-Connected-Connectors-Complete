import cv2
import tkinter as tk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np

# --- CONFIGURATION ---
CAMERA_INDEX = 1
MODEL_PATH = 'full_image_ai.pth'

# PyTorch ImageFolder sorts classes alphabetically!
CLASS_NAMES = ['fail_disconnected', 'fail_partial', 'pass_seated'] 
# ---------------------

# 1. DEFINE THE AI'S VISION PIPELINE
# We must process the live video exactly how we processed the training photos
preprocess = transforms.Compose([
    transforms.ToPILImage(),             # Convert OpenCV array to PIL Image
    transforms.Resize((224, 224)),       # Squish to ResNet size
    transforms.ToTensor(),               # Convert to PyTorch Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. LOAD THE BRAIN
print("Loading AI Model...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None) # We don't need Microsoft's weights, we have our own!
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

# Load your specific trained weights
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.eval() # CRITICAL: Tells the brain "This is a test, stop learning."
print("AI Model Armed and Ready.")

class AI_InspectionGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("AI Connector Inspection")
        self.is_tracking = False

        # --- LAYOUT ---
        self.sidebar = tk.Frame(window, width=300, bg="#2c3e50", padx=20, pady=20)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        self.video_frame = tk.Frame(window, bg="black")
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(self.sidebar, text="SYSTEM IDLE", fg="white", bg="#2c3e50", font=("Arial", 16, "bold"))
        self.status_label.pack(pady=20)

        self.track_btn = tk.Button(self.sidebar, text="TEACH CONNECTOR", font=("Arial", 12, "bold"), 
                                  bg="#f39c12", fg="white", command=self.run_setup_phase)
        self.track_btn.pack(pady=10, fill=tk.X, ipady=10)

        # Output label for the AI's decision
        self.ai_output_label = tk.Label(self.sidebar, text="WAITING FOR AI...", fg="#bdc3c7", bg="#2c3e50", font=("Arial", 14, "bold"))
        self.ai_output_label.pack(pady=50)

        self.canvas = tk.Label(self.video_frame)
        self.canvas.pack()

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.update_video()

    def run_setup_phase(self):
        ret, setup_frame = self.cap.read()
        if not ret: return

        # 1. THE ANCHOR (Tracks movement)
        print("Draw box around a STATIC part of the connector (e.g., the back housing)")
        anchor_box = cv2.selectROI("1. Draw Tracking Anchor", setup_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("1. Draw Tracking Anchor")
        self.ax, self.ay, self.aw, self.ah = anchor_box
        
        # Memorize the Anchor
        self.template_gray = cv2.cvtColor(setup_frame[self.ay:self.ay+self.ah, self.ax:self.ax+self.aw], cv2.COLOR_BGR2GRAY)

        # 2. THE AI BOX (Inspects the state)
        print("Draw tight box around the mating area (where the gap happens)")
        ai_box = cv2.selectROI("2. Draw AI Vision Box", setup_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("2. Draw AI Vision Box")
        self.cx, self.cy, self.cw, self.ch = ai_box

        # 3. CALCULATE THE OFFSET
        self.offset_x = self.cx - self.ax
        self.offset_y = self.cy - self.ay
        
        self.is_tracking = True
        self.status_label.config(text="AI ONLINE", fg="#2ecc71")
        self.track_btn.config(text="RE-TEACH CONNECTOR", bg="#34495e")
    
    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            display_frame = frame.copy()

            if self.is_tracking:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 1. TRACK THE CONNECTOR
                res = cv2.matchTemplate(frame_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val > 0.6:
                    # 1. FIND THE ANCHOR
                    track_ax, track_ay = max_loc
                    
                    # 2. CALCULATE THE AI BOX POSITION
                    ai_x = track_ax + self.offset_x
                    ai_y = track_ay + self.offset_y
                    
                    # Safety check to ensure the AI box doesn't go off the edge of the screen
                    if ai_y >= 0 and ai_x >= 0 and (ai_y + self.ch) < frame.shape[0] and (ai_x + self.cw) < frame.shape[1]:
                        
                        # Crop the AI Vision area
                        cropped_roi = frame[ai_y:ai_y+self.ch, ai_x:ai_x+self.cw]
                        
                        if cropped_roi.size > 0:
                            # --- THE AI BRAIN SWAP ---
                            rgb_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2RGB)
                            input_tensor = preprocess(rgb_roi).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                outputs = model(input_tensor)
                                _, preds = torch.max(outputs, 1)
                                prediction_text = CLASS_NAMES[preds[0].item()]

                            # --- UPDATE THE UI LOGIC ---
                            if prediction_text == "pass_seated":
                                color = (0, 255, 0)
                                display_text = "PASS - FULLY SEATED"
                            elif prediction_text == "fail_partial":
                                color = (0, 165, 255)
                                display_text = "FAIL - PARTIAL"
                            else:
                                color = (0, 0, 255)
                                display_text = "FAIL - DISCONNECTED"

                            # Draw the Tracker Anchor (Blue)
                            cv2.rectangle(display_frame, (track_ax, track_ay), (track_ax + self.aw, track_ay + self.ah), (255, 255, 0), 2)
                            cv2.putText(display_frame, "ANCHOR", (track_ax, track_ay - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                            # Draw the AI Bounding Box (Color changes based on state)
                            cv2.rectangle(display_frame, (ai_x, ai_y), (ai_x + self.cw, ai_y + self.ch), color, 3)
                            cv2.putText(display_frame, display_text, (ai_x, ai_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            
                            self.ai_output_label.config(text=display_text, fg=f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}")
                else:
                    self.ai_output_label.config(text="SEARCHING...", fg="#e74c3c")
                    cv2.putText(display_frame, "SEARCHING FOR CONNECTOR...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Resize for Tkinter display
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

if __name__ == "__main__":
    root = tk.Tk()
    app = AI_InspectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()