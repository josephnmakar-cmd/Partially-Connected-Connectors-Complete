import cv2
import tkinter as tk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import time

# --- CONFIGURATION ---
CAMERA_INDEX = 1
MODEL_PATH = 'full_image_ai.pth'
CLASS_NAMES = ['fail_disconnected', 'fail_partial', 'pass_seated'] 
LOG_DIR = "inspection_logs"
# ---------------------

# Create the logging directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# 1. DEFINE THE AI'S VISION PIPELINE
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. LOAD THE BRAIN
print("Loading Full-Frame AI Model...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.eval() 
print("AI Model Armed and Ready.")

# 3. THE TKINTER UI
class FullFrameGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("AI Quality Inspection System")

        # Variables to hold data for the capture button
        self.current_frame = None
        self.current_status = "UNKNOWN"

        # --- LAYOUT ---
        self.sidebar = tk.Frame(window, width=300, bg="#2c3e50", padx=20, pady=20)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        self.video_frame = tk.Frame(window, bg="black")
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(self.sidebar, text="SYSTEM STATUS", fg="white", bg="#2c3e50", font=("Arial", 16, "bold")).pack(pady=20)

        # Output label for the AI's decision
        self.ai_output_label = tk.Label(self.sidebar, text="STARTING...", fg="#bdc3c7", bg="#2c3e50", font=("Arial", 14, "bold"))
        self.ai_output_label.pack(pady=30)

        # The Capture Button
        self.capture_btn = tk.Button(self.sidebar, text="📸 CAPTURE SNAPSHOT", font=("Arial", 12, "bold"), 
                                  bg="#3498db", fg="white", command=self.capture_image)
        self.capture_btn.pack(pady=30, fill=tk.X, ipady=15)

        self.canvas = tk.Label(self.video_frame)
        self.canvas.pack()

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.update_video()

    def capture_image(self):
        # This runs when the button is clicked!
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # Save format: inspection_logs/pass_seated_20260416_143000.jpg
            filename = f"{LOG_DIR}/{self.current_status}_{timestamp}.jpg"
            
            # We save the raw frame without the Tkinter UI drawings so it's a clean record
            cv2.imwrite(filename, self.current_frame)
            print(f"Snapshot saved: {filename}")
            
            # Flash the button green to give the user feedback
            self.capture_btn.config(bg="#2ecc71", text="✅ SAVED!")
            self.window.after(1000, lambda: self.capture_btn.config(bg="#3498db", text="📸 CAPTURE SNAPSHOT"))

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy() # Store the clean frame for the Capture button
            display_frame = frame.copy()

            # --- THE FULL FRAME AI INFERENCE ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(rgb_frame).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)
                self.current_status = CLASS_NAMES[preds[0].item()] # Update class variable

            # --- UPDATE THE UI LOGIC ---
            if self.current_status == "pass_seated":
                color = (0, 255, 0)
                display_text = "PASS - FULLY SEATED"
            elif self.current_status == "fail_partial":
                color = (0, 165, 255)
                display_text = "FAIL - PARTIAL"
            else:
                color = (0, 0, 255)
                display_text = "FAIL - DISCONNECTED"

            cv2.rectangle(display_frame, (0, 0), (1280, 720), color, 20)
            cv2.putText(display_frame, display_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
            
            self.ai_output_label.config(text=display_text, fg=f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}")

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
    app = FullFrameGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()