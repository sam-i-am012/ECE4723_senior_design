import cv2
from picamera2 import Picamera2
import time
import numpy as np
import tkinter as tk
import sys

# --- CONFIGURATION ---
# Monitor Settings
MONITOR_WIDTH = 1920
MONITOR_HEIGHT = 1080
X_OFFSET = 0
Y_OFFSET = 0  # Set to -1080 if monitor is above laptop

# Detection Settings
FACE_CASCADE_PATH = '/home/ece4723inter/Documents/ECE4723_senior_design/facial_recognition/haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = '/home/ece4723inter/Documents/ECE4723_senior_design/facial_recognition/haarcascade_eye.xml'

# Smoothing & Scaling
BLOCK_SIZE = 150
SMOOTHING = 0.6  # 0.0 = instant, 0.9 = very slow/smooth
avg_x, avg_y = 0, 0

# --- INITIALIZATION ---
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

print("InvisiVisor Initializing...")
time.sleep(2)

# --- GUI SETUP ---
root = tk.Tk()
root.overrideredirect(True)
root.geometry(f"{MONITOR_WIDTH}x{MONITOR_HEIGHT}+{X_OFFSET}+{Y_OFFSET}")
root.attributes('-topmost', True)
root.config(cursor="none", bg="white")

# CRITICAL FIX: Force focus so the window actually hears key presses
root.focus_force()

canvas = tk.Canvas(root, highlightthickness=0, bg="white")
canvas.pack(fill="both", expand=True)

# Create the glare block
rect = canvas.create_rectangle(-200, -200, 0, 0, fill="black")

def quit_app(event=None):
    """Safely shut down the camera and the GUI and kill the process."""
    print("\nInvisiVisor shutting down...")
    try:
        picam2.stop()
        cv2.destroyAllWindows()
    except:
        pass
    root.quit()
    root.destroy()
    sys.exit(0) # Hard exit to ensure terminal control returns

def update_visor():
    global avg_x, avg_y
    
    try:
        # 1. Capture Frame
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Detect Face
        faces = face_cascade.detectMultiScale(gray, 1.1, 8)
        
        found_eye = False
        for (x, y, w, h) in faces:
            # Look for eyes in the top 60% of the face
            roi_gray = gray[y:y+int(0.6*h), x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            
            if len(eyes) > 0:
                ex, ey, ew, eh = eyes[0]
                target_x = x + ex + (ew // 2)
                target_y = y + ey + (eh // 2)
                
                # --- COORDINATE MAPPING ---
                mapped_x = (target_x / 640) * MONITOR_WIDTH
                mapped_y = (target_y / 480) * MONITOR_HEIGHT
                
                # --- SMOOTHING ---
                avg_x = (avg_x * SMOOTHING) + (mapped_x * (1 - SMOOTHING))
                avg_y = (avg_y * SMOOTHING) + (mapped_y * (1 - SMOOTHING))
                found_eye = True
                break 

        # 3. Update GUI
        if found_eye:
            x1 = avg_x - (BLOCK_SIZE / 2)
            y1 = avg_y - (BLOCK_SIZE / 2)
            x2 = avg_x + (BLOCK_SIZE / 2)
            y2 = avg_y + (BLOCK_SIZE / 2)
            canvas.coords(rect, x1, y1, x2, y2)

        # 4. Schedule the next update
        root.after(10, update_visor)
    except Exception as e:
        print(f"Loop Error: {e}")
        quit_app()

# Key Bindings - Added 'q' and 'Ctrl+C' as fallbacks
root.bind('<Escape>', quit_app)
root.bind('q', quit_app)
root.bind('<Control-c>', quit_app)

# Start the loop
root.after(10, update_visor)

try:
    root.mainloop()
except KeyboardInterrupt:
    quit_app()
finally:
    # Final safety cleanup
    try:
        picam2.stop()
        cv2.destroyAllWindows()
    except:
        pass