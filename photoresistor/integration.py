import cv2
from picamera2 import Picamera2
import time
import numpy as np
import RPi.GPIO as GPIO
import sys

# --- HARDWARE CONFIGURATION (GPIO) ---
PIN_TO_CIRCUIT = 7
GLARE_THRESHOLD = 500  # ADJUST THIS: Lower count = more light. 
LIGHT_CHECK_INTERVAL = 2.0 # Only check light every 2 seconds to prevent lag

GPIO.setmode(GPIO.BOARD)

def rc_time(pin):
    # This function is now faster because the sleep happens outside the main loop logic
    count = 0
    # Discharge the capacitor
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)
    time.sleep(0.01) # Reduced from 0.1 to 0.01 for faster polling
    # Switch to input and time how long it takes to go HIGH
    GPIO.setup(pin, GPIO.IN)
    while (GPIO.input(pin) == GPIO.LOW):
        count += 1
        if count > 50000: break # Safety timeout
    return count

# --- CAMERA & CV CONFIGURATION ---
face_cascade = cv2.CascadeClassifier('/home/ece4723inter/Documents/ECE4723_senior_design/facial_recognition/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/ece4723inter/Documents/ECE4723_senior_design/facial_recognition/haarcascade_eye.xml')

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Display Settings
CAM_W, CAM_H = 640, 480
SUN_W, SUN_H = 2560, 1600
SECOND_SCREEN_X = 1920
SECOND_SCREEN_Y = 0

print("InvisiVisor System Warming up...")
time.sleep(2)

# Window Setup
cv2.namedWindow("camera feed", cv2.WINDOW_NORMAL)
cv2.moveWindow("camera feed", 0, 0)

cv2.namedWindow("Sun Blocker", cv2.WINDOW_NORMAL)
cv2.moveWindow("Sun Blocker", SECOND_SCREEN_X, SECOND_SCREEN_Y)
cv2.setWindowProperty("Sun Blocker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

last_box = None
is_glare_detected = False
last_light_check = 0

try:
    while True:
        # 1. Non-blocking Light Level Check
        # Only runs the slow rc_time function once every few seconds
        current_time = time.time()
        if current_time - last_light_check > LIGHT_CHECK_INTERVAL:
            light_count = rc_time(PIN_TO_CIRCUIT)
            is_glare_detected = light_count < GLARE_THRESHOLD
            last_light_check = current_time
            print(f"\nLight Sensor Updated: {light_count} | Glare: {is_glare_detected}")

        # 2. Capture and Process Frame (Now runs at full speed)
        frame = picam2.capture_array()
        frame = cv2.flip(frame, -1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Prepare the Visor Screen (Initially all white/transparent)
        white_screen = np.ones((SUN_H, SUN_W, 3), dtype=np.uint8) * 255
        
        # 3. Detect Faces & Eyes
        faces = face_cascade.detectMultiScale(gray, 1.1, 8)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            roi_gray = gray[y : y+int(h * 0.6), x : x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            
            eyes_in_frame = [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]
            
            if len(eyes_in_frame) >= 1: 
                ex1 = min(ex for (ex, ey, ew, eh) in eyes_in_frame)
                ey1 = min(ey for (ex, ey, ew, eh) in eyes_in_frame)
                ex2 = max(ex + ew for (ex, ey, ew, eh) in eyes_in_frame)
                ey2 = max(ey + eh for (ex, ey, ew, eh) in eyes_in_frame)
                
                PAD_X, PAD_Y = 20, 10 
                last_box = (
                    max(ex1 - PAD_X, 0),
                    max(ey1 - PAD_Y, 0),
                    min(ex2 + PAD_X, CAM_W),
                    min(ey2 + PAD_Y, CAM_H))
                
            # 4. Apply Darkening Logic ONLY if Glare is present
            if last_box is not None:
                bx1, by1, bx2, by2 = last_box
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                
                if is_glare_detected:
                    scale_x = SUN_W / CAM_W
                    scale_y = SUN_H / CAM_H
                    
                    sx1 = int(bx1 * scale_x)
                    sy1 = int(by1 * scale_y)
                    sx2 = int(bx2 * scale_x)
                    sy2 = int(by2 * scale_y)
                    
                    cv2.rectangle(white_screen, 
                                 (SUN_W - sx2, sy1), 
                                 (SUN_W - sx1, sy2), 
                                 (0, 0, 0), 
                                 thickness=-1)
            
        # 5. Output to Displays
        cv2.imshow("camera feed", frame)
        cv2.imshow("Sun Blocker", white_screen)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    GPIO.cleanup()