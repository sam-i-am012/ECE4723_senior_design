# Working code - shows bounding boxes around faces
import cv2
from picamera2 import Picamera2
import time
import numpy as np

face_cascade = cv2.CascadeClassifier('/home/ece4723inter/Documents/ECE4723_senior_design/facial_recognition/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/ece4723inter/Documents/ECE4723_senior_design/facial_recognition/haarcascade_eye.xml')

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# display settings
CAM_DISPLAY_W, CAM_DISPLAY_H = 640, 480
SUN_DISPLAY_W, SUN_DISPLAY_H = 2560, 1600
SECOND_SCREEN_X = 1920
SECOND_SCREEN_Y = 0

print("Camera warming up...")
time.sleep(2)

cv2.namedWindow("camera feed", cv2.WINDOW_NORMAL)
cv2.moveWindow("camera feed", 0, 0)
cv2.namedWindow("Sun Blocker", cv2.WINDOW_NORMAL)
cv2.moveWindow("Sun Blocker", SECOND_SCREEN_X, SECOND_SCREEN_Y)
cv2.setWindowProperty("Sun Blocker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
last_box = None

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, -1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # white screen for sun blocker
        white_screen = np.ones((SUN_DISPLAY_H, SUN_DISPLAY_W, 3), dtype=np.uint8) * 255

        faces = face_cascade.detectMultiScale(gray, 1.1, 8)
        
        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            roi_gray = gray[y : y+int(h * 0.6), x : x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            print(eyes)
            
            eyes_in_frame = [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes] 

            if len(eyes_in_frame) == 2:
                ex1 = min(ex for (ex, ey, ew, eh) in eyes_in_frame)
                ey1 = min(ey for (ex, ey, ew, eh) in eyes_in_frame)
                ex2 = max(ex + ew for (ex, ey, ew, eh) in eyes_in_frame)
                ey2 = max(ey + eh for (ex, ey, ew, eh) in eyes_in_frame)
                
                PAD_X, PAD_Y = 10, 5
                last_box = (
                    max(ex1 - PAD_X, 0),
                    max(ey1 - PAD_Y, 0),
                    min(ex2 + PAD_X, CAM_DISPLAY_W),
                    min(ey2 + PAD_Y, CAM_DISPLAY_H))
                
            if last_box is not None:
                bx1, by1, bx2, by2 = last_box
                cv2.rectangle(frame, (bx1,by1), (bx2, by2), (0, 255, 0), 2)
                
                scale_x = SUN_DISPLAY_W / CAM_DISPLAY_W
                scale_y = SUN_DISPLAY_H / CAM_DISPLAY_H
                sx1 = int(bx1 * scale_x)
                sy1 = int(by1 * scale_y)
                sx2 = int(bx2 * scale_x)
                sy2 = int(by2 * scale_y)
                
                # scaling code
                cx = (sx1 + sx2) // 2
                cy = (sy1 + sy2) // 2
                half_w = int((sx2 - sx1) * 2.5 / 2)
                half_h = int((sy2 - sy1) * 2.5 / 2)
                sx1 = max(cx - half_w, 0)
                sy1 = max(cy - half_h, 0)
                sx2 = min(cx + half_w, SUN_DISPLAY_W)
                sy2 = min(cy + half_h, SUN_DISPLAY_H)
                cv2.rectangle(white_screen, (SUN_DISPLAY_W - sx2, sy1), (SUN_DISPLAY_W - sx1, sy2), (0, 0, 0), thickness=-1)
            
        cv2.imshow("camera feed", frame)
        # for 2nd display:
        cv2.imshow("Sun Blocker", white_screen)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()