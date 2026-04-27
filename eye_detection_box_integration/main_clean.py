import cv2
from picamera2 import Picamera2
import time
import numpy as np

# 1. Load Face Cascade (Haar eye cascade is no longer needed!)
face_cascade = cv2.CascadeClassifier('/home/ece4723inter/Documents/ECE4723_senior_design/facial_recognition/haarcascade_frontalface_default.xml')

# 2. Configure Camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})

# OPTIMIZATION: Tell the camera hardware to flip the image natively
config["main"]["transform"] = {"hflip": True, "vflip": True}

picam2.configure(config)
picam2.start()

# Display settings
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

# OPTIMIZATION: Create the white screen ONCE before the loop to save memory
white_screen = np.ones((SUN_DISPLAY_H, SUN_DISPLAY_W, 3), dtype=np.uint8) * 255
previous_shadow = None

try:
    while True:
        frame = picam2.capture_array()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # OPTIMIZATION: Shrink the image by half for much faster face detection
        small_gray = cv2.resize(gray, (320, 240))
        faces = face_cascade.detectMultiScale(small_gray, 1.1, 8)
        
        pupils_in_frame = []

        for (x, y, w, h) in faces:
            # Map the smaller coordinates back to the original 640x480 frame size
            x, y, w, h = x * 2, y * 2, w * 2, h * 2
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Geometrically define the two eye regions based on the face box dimensions
            ey_offset = int(h * 0.20)
            eh_size = int(h * 0.30)
            
            left_ex_offset = int(w * 0.15)
            right_ex_offset = int(w * 0.55)
            ew_size = int(w * 0.30)

            eye_regions = [
                (left_ex_offset, ey_offset, ew_size, eh_size),
                (right_ex_offset, ey_offset, ew_size, eh_size)
            ]

            for (ex, ey, ew, eh) in eye_regions:
                # 1. Extract geometric eye region directly from the grayscale frame
                eye_roi = gray[y + ey : y + ey + eh, x + ex : x + ex + ew]
                
                # Optional: Draw green box to visualize the geometric search region
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 1)
                
                # 2. Blur to remove noise
                blurred_eye = cv2.GaussianBlur(eye_roi, (7, 7), 0)
                
                # 3. Thresholding
                _, threshold = cv2.threshold(blurred_eye, 40, 255, cv2.THRESH_BINARY_INV)
                
                # 4. Find contours
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Sort contours by area to find the largest dark spot (the pupil)
                    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
                    largest_contour = contours[0]
                    
                    # Get the center point of the pupil contour
                    M = cv2.moments(largest_contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # Calculate absolute coordinates on the main frame
                        pupil_x = x + ex + cx
                        pupil_y = y + ey + cy
                        
                        pupils_in_frame.append((pupil_x, pupil_y))
                        
                        # Draw a tiny red dot on the pupil
                        cv2.circle(frame, (pupil_x, pupil_y), 3, (0, 0, 255), -1)

        # Update sun blocker logic using the pupils
        if len(pupils_in_frame) > 0:
            px_min = min(p[0] for p in pupils_in_frame)
            py_min = min(p[1] for p in pupils_in_frame)
            px_max = max(p[0] for p in pupils_in_frame)
            py_max = max(p[1] for p in pupils_in_frame)
            
            PAD_X, PAD_Y = 30, 20
            last_box = (
                max(px_min - PAD_X, 0),
                max(py_min - PAD_Y, 0),
                min(px_max + PAD_X, CAM_DISPLAY_W),
                min(py_max + PAD_Y, CAM_DISPLAY_H)
            )
            
        if last_box is not None:
            bx1, by1, bx2, by2 = last_box
            
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
            
            # OPTIMIZATION: Erase the old shadow by drawing white over it
            if previous_shadow is not None:
                old_x1, old_y1, old_x2, old_y2 = previous_shadow
                cv2.rectangle(white_screen, (old_x1, old_y1), (old_x2, old_y2), (255, 255, 255), thickness=-1)
            
            # Draw the new black shadow
            new_x1 = SUN_DISPLAY_W - sx2
            new_y1 = sy1
            new_x2 = SUN_DISPLAY_W - sx1
            new_y2 = sy2
            cv2.rectangle(white_screen, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 0), thickness=-1)
            
            # Save these coordinates to erase them next frame
            previous_shadow = (new_x1, new_y1, new_x2, new_y2)
        
        cv2.imshow("camera feed", frame)
        cv2.imshow("Sun Blocker", white_screen)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()