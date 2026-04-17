# Working code - shows bounding boxes around faces
import cv2
from picamera2 import Picamera2
import time
import numpy as np

# 1. Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier('/home/ece4723inter/Documents/ECE4723_senior_design/facial_recognition/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/ece4723inter/Documents/ECE4723_senior_design/facial_recognition/haarcascade_eye.xml')

# 2. Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# display settings
CAM_DISPLAY_W, CAM_DISPLAY_H = 640, 480
# uncomment the below once second display is connected
SUN_DISPLAY_W, SUN_DISPLAY_H = 2560, 1600
SECOND_SCREEN_X = 1920
SECOND_SCREEN_Y = 0

# 3. Load your reference image
# Note: Ensure 'your_face_image.jpg' exists in the same folder
#face_reference = cv2.imread('your_face_image.jpg', cv2.IMREAD_GRAYSCALE)

#BOX_SCALE_W = 2.5
#BOX_SCALE_H = 2.0 

print("Camera warming up...")
time.sleep(2)

cv2.namedWindow("camera feed", cv2.WINDOW_NORMAL)
cv2.moveWindow("camera feed", 0, 0)
# uncomment when 2nd display is connected
cv2.namedWindow("Sun Blocker", cv2.WINDOW_NORMAL)
cv2.moveWindow("Sun Blocker", SECOND_SCREEN_X, SECOND_SCREEN_Y) # edit for 2nd display
cv2.setWindowProperty("Sun Blocker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#def draw_box(frame, eye_coords):
 #   if len(eye_coords) == 0:
        #return
    
  #  x1 = min(ex for (ex, ey, ew, eh) in eye_coords)
  #  y1 = min(ey for (ex, ey, ew, eh) in eye_coords)
  #  x2 = min(ex + ew for (ex, ey, ew, eh) in eye_coords)
  #  y2 = min(ey + eh for (ex, ey, ew, eh) in eye_coords)
    
  #  H_PAD = 10 #padding
  #  x1 = max(x1 - H_PAD, 0)
  #  x2 = min(x2 + H_PAD, frame.shape[1])
    
  #  y1 = max(y1 - 5, 0)
  #  y2 = min(y2, frame.shape[0])
    

    #cx = int(np.mean([ex + ew // 2 for (ex, ey, ew, eh) in eye_coords]))
    #cy = int(np.mean([ey + eh // 2 for (ex, ey, ew, eh) in eye_coords]))
    
 #   avg_w = int(np.mean([ew for (_,_, ew, _) in eye_coords]))
  #  avg_h = int(np.mean([eh for (_,_, _, eh) in eye_coords]))
   # 
    #box_w = int(avg_w * BOX_SCALE_W)
    #box_h = int(avg_h * BOX_SCALE_H)
    
    ##x1 = max(cx - box_w // 2, 0) 
    #y1 = max(cy - box_h // 2, 0)
    #x2 = max(cx - box_w // 2, frame.shape[1]) 
    #y2 = max(cy - box_h // 2, frame.shape[0])
    
    #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    
last_box = None

try:
    while True:
        # Capture a frame directly as a NumPy array (OpenCV format)
        frame = picam2.capture_array()
        frame = cv2.flip(frame, -1)
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # white screen for sun blocker
        white_screen = np.ones((SUN_DISPLAY_H, SUN_DISPLAY_W, 3), dtype=np.uint8) * 255
        
        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 8)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around detected faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            roi_gray = gray[y : y+int(h * 0.6), x : x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            print(eyes)
            
            eyes_in_frame = [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]
            #draw_box(frame, eyes_in_frame) 
                
            #for (ex, ey, ew, eh) in eyes:
                #cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
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
                
                # check prompt for code for second display to add here!!!
                scale_x = SUN_DISPLAY_W / CAM_DISPLAY_W
                scale_y = SUN_DISPLAY_H / CAM_DISPLAY_H
                sx1 = int(bx1 * scale_x)
                sy1 = int(by1 * scale_y)
                sx2 = int(bx2 * scale_x)
                sy2 = int(by2 * scale_y)
                cv2.rectangle(white_screen, (SUN_DISPLAY_W - sx2, sy1), (SUN_DISPLAY_W - sx1, sy2), (0, 0, 0), thickness=-1)
            
        # Display the resulting frame
        cv2.imshow("camera feed", frame)
        # for 2nd display:
        cv2.imshow("Sun Blocker", white_screen)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 4. Clean up
    picam2.stop()
    cv2.destroyAllWindows()