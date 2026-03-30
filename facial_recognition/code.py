import cv2
from picamera2 import Picamera2
import time
import numpy as np

# 1. Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier('/home/ece4723inter/Documents/faceRecognition/haarcascade_frontalface_default.xml')

# 2. Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# 3. Load your reference image
# Note: Ensure 'your_face_image.jpg' exists in the same folder
face_reference = cv2.imread('your_face_image.jpg', cv2.IMREAD_GRAYSCALE)

print("Camera warming up...")
time.sleep(2)

try:
    while True:
        # Capture a frame directly as a NumPy array (OpenCV format)
        frame = picam2.capture_array()
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Draw rectangle around detected faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Compare detected face with reference (Simple Template Matching)
            if face_reference is not None:
                roi_gray = gray[y:y+h, x:x+w]
                # Resize reference to match detected face size for comparison
                ref_resized = cv2.resize(face_reference, (w, h))
                
                res = cv2.matchTemplate(roi_gray, ref_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                
                if max_val >= 0.8:
                    cv2.putText(frame, "Match Found", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow("Face Recognition", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 4. Clean up
    picam2.stop()
    cv2.destroyAllWindows()