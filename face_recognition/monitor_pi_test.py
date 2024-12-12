import cv2 as cv
import os
from picamera2 import Picamera2 as picam

# Set the DISPLAY variable if not set
os.environ["DISPLAY"] = ":0"

# Initialize Picamera2
camera = picam()
camera.configure(camera.create_preview_configuration(main={"size": (640, 480)}))
camera.start()

try:
    # Capture a frame
    frame = camera.capture_array()

    if frame is None:
        print("No valid frame")
    else:
    # Example frame display
        print("Frame found")
        cv.imshow("Example Frame", frame)
        cv.waitKey(0)  # Press any key to close the window
        cv.destroyAllWindows()

finally:
    camera.stop()
    print("Camera stopped and window closed.")
