from picamera2 import Picamera2 as picam
import cv2 as cv
import numpy as np
from tqdm import tqdm

# Initialize Picamera2
camera = picam()
camera.configure(camera.create_preview_configuration(main={"size": (640, 480)}))
camera.start()

# Initialize variables
face_detect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prompt for user ID
user_id = input("Enter your ID: ")

# Number of images to capture
num_images = 500
captured_images = 0

try:
    # Progress bar for tracking progress
    with tqdm(total=num_images, desc="Capturing Images") as pbar:
        while captured_images < num_images:
            # Capture a frame
            frame = camera.capture_array()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_detect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                captured_images += 1
                # Save face image
                img_path = f'face_recognition/datasets/User.{str(user_id)}.{str(captured_images)}.jpg'
                face_img = gray[y:y+h, x:x+w]
                success = cv.imwrite(img_path, face_img)
                if success:
                    print(f"Image {img_path} saved")
                else:
                    print(f"Image {img_path} NOT saved")

                # Update progress bar
                pbar.update(1)

                if captured_images >= num_images:
                    break

finally:
    camera.stop()
    print("Dataset collection completed!")
