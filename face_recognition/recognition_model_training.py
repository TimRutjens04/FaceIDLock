import cv2 as cv
import numpy as np
import os
import time

start_time = time.time()

dataset_path = 'face_recognition/datasets'

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognizer = cv.face.LBPHFaceRecognizer.create()

def get_images_and_labels(dataset_path): 
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = face_cascade.detectMultiScale(img)

        for (x,y,w,h) in faces:
            face_samples.append(img[y:y+h, x:x+w])
            ids.append(id)

    return face_samples, ids

faces, ids = get_images_and_labels(dataset_path)

recognizer.train(faces, np.array(ids))

model_path = 'face_recognition/trainer.yml'
recognizer.write(model_path)

end_time = time.time()
total_time = end_time - start_time

print(f"Model training completed in: {total_time}")
