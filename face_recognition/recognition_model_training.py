import cv2 as cv
import numpy as np
import os
import time
import io
import tempfile
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

start_time = time.time()

#Azure Blob config
connection_string = os.getenv("AZURE_BLOB_CONN_STRING")
if not connection_string:
    raise ValueError("AZURE_BLOB_CONN_STRING not found in .env")
container_name = "models"
blob_name = "trainer.yml"

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

dataset_path = 'face_recognition/datasets'
faces, ids = get_images_and_labels(dataset_path)

recognizer.train(faces, np.array(ids))

#Save model to an in-memory buffer, deprecated since OpenCV does not support this
'''
model_buffer = io.BytesIO()
recognizer.write(model_buffer)
model_buffer.seek(0)
'''
#Save model to tempfile
with tempfile.NamedTemporaryFile(delete=False, suffix=".yml") as temp_file:
    model_path = temp_file.name
    recognizer.write(model_path)

#Upload to Azure Blob
try:
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    print(f"Uploading {blob_name} to Azure Blob...")
    with open(model_path, "rb") as model_file:
        blob_client.upload_blob(model_file, overwrite=True)
    print("Model uploaded succesfully!")
except Exception as e:
    print(f"Error uploading to Blob: {e}")
finally:
    os.remove(model_path)

end_time = time.time()
total_time = end_time - start_time

print(f"Model training completed in: {total_time}")
