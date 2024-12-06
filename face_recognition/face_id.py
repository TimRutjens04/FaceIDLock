import cv2 as cv
from azure.storage.blob import BlobServiceClient
import time
import os

#Azure Blob config
connection_string = os.getenv("AZURE_BLOB_CONN_STRING")
if not connection_string:
    raise ValueError("AZURE_BLOB_CONN_STRING not found in .env")
container_name = "models"
blob_name = "trainer.yml"

def download_model_from_azure(local_path):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    try:
        print(f"Downloading {blob_name} from Azure Blob Storage...")
        with open(local_path, "wb") as file:
            data = blob_client.download_blob()
            file.write(data.readall())
        print(f"Model downloaded succesfully to {local_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

model_local_path = "trainer.yml"
download_model_from_azure(model_local_path)

recognizer = cv.face.LBPHFaceRecognizer.create()
recognizer.read(model_local_path)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

video = cv.VideoCapture(0)

while True:
    ret, frame = video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        id, confidence = recognizer.predict(face)

        if confidence < 50:
            text = f"ID: {id}, Confidence: {round(100 - confidence, 2)}%"
        else:
            text = "Unknown user"

        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(frame, text, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.9, (255,255,255), 2)

        cv.imshow("Face Recognition", frame)

        quitKey = cv.waitKey(1)
        if quitKey == ord('q'):
            break

        