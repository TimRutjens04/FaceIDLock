import cv2 as cv
from azure.storage.blob import BlobServiceClient
from picamera2 import Picamera2 as picam
import signal, sys
import os
from lock_mechanism.unlock_on_recognition import unlock_door, lock_door

#.env var init
os.environ["DISPLAY"] = ":0"

# Azure Blob config
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
        print(f"Model downloaded successfully to {local_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

#Define signal handler:
def signal_handler(sig, frame):
    print("Interruption received, quitting application...")
    camera.stop()
    cv.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Download the model
model_local_path = "trainer.yml"
download_model_from_azure(model_local_path)

# Initialize the recognizer
recognizer = cv.face.LBPHFaceRecognizer.create()
recognizer.read(model_local_path)

# Load the Haar cascade for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize Picamera2
camera = picam()
camera.configure(camera.create_preview_configuration(main={"size": (640, 480)}))
camera.start()

count = 0
try:
    while True:
        # Capture a frame
        frame = camera.capture_array()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # print("Frame captured")

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            # Perform recognition
            id, confidence = recognizer.predict(face)

            if confidence < 50:
                text = f"ID: {id}, Confidence: {round(100 - confidence, 2)}%"
                print(f"ID: {id}, Confidence: {round(100 - confidence, 2)}%")
                count = count+1
                if count >= 5:
                    unlock_door()
                    count = 0
            else:
                text = "Unknown user"

            # Draw rectangle and add text
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(frame, text, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2)

        # Display the frame
        # print(f"Frame captured and processed... {frame}")
        # cv.imshow("Face Recognition", frame)

        #Deprecated since keyboard requires root access, might not be most secure.
        # Break the loop on 'q' key press
        # if keyboard.is_pressed('q'):
        #     print("Exiting program...")
        #     break
except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    lock_door()
    camera.stop()
    cv.destroyAllWindows()
    print("Face recognition terminated.")
