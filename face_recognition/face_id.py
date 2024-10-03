import cv2 as cv

recognizer = cv.face.LBPHFaceRecognizer.create()
recognizer.read('face_recognition/trainer.yml')

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

    video.release()
    cv.destroyAllWindows()

        