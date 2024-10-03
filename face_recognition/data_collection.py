import cv2 as cv

count = 0
video = cv.VideoCapture(0) #0 is default webcam
face_detect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

id = input("Enter your ID:")

while True:
    ret, frame = video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        count = count+1
        img_path = f'face_recognition/datasets/User.{str(id)}.{str(count)}.jpg'
        success = cv.imwrite(img_path, gray[y:y+h, x:x+w])
        if success:
            print(f"Image {img_path} saved")
        else:
            print(f"Image {img_path} NOT saved")

        cv.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

    cv.imshow("Frame", frame)

    quitKey = cv.waitKey(1)
    if count > 500:
        break

    if quitKey == ord('q'):
        break

video.release()
cv.destroyAllWindows()
print("Dataset collection done")