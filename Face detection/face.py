import cv2 

trainedfacemodel = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0)

while True:
    workingcorrectly, video = webcam.read()

    if not workingcorrectly:
        break

    # Convert to GRAYSCALE (Correct way)
    blacknwhite = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)

    face = trainedfacemodel.detectMultiScale(
        blacknwhite,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in face:
        cv2.rectangle(video, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Web cam", video)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()