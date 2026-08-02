import cv2

# Load trained face model
trainedfacemodel = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start webcam
webcam = cv2.VideoCapture(0)

while True:
    workingcorrectly, video = webcam.read()

    if not workingcorrectly:
        break

    # Convert to grayscale
    blacknwhite = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face = trainedfacemodel.detectMultiScale(blacknwhite)

    # Draw rectangles
    for (x, y, w, h) in face:
        cv2.rectangle(video, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Show video
    cv2.imshow("Face Detector", video)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()