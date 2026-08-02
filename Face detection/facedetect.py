import cv2 
trainedfacemodel=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
webcam=cv2.VideoCapture(0)
workingcorrectly,video=webcam.read()
cv2.imshow("Web cam",video)
cv2.waitKey(1)