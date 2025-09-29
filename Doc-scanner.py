import cv2
import mediapipe as mp
import time 

ptime =0 
ctime =0 


capture = cv2.VideoCapture(0)

while True:
    success , img = capture.read()

    if not success:
        break
    img = cv2.flip(img,1)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0) 

    
    cv2.imshow("image",img)
    cv2.imshow("blurred_img",blurred_img)
    cv2.imshow("gray_image",gray_image)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
