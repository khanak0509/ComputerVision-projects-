import cv2
import mediapipe as mp
import math
import time 
import numpy as np
import pyvolume
ptime = 0
ctime = 0 
capture = cv2.VideoCapture(0)
min_dist = 14 
max_dist = 400
mphand = mp.solutions.hands
hands = mphand.Hands()
mpdrow = mp.solutions.drawing_utils

while True:
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime 

    success , img = capture.read()
    if not success:
        break
    img = cv2.flip(img,1)
    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    lm_list = []
    result = hands.process(imgrgb)

    if result.multi_hand_landmarks:
        for hndlms in result.multi_hand_landmarks:
            for id , lm in enumerate(hndlms.landmark):
                h,w,c = img.shape
                cx , cy = int(lm.x*w) , int(lm.y*h)
                lm_list.append([id,cx,cy])
    finger = 0 
    thumb = 0 

    if lm_list:
        
        if lm_list[4][1]<lm_list[2][1]:  
            thumb = 1 
        if lm_list[8][2]<lm_list[6][2]:
            finger +=1
        if lm_list[12][2]<lm_list[10][2]:
            finger +=1
        if lm_list[16][2]<lm_list[14][2]:
            finger +=1
        if lm_list[20][2]<lm_list[18][2]:
            finger +=1

        cv2.putText(img,f"thumb = {thumb}, finger = {finger}",(100,100),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        




    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()



    
