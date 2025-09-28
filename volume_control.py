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
    if lm_list:

        coord1 = lm_list[4]
        coord2 = lm_list[8]
        x1, y1 = coord1[1], coord1[2]
        x2, y2 = coord2[1], coord2[2]

        distance = math.hypot(x2 - x1, y2 - y1)
        print("Distance:", distance)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  
        cv2.circle(img, (x1, y1), 8, (0, 255, 0), cv2.FILLED)  
        cv2.circle(img, (x2, y2), 8, (0, 255, 0), cv2.FILLED) 
        volume = np.interp(distance, [min_dist, max_dist], [0, 100])
        pyvolume.custom(percent=int(volume))
        cv2.putText(img, f'Volume: {int(volume)} %', (50,100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.rectangle(img, (50,150), (85,400), (0,255,0), 3)
        fill = np.interp(distance, [min_dist, max_dist], [400, 150])
        cv2.rectangle(img, (50,int(fill)), (85,400), (0,255,0), cv2.FILLED)


    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()



    
