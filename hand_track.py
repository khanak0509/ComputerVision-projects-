import cv2
import mediapipe as mp
import time 

mphand = mp.solutions.hands
hands = mphand.Hands()
mpdraw = mp.solutions.drawing_utils
ptime = 0 
ctime = 0 
capture = cv2.VideoCapture(0)
while True:
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    success , img = capture.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    lm_list = []
    result = hands.process(imgrgb)
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            for id , lm in enumerate(handlms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w) , int(lm.y*h)
                lm_list.append([id,cx,cy])
                if id == 0:
                    cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
            mpdraw.draw_landmarks(img,handlms,mphand.HAND_CONNECTIONS)
        print(lm_list)  

    

    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
