import cv2 
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
ctime = 0 
ptime = 0
capture = cv2.VideoCapture(0)
while True:
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime =ctime 

    success, img = capture.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    result = pose.process(imgrgb)
    lm_list = []
    if result.pose_landmarks:

        for id , lm in enumerate(result.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy = int(lm.x*w) , int(lm.y*h)
            lm_list.append([id,cx,cy])
            print(lm_list)
            if id == 0:  # nose
                cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break