import cv2
import mediapipe as mp

import time
mpface = mp.solutions.face_mesh
face = mpface.FaceMesh()
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

    result = face.process(imgrgb)
    if result.multi_face_landmarks:
        for facelms in result.multi_face_landmarks:
            h,w,c = img.shape
            lm_list = []
            for id , lm in enumerate(facelms.landmark):

                cx,cy = int(lm.x*w) , int(lm.y*h)
                lm_list.append([id,cx,cy])
                # print(lm_list)
                if id == 1:
                    cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
            mpdraw.draw_landmarks(img,facelms,mpface.FACEMESH_CONTOURS,
                                 mpdraw.DrawingSpec((0,255,0),1,1),
                                  mpdraw.DrawingSpec((0,0,255),1,1)
                                  )
    
    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

