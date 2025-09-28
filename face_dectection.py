import cv2
import mediapipe as mp
import time

mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection()
mp_draw = mp.solutions.drawing_utils
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
    if result.detections:
        for id , detection in enumerate(result.detections):
            print(detection)
            print(detection.score)
            h,w,c = img.shape
            bboxc = detection.location_data.relative_bounding_box

            bbox = int(bboxc.xmin*w) , int(bboxc.ymin*h) , int(bboxc.width*w) , int(bboxc.height*h)
            
            cv2.rectangle(img,bbox,(255,0,255),2)

            cv2.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    
    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break