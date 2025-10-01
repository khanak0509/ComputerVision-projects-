import cv2
import face_recognition
import time 
import os 
import pickle
from save_encoding import encode_face
name = "Unknown"

ptime =0 
ctime = 0
if not os.path.exists("encodings.pickle"):
    encode_face()
with open("encodings.pickel", "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)


capture = cv2.VideoCapture(0)

while True:
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    success ,img = capture.read()
    if  not success:
        break

    img = cv2.flip(img,1)
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    face_location = face_recognition.face_locations(imgrgb)
    face_encodings = face_recognition.face_encodings(imgrgb,face_location)
    face_name = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_name.append(name)
        for (top, right, bottom, left) in (face_location):
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

        for (top, right, bottom, left), name in zip(face_location, face_name):
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)




    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()



    
