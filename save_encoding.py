import face_recognition
import cv2
import numpy as np
import os 
import pickle
path_dir = "data"
save_dir = "encodings.pickel"
face_encodings = []
face_names = []
def encode_face():
    for person_name in os.listdir(path_dir):
        person_folder = os.path.join(path_dir,person_name)
        if os.path.isdir(person_folder):
            for file in os.listdir(person_folder):
                if file.lower().endswith(("jpg", "jpeg", "png")):
                    img_path = os.path.join(person_folder, file)
                    img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        face_encodings.append(encodings[0])
                        face_names.append(person_name)
            


    with open(save_dir, "wb") as f:
        pickle.dump((face_encodings, face_names), f)

    print(f"✅ Encodings saved to {save_dir}")
    
