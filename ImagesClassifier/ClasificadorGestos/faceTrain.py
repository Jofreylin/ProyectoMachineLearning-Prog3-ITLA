import os
import cv2
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt


def train():
    try:

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        IMAGE_DIR = os.path.join(BASE_DIR,"files_dependencies/faces/images/train")

        face_cascade = cv2.CascadeClassifier('files_dependencies/faces/data/cascades/haarcascade_frontalface_alt2.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()


        current_id = 0
        label_ids = {}
        y_labels = []
        x_train = []

        conteo = 0

        for root, dirs, files in os.walk(IMAGE_DIR):
            
            file = []
            for file in files:
                if file.lower().endswith("png") or file.lower().endswith("jpg") or file.lower().endswith("jpeg"):
                    path = os.path.join(root,file)
                    label = os.path.basename(os.path.dirname(path)).replace(" ","_")
                    
                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id += 1

                    id_ = label_ids[label]
                    #print(label_ids)
                    #y_labels.append(label)
                    #x_train.append(path)
                    image_r = cv2.imread(path) 
                    image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)
                    #pil_image = Image.open(path).convert("L") #grayscale
                    size = (550,550)
                    final_image = cv2.resize(image_r,size)
                    #final_image = pil_image.resize(size, Image.ANTIALIAS)
                    image_array = np.array(final_image)
                    faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.2,minNeighbors=5)
                    for (x,y,w,h) in faces:
                        conteo += 1
                        roi = image_array[y:y+h,x:x+w]
                        #cv2.imwrite('files_dependencies/faces/images/test/my-image-2.png',roi)
                        x_train.append(roi)
                        y_labels.append(id_)


        #print(y_labels)
        #print(x_train)
        print(conteo)
        with open("files_dependencies/faces/data/labels.pickle","wb") as f:
            pickle.dump(label_ids,f)
        print('Pickle creado exitosamente.')

        recognizer.train(x_train, np.array(y_labels))
        recognizer.save("files_dependencies/faces/model/trainner.yml")
        
        return('Rostros registrados exitosamente.')
    except:
        return('ERROR: No se pudo entrenar el modelo.')
    

#print(train())
