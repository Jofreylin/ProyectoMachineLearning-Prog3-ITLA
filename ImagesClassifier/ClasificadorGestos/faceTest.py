import numpy as np
import cv2
import pickle

def test():
    try:

        try:
            face_cascade = cv2.CascadeClassifier('files_dependencies/faces/data/cascades/haarcascade_frontalface_alt2.xml')
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read("files_dependencies/faces/model/trainner.yml")
        except:
            return('ERROR: Es posible que no haya rostros registrados.')

        labels = {}
        with open("files_dependencies/faces/data/labels.pickle","rb") as f:
            labels = pickle.load(f)
            labels = {v:k for k,v in labels.items()} #v=value,k=key

        cap = cv2.VideoCapture(0)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=5)
            
            for(x,y,w,h) in faces:
                #print(x,y,w,h)
                roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
                
                # recognize - deep learned model
                id_, conf = recognizer.predict(roi_gray)
                if conf>=30: #and conf<=85:
                    #print(id_)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = labels[id_]
                    color = (255,255,255)
                    stroke = 2
                    cv2.putText(frame,name,(x,y-10),font,1,color,stroke,cv2.LINE_AA)
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = labels[id_]
                    color = (255,255,255)
                    stroke = 2
                    cv2.putText(frame,'Desconocido',(x,y-10),font,1,color,stroke,cv2.LINE_AA)
                    #img_item = "files_dependencies/faces/images/test/my-image.png"
                    #cv2.imwrite(img_item,roi_gray)
                color = (255,0,0) #BGR(BLUE-GREEN-RED) 0-255
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)

            cv2.putText(frame, "Presione 'q' para salir",(5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            # Display the resulting frame
            cv2.imshow('WebCam',frame)
            #cv2.imshow('gray',gray)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        return('Proceso terminado.')
    except:
        return('ERROR: No se puede inicializar el modulo para detectar Rostros.')

#test()