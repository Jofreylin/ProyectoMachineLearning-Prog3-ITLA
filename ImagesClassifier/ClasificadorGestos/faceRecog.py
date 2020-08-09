import numpy as np
import matplotlib.pyplot as plt
import math
import cv2                     # OpenCV library for computer vision
from PIL import Image
import pickle
from imageio import imread
import io
import base64

# Auxiliary functions

def read_image(path):
    """ Method to read an image from file to matrix """
    image = cv2.imread(path)
    imag = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return imag

def plot_image(image, title=''):
    """ It plots an image as it is in a single column plot """
    # Plot our image using subplots to specify a size and title
    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_subplot(111)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_title(title)
    ax1.imshow(image)

def get_faces(image):
    """
    It returns an array with the detected faces in an image
    Every face is defined as OpenCV does: top-left x, top-left y, width and height.
    """
    # To avoid overwriting
    image_copy = np.copy(image)
    
    # The filter works with grayscale images
    gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

    # Extract the pre-trained face detector from an xml file
    face_classifier = cv2.CascadeClassifier('files_dependencies/faces/data/cascades/haarcascade_frontalface_alt2.xml')
    
    # Detect the faces in image
    faces = face_classifier.detectMultiScale(gray, 1.2, 5)
    
    return faces 

def draw_faces(image, faces=None, plot=False):
    
    
        
    """
    It plots an image with its detected faces. If faces is None, it calculates the faces too
    """
    if faces is None:
        faces = get_faces(image)
    
    # To avoid overwriting
    image_with_faces = np.copy(image)
    
    # Get the bounding box for each detected face
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("files_dependencies/faces/model/trainner.yml")
    gray = cv2.cvtColor(image_with_faces, cv2.COLOR_RGB2GRAY)
    labels = {}

    with open("files_dependencies/faces/data/labels.pickle","rb") as f:
        labels = pickle.load(f)
        labels = {v:k for k,v in labels.items()} #v=value,k=key

    for (x,y,w,h) in faces:
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
            cv2.putText(image_with_faces,name,(x,y-10),font,1,color,stroke,cv2.LINE_AA)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(image_with_faces,'Desconocido',(x,y-10),font,1,color,stroke,cv2.LINE_AA)
            #img_item = "files_dependencies/faces/images/test/my-image.png"
            #cv2.imwrite(img_item,roi_gray)
        color = (255,0,0) #BGR(BLUE-GREEN-RED) 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(image_with_faces,(x,y),(end_cord_x,end_cord_y),color,stroke)
        # Add a red bounding box to the detections image
        #cv2.rectangle(image_with_faces, (x,y), (x+w,y+h), (255,0,0), 3)
        
    if plot is True:
        plot_image(image_with_faces)
    else:
        return image_with_faces

def detect(img):
    image = read_image(img)
    faces = get_faces(image)
    print("Faces detected: {}".format(len(faces)))
    final_image = draw_faces(image,faces)
    cv2_img = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

    filename = "files_dependencies/faces/images/test/test-reconstructed.jpg"
    cv2.imwrite(filename, cv2_img)

    with open(filename, "rb") as fid:
        data = fid.read()

    b64_bytes = base64.b64encode(data)
    b64_string = b64_bytes.decode()
    b64_string = 'data:image/jpeg;base64,'+b64_string

    result_array = []

    rd = "Rostros detectados: {}".format(len(faces))

    result_array.append(b64_string)
    result_array.append(rd)

    
    return(result_array)

def decodeIt(b64_string):
    try:
        #Convierte b64 en un array
        img = imread(io.BytesIO(base64.b64decode(b64_string)))
        #Agrega color a la imagen
        cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #Guarda la imagen
        cv2.imwrite("files_dependencies/faces/images/test/test.jpg", cv2_img)
        #Retorna el reconocimiento de gesto
        return detect('files_dependencies/faces/images/test/test.jpg')
    except:
        return('ERROR: No se pudo decodificar la imagen.')

#detect()