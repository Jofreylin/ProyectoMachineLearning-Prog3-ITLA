import cv2
import tensorflow as tf
import numpy as np
import os
import base64
import io
from imageio import imread

"""
-------------------------
INSTALACIONES
-------------------------
pip install tensorflow
pip install opencv-python
pip install imageio
pip install numpy
=========================
"""
def test(image_file):
    try:
        try:
            labels_file = open('files_dependencies/gestures/labels.txt', 'r')
        except:
            return ('No se ha podido encontrar el archivo labels.txt (se crea al entrenar el modelo)')

        CATEGORIES = []
        CATEGORIES = labels_file.read().split(',')
        class_names = np.array(CATEGORIES)

        IMG_SIZE = 150

        def prepare(filepath):
            img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


        model = tf.keras.models.load_model("files_dependencies/gestures/model/model.h5")

        #model.summary()
        #print(class_names)
        images_reshaped = tf.cast(prepare(image_file), tf.float32)
        prediction = model.predict(images_reshaped)

        predicted_label = np.argmax(prediction[0])
        confidence = 100 * np.max(prediction[0])
        if confidence < 80:
            return ('No se ha reconocido la imagen.')
        else:
            return ("{} {:2.0f}%".format(class_names[predicted_label],100 * np.max(prediction[0])))
    except:
        return ('ERROR: No se ha podido realizar el analisis.')


def decodeIt(b64_string):
    try:
        #Convierte b64 en un array
        img = imread(io.BytesIO(base64.b64decode(b64_string)))
        #Agrega color a la imagen
        cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #Guarda la imagen
        cv2.imwrite("files_dependencies/gestures/test/test.jpg", cv2_img)
        #Retorna el reconocimiento de gesto
        return test('files_dependencies/gestures/test/test.jpg')
    except:
        return('No se pudo decodificar la imagen.')


#print(test('files_dependencies/gestures/test/test.jpg'))

#print(decodeIt(estoes))


