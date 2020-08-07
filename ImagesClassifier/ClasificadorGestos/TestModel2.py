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
        if confidence < 50:
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
#estoes="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"
#print(decodeIt(estoes))


