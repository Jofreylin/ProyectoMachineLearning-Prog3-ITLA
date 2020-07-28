import cv2
import tensorflow as tf
"""
-------------------------
INSTALACIONES
-------------------------
pip install tensorflow
pip install opencv-python
=========================
"""

labels_file= open('labels.txt','r')

CATEGORIES = []

CATEGORIES = labels_file.read().split(',');

def prepare(filepath):
    IMG_SIZE = 150
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare('test/gato2.jpg')])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])