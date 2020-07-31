# Ignore  the warnings
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from random import shuffle
import os
import cv2
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Activation
from PIL import Image
from zipfile import ZipFile
from tqdm import tqdm
import random as rn
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import random
#import seaborn as sns
from matplotlib import style
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation

# configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
#%matplotlib inline
#style.use('fivethirtyeight')
#sns.set(style='whitegrid', color_codes=True)

# model selection

# preprocess.

# dl libraraies

# specifically for cnn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.

base_dir = 'files_dependencies/images'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

list_train_dir = os.listdir('files_dependencies/images/train')
        #print(list_train_dir)
list_validation_dir = os.listdir('files_dependencies/images/validation')

labels_file = open('files_dependencies/labels.txt', 'r')

CATEGORIES = []

CATEGORIES = labels_file.read().split(',')

class_names = np.array(CATEGORIES)

training_data = []
testing_data = []
IMG_SIZE = 100

def create_training_data():
        u_ = 0
        train_count = []
        train_category = []
        for category in CATEGORIES:  # do dogs and cats
            train_category.append(category)
            train_count.append(0)
            # create path to dogs and cats
            path = os.path.join(train_dir, category)
            # get the classification  (0 or a 1). 0=dog 1=cat
            class_num = CATEGORIES.index(category)
            # iterate over each image per dogs and cats
            for img in tqdm(os.listdir(path)):
                try:
                    img_array = cv2.imread(os.path.join(
                        path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                    # resize to normalize data size
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    # add this to our training_data
                    training_data.append([new_array, class_num])
                    train_count[u_] += 1
                except Exception as e:  # in the interest in keeping the output clean...
                    pass
                # except OSError as e:
                #    print("OSErrroBad img most likely", e, os.path.join(path,img))
                # except Exception as e:
                #    print("general exception", e, os.path.join(path,img))
            u_ += 1
        for i in range(len(train_count)):
            if train_count[i] < 100:
                print('CUIDADO: El numero minimo de imagenes por carpeta para entrenar ("train") deben ser 100. Hay: ' +
                      str(train_count[i]) + ', en: ' + str(train_category[i]))

create_training_data()

def create_testing_data():
        u_ = 0
        testing_count = []
        testing_category = []
        for category in CATEGORIES:  # do dogs and cats
            testing_category.append(category)
            testing_count.append(0)
            # create path to dogs and cats
            path = os.path.join(validation_dir, category)
            # get the classification  (0 or a 1). 0=dog 1=cat
            class_num = CATEGORIES.index(category)
            # iterate over each image per dogs and cats
            for img in tqdm(os.listdir(path)):
                try:
                    img_array = cv2.imread(os.path.join(
                        path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                    # resize to normalize data size
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    # add this to our training_data
                    testing_data.append([new_array, class_num])
                    testing_count[u_] += 1
                except Exception as e:  # in the interest in keeping the output clean...
                    pass
                # except OSError as e:
                #    print("OSErrroBad img most likely", e, os.path.join(path,img))
                # except Exception as e:
                #    print("general exception", e, os.path.join(path,img))
            u_ += 1

        for i in range(len(testing_count)):
            if testing_count[i] < 20:
                print('CUIDADO: El numero minimo de imagenes por carpeta para entrenar ("validation") deben ser 20. Hay: ' +
                      str(testing_count[i]) + ', en: ' + str(testing_category[i]))

create_testing_data()

total_train = len(training_data)
total_val = len(testing_data)

random.shuffle(training_data)
random.shuffle(testing_data)

#BATCH_SIZE = 32
IMG_SHAPE = 150  # square image

X = []
y = []
for features, label in training_data:
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0], IMG_SIZE, IMG_SIZE, 1)
X = X/255.0

X_test = []
y_test = []
for features, label in testing_data:
        X_test.append(features)
        y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

X_test = X_test.reshape(X_test.shape[0], IMG_SIZE, IMG_SIZE, 1)
X_test = X_test/255.0

model = Sequential()
model.add(Conv2D(filters=32, kernel_size = (5, 5), padding = 'Same', activation ='relu', input_shape = (IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(filters=64, kernel_size=(3, 3), padding = 'Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(Conv2D(filters=96, kernel_size=(3, 3), padding= 'Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=96, kernel_size=(3, 3), padding = 'Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(10, activation="softmax"))

batch_size=128
epochs=10

checkpoint = ModelCheckpoint(
    './base.model',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=30,
    verbose=1,
    mode='auto'
)
tensorboard = TensorBoard(
    log_dir = './logs',
    histogram_freq=0,
    batch_size=16,
    write_graph=True,
    write_grads=True,
    write_images=False,
)

csvlogger = CSVLogger(
    filename= "training_csv.log",
    separator = ",",
    append = False
)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1, 
    mode='auto'
)

callbacks = [checkpoint,tensorboard,csvlogger,reduce]

model.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

History = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

model.save("files_dependencies/model/model.h5")

