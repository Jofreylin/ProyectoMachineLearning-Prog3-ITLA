import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image # For handling the images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Plotting
import random
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from tqdm import tqdm
import cv2
import tensorflow as tf

def train():
    #try:
        try:
            # get the data
            base_dir = 'files_dependencies/gestures/images'
            train_dir = os.path.join(base_dir, 'train')
        except:
            return ('ERROR: No se encontro la carpeta "train" dentro de la carpeta "images"')



        list_train_dir = os.listdir('files_dependencies/gestures/images/train')
        #print(list_train_dir)

        
        labels_names = ''
        for x in range(len(list_train_dir)):
            if x == 0:
                labels_names = list_train_dir[x]
            else:
                labels_names = labels_names + ',' + list_train_dir[x]
        labels_file = open('files_dependencies/gestures/labels.txt', 'w')
        labels_file.write(labels_names)
        labels_file.close()
        labels_file = open('files_dependencies/gestures/labels.txt', 'r')
        print('Labels creados: ' + str(labels_file.read().split(',')))

        labels_file = open('files_dependencies/gestures/labels.txt', 'r')
        CATEGORIES = []

        CATEGORIES = labels_file.read().split(',')

        class_names = np.array(CATEGORIES)

        

        training_data = []
        IMG_SIZE = 150



        def create_training_data():
            u_ = 0
            train_count = []
            train_category = []
            for category in CATEGORIES:  # do 
                train_category.append(category)
                train_count.append(0)
                path = os.path.join(train_dir,category)  # create path 
                class_num = CATEGORIES.index(category)  # get the classification  
                for img in tqdm(os.listdir(path)):  # iterate over each image 
                    try:
                        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                        training_data.append([new_array, class_num])  # add this to our training_data
                        train_count[u_] += 1
                    except Exception as e:  # in the interest in keeping the output clean...
                        pass
                    #except OSError as e:
                    #    print("OSErrroBad img most likely", e, os.path.join(path,img))
                    #except Exception as e:
                    #    print("general exception", e, os.path.join(path,img))
                u_ += 1
            for i in range(len(train_count)):
                if train_count[i] < 100:
                    print ('CUIDADO: El numero minimo de imagenes por carpeta para entrenar ("train") deben ser 100. Hay: ' + str(train_count[i]) + ', en: ' + str(train_category[i]))

        create_training_data()

        total_train = len(training_data)
        random.shuffle(training_data)

        x_data = []
        y_data = []
        datacount = total_train 
        for features,label in training_data:
            x_data.append(features)
            y_data.append(label)

        x_data = np.array(x_data)
        
        y_data = to_categorical(y_data,len(class_names))

        x_data = x_data.reshape((datacount, IMG_SIZE, IMG_SIZE, 1))
        x_data = x_data/255
        x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
        x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)

        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                            input_shape=(IMG_SIZE, 
                                                                        IMG_SIZE,
                                                                        1)),
                layers.experimental.preprocessing.RandomRotation(0.1),
                layers.experimental.preprocessing.RandomZoom(0.1),
            ]
        )

        def seeImageAugmented():
            plt.figure(figsize=(10,10))
            for i in range(9):
                augmented_image = data_augmentation(x_train)
                plt.subplot(3,3,i+1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(augmented_image[0], cmap=plt.cm.binary) 
            plt.show()
        
        #seeImageAugmented()

        model = tf.keras.models.Sequential([
            # Note the input shape is the desired size of the image 150x150 with 3 bytes color
            # This is the first convolution
            data_augmentation,
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            #,input_shape=(IMG_SIZE,IMG_SIZE,1)
            tf.keras.layers.MaxPooling2D(2, 2),
            # The second convolution
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            # The third convolution
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            # The fourth convolution
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            # 512 neuron hidden layer
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(len(class_names), activation='softmax')
        ])
        '''model=models.Sequential()
        model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE,1))) 
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(len(class_names), activation='softmax'))'''

        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=30, batch_size=128, verbose=1, validation_data=(x_validate, y_validate))

        [loss, acc] = model.evaluate(x_test,y_test,verbose=1)
        print("Accuracy:" + str(acc))

        model.save("files_dependencies/gestures/model/model9.h5")

train()