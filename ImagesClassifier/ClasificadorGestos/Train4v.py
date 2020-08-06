import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import numpy as np
import random

"""
-------------------------
INSTALACIONES
-------------------------
pip install tensorflow
pip install opencv-python
pip install matplotlib
pip install numpy
pip install tqdm
=========================
"""

def train():
    #try:
        try:
            # get the data
            base_dir = 'files_dependencies/gestures/images'
            train_dir = os.path.join(base_dir, 'train')
            validation_dir = os.path.join(base_dir, 'validation')
        except:
            return ('ERROR: No se encontro la carpeta "train" o "validation" dentro de la carpeta "images"')



        list_train_dir = os.listdir('files_dependencies/gestures/images/train')
        #print(list_train_dir)
        list_validation_dir = os.listdir('files_dependencies/gestures/images/validation')
        #print(list_validation_dir)

        if list_train_dir == list_validation_dir:
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
        else:
            return ('ERROR: Los nombres carpetas dentro de "train" y "validation" deben ser justamente iguales. EJEMPLO: train/Dog   --  validation/Dog')


        labels_file= open('files_dependencies/gestures/labels.txt','r')

        CATEGORIES = []

        CATEGORIES = labels_file.read().split(',')

        class_names = np.array(CATEGORIES)

        training_data = []
        testing_data = []
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




        def create_testing_data():
            u_ = 0
            testing_count = []
            testing_category = []
            for category in CATEGORIES:  # do 
                testing_category.append(category)
                testing_count.append(0)
                path = os.path.join(validation_dir,category)  # create path 
                class_num = CATEGORIES.index(category)  # get the classification  
                for img in tqdm(os.listdir(path)):  # iterate over each image 
                    try:
                        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                        testing_data.append([new_array, class_num])  # add this to our testing_data
                        testing_count[u_] += 1
                    except Exception as e:  # in the interest in keeping the output clean...
                        pass
                    #except OSError as e:
                    #    print("OSErrroBad img most likely", e, os.path.join(path,img))
                    #except Exception as e:
                    #    print("general exception", e, os.path.join(path,img))
                u_ += 1

            for i in range(len(testing_count)):
                if testing_count[i] < 20:
                    print ('CUIDADO: El numero minimo de imagenes por carpeta para entrenar ("validation") deben ser 20. Hay: ' + str(testing_count[i]) + ', en: ' + str(testing_category[i]))

        create_testing_data()


        total_train = len(training_data)
        total_val = len(testing_data)

        random.shuffle(training_data)
        random.shuffle(testing_data)

        BATCH_SIZE = 128
        EPOCHS = 11
        #IMG_SHAPE = IMG_SIZE  # square image

        X = []
        y = []
        for features,label in training_data:
            X.append(features)
            y.append(label)

        X = np.array(X)#.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        #X = X/255.0
        #y = np.array(y)
        OTHER_Y = np.array(y)
        y = tf.keras.utils.to_categorical(y,num_classes=len(class_names))

        X_test = []
        y_test = []
        for features,label in testing_data:
            X_test.append(features)
            y_test.append(label)

        X_test = np.array(X_test)#.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        #X_test = X_test/255.0
        #y_test = np.array(y_test)
        y_test = tf.keras.utils.to_categorical(y_test,num_classes=len(class_names))

        # show the images
        plt.figure(figsize=(10,10))
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(X[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[OTHER_Y[i]])
        #plt.show()
        # reshaping
        train_images = X.reshape(X.shape[0], IMG_SIZE, IMG_SIZE,1)
        test_images = X_test.reshape(X_test.shape[0], IMG_SIZE, IMG_SIZE,1)
        train_labels = np.array(y)
        test_labels = np.array(y_test)
        # define the model
        model = tf.keras.models.Sequential([
            # Note the input shape is the desired size of the image 150x150 with 1 bytes color
            # This is the first convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The second convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The third convolution
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The fourth convolution
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            # 512 neuron hidden layer
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(len(class_names), activation='softmax')
                ])
        

        

        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir = './files_dependencies/gestures/model/logs',
            histogram_freq=0,
            write_graph=True,
            write_grads=True,
            write_images=False,
        )

        callbacks = tensorboard

        model.compile(optimizer='adam', loss="categorical_crossentropy",
                      metrics=['accuracy'])
        # train
        model.fit(train_images, train_labels, validation_split=0.25, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print("Test accuracy: ", test_acc)
        model.save("files_dependencies/gestures/model/model.h5")
        return('Modelo creado exitosamente.')
    #except:
        #return('ERROR: No se ha podido crear el modelo.')

'''labels_file = open('labels.txt', 'r')
file_image = 'test/dog.jpg'
CATEGORIES = []
CATEGORIES = labels_file.read().split(',')
class_names = np.array(CATEGORIES)
IMG_SIZE = 150

def prepare_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    total = new_array.reshape(-1,IMG_SIZE, IMG_SIZE, 1)
    return total

def get_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    return new_array

images = np.array([get_image(file_image)])
plt.figure(figsize=(10,10))
for i in range(1):
    plt.subplot(1,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i], cmap=plt.cm.binary)
plt.show()
#images_reshaped = images.reshape(-1,IMG_SIZE, IMG_SIZE, 1)
X = np.array(prepare_image(file_image))
images_reshaped = tf.cast(prepare_image(file_image), tf.float32)

# model and predictions
model = tf.keras.models.load_model("model.h5")
model.summary()
preds = model.predict(images_reshaped)
print(preds)

def plot_image(prediction, img):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(prediction)
    plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                    100 * np.max(prediction),
                                    ),
               color="blue")


def plot_value_array(prediction):
    plt.xticks(range(4))
    plt.yticks([])
    thisplot = plt.bar(range(4),np.add(prediction), color="#888888")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction)
    thisplot[predicted_label].set_color('blue')


plt.figure(figsize=(8, 12))
for i in range(1):
    # image
    plt.subplot(3, 2, 2 * i + 1)
    plot_image(preds[i], images[i])
    # bar chart
    #plt.subplot(3, 2, 2 * i + 2)
    #plot_value_array(preds[i])
plt.show()'''

print(train())