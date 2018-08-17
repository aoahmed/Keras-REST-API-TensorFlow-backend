import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split

import numpy as np
import json
import cv2
import sys
import os

# Arguments
# Name of the owner
OWNER = sys.argv[1]
# ID of the Neural convnet
ID_NEURAL_convnet = sys.argv[2]
# Name of the training file 
TRAIN_DIR = sys.argv[3]

# Image Size
IMG_SIZE = 50
# Number of classes (Number of Persons in Database)
NB_CLASSES = 0

MODEL_NAME = OWNER + "." + ID_NEURAL_convnet 
# just so we remember which saved model is which, sizes must match


def label_vocabulary():
    labels_dict = {}
    i = 0
    for img in os.listdir(TRAIN_DIR):
        label = img.split('-')[0]
        if label not in labels_dict:
            labels_dict[label] = i
            i=i+1
    return labels_dict

def create_train_data():
    X = []
    Y = []
    for img in os.listdir(TRAIN_DIR):
    	word_label = img.split('-')[0]
        label = labels_dict[word_label]
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        X.append(np.array(img))
        Y.append(label)
    return X,Y


# Creating the dictionary of labels
labels_dict = label_vocabulary()
np.save('labels_dict.npy', labels_dict)

NB_CLASSES = len(labels_dict)


# Creating train data
train_X,train_Y = create_train_data()

train_X = np.array(train_X).reshape(-1, IMG_SIZE,IMG_SIZE, 1)
train_X = train_X.astype('float32')
train_X = train_X / 255.


# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)

# Split the training set in validation and training data 
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, 
														   test_size=0.2, random_state=13)


# Building the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(IMG_SIZE,IMG_SIZE,1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))           
model.add(Dropout(0.3))
model.add(Dense(NB_CLASSES, activation='softmax'))



#model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, 
					  optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

model_dropout = model.fit(train_X, train_label, batch_size=64,
										  epochs=40,verbose=1,validation_data=(valid_X, valid_label))

model.save(MODEL_NAME+".h5")

accuracy = model_dropout.history['acc']

# Create JSON output
result = {}
result['accuracy'] = float(accuracy[len(accuracy)-1])
result['nbClasses'] = NB_CLASSES

json_data = json.dumps(result)

print(json_data)
