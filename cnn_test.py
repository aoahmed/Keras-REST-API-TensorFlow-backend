import keras
from keras.utils import to_categorical
from keras.models import load_model


import numpy as np
import json 
import time       
import sys
import cv2
import os
              


# Arguments
# Name of the owner
OWNER = sys.argv[1]
# ID of the Neural Network
ID_NEURAL_NETWORK = sys.argv[2]
# Test Image Path
TEST_IMAGE_PATH = sys.argv[3]
# Number of classes (Number of Persons in Database)
NB_CLASSES = sys.argv[4]

# Image Size
IMG_SIZE = 50


MODEL_NAME = OWNER + "." + ID_NEURAL_NETWORK 
# just so we remember which saved model is which, sizes must match

# Load the dictionary of labels
labels_dict = np.load('labels_dict.npy').item()
print(labels_dict)
NB_CLASSES = len(labels_dict)


def process_test_data():
    training_data = []
    X = []
    Y = []
    num = os.path.basename(TEST_IMAGE_PATH).split('-')[0]
    img = cv2.imread(TEST_IMAGE_PATH,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    X.append(np.array(img))
    Y.append(labels_dict[num])
    return X,Y

# Creating test data
test_X,test_Y = process_test_data()

test_X = np.array(test_X).reshape(-1, IMG_SIZE,IMG_SIZE, 1)
test_X = test_X.astype('float32')
test_X = test_X / 255.


# Change the labels from categorical to one-hot encoding
test_Y_one_hot = to_categorical(test_Y,num_classes=NB_CLASSES)

model = load_model(MODEL_NAME+".h5")

predictions = model.predict(test_X)
predicted_classe = np.argmax(np.round(predictions),axis=1)


# Create JSON output
prediction = {}
prediction['prediction'] = labels_dict.keys()[labels_dict.values().index(predicted_classe)]
prediction['probability'] = float(max(predictions[0]))

json_data = json.dumps(prediction)

print(json_data)
