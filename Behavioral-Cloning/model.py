import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Dropout

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
datapath = './'
csvpath = datapath + 'driving_log.csv'

with open(csvpath) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# load images
images = []
measurements = []
for line in lines:
    source_path = line[0]  #front image
    filename = source_path.split('/')[-1]
    current_path = datapath + 'IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])  #steering
    measurements.append(measurement)

# data augmentation
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# CNN model
# https://devblogs.nvidia.com/deep-learning-self-driving-cars/
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0-0.5))

model.add(Conv2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16,5,5, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, \
          shuffle=True, nb_epoch=7, verbose=1)

model.save('model.h5')
