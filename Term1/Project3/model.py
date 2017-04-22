from random import random
import csv
import cv2
import numpy as np
from numpy.random import randint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D

def process_csv(dir):
    with open(dir + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        image_paths = []
        steering_angles = []

        for line in reader:
            source_path_center = line[0]
            source_path_left = line[1]
            source_path_right = line[2]
            filename_center = source_path_center.split('/')[-1]
            filename_left = source_path_left.split('/')[-1]
            filename_right = source_path_right.split('/')[-1]

            # As we are adding the correction for the camera offset we include the images from the 3 cameras 
            # as different training cases without making any distinction between them
            image_paths.append(dir + '/IMG/' + filename_center)
            image_paths.append(dir + '/IMG/' + filename_left)
            image_paths.append(dir + '/IMG/' + filename_right)

            steering_center = float(line[3])

            correction = 0.2
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            steering_angles.append(steering_center)
            steering_angles.append(steering_left)
            steering_angles.append(steering_right)

        return image_paths, steering_angles

def get_random_image(image_paths, steering_angles):
    m = randint(0, len(image_paths))

    # Half of the time we provide a flipped image with it's 
    # corresponding flipped angle
    if random() > 0.5:
        image = cv2.imread(image_paths[m])
        angle = steering_angles[m]
    else:
        image = cv2.flip(cv2.imread(image_paths[m]), 1)
        angle = -1*steering_angles[m]
        
    return image, angle

def get_generator(image_paths, steering_angles, batch_size = 50):
    while True:
        image_paths, steering_angles = shuffle(image_paths, steering_angles)
        images = []
        angles = []
        for idx in range(batch_size):
            image, angle = get_random_image(image_paths, steering_angles)
            images.append(image)
            angles.append(angle)

        yield shuffle(np.array(images), np.array(angles))

def get_validation(image_paths, steering_angles):
    images = []
    angles = []

    for idx, image_path in enumerate(image_paths):
        images.append(cv2.imread(image_path))
        angles.append(steering_angles[idx])

    return shuffle(np.array(images), np.array(angles))

# We start by processing the CSV file
image_paths, steering_angles = process_csv('data_udacity')      

# We split the data and we save 33% for validation
image_paths, image_paths_val, steering_angles, steering_angles_val = train_test_split(image_paths, steering_angles, test_size=0.33)

# Get a generator that will return a batch (default of 50) of images 
# picked randomly from our test cases
generator = get_generator(image_paths, steering_angles)

# The validation set, has a fix number and does not rely on a generator
X_val, Y_val = get_validation(image_paths_val, steering_angles)

epochs = 10

# We finally create the model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')

# Train it
model.fit_generator(generator, samples_per_epoch=20000, nb_epoch=epochs, validation_data=(X_val,Y_val),verbose=1)

# And save it to use it later with drive.py
model.save('model.h5')