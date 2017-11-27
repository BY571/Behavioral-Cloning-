import csv
import cv2
import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt

lines = []
with open("/home/sebastian/Schreibtisch/data_mixed/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
batch_size = 64
steering_correction = 0.2

def generator(lines, batch_size):
    num_samples = len(lines)
    while 1: 
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]
            
            center_images = []
            right_images = []
            left_images = []
                        
            steering_angle_c = []
            steering_angle_r = []
            steering_angle_l = []
            
            
            for batch_sample in batch_samples:
                # loading the 3 images 
                center_name = "/home/sebastian/Schreibtisch/data_mixed/IMG/" +batch_sample[0].split("/")[-1]
                right_name = "/home/sebastian/Schreibtisch/data_mixed/IMG/" +batch_sample[2].split("/")[-1]
                left_name = "/home/sebastian/Schreibtisch/data_mixed/IMG/" +batch_sample[1].split("/")[-1]
                # reading and color correction of the images 
                center_images.append(cv2.cvtColor(cv2.imread(center_name), cv2.COLOR_BGR2RGB))
                right_images.append(cv2.cvtColor(cv2.imread(right_name), cv2.COLOR_BGR2RGB))
                left_images.append(cv2.cvtColor(cv2.imread(left_name), cv2.COLOR_BGR2RGB))
                
                steering_angle_c.append(float(batch_sample[3]))
                #adding steering correction of 0.2
                steering_angle_r.append(float(batch_sample[3]) - steering_correction)
                steering_angle_l.append(float(batch_sample[3]) + steering_correction)
                
                images = center_images + right_images + left_images
                angles = steering_angle_c + steering_angle_r + steering_angle_l

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train) 

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)
# Image format
ch, row, col = 3, 160, 320  
        ### CNN ###
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,20),(0,0)))) #entfernt die oberen 70 pixel und die unteren 20. l/r bleibt.
model.add(Convolution2D(24,5,strides = (2,2), activation = "relu"))
model.add(Convolution2D(36,5,strides = (2,2), activation = "relu"))
model.add(Convolution2D(48,5,strides = (2,2), activation = "relu"))
model.add(Convolution2D(64,3, activation = "relu"))
model.add(Convolution2D(64,3, activation = "relu"))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dropout(0.4))
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history=model.fit_generator(generator = train_generator,steps_per_epoch=(len(train_samples)/batch_size),epochs=10,verbose=1,validation_data=validation_generator, validation_steps=(len(validation_samples)/batch_size))
# saving the model 
name = "model_sedi_mixedv2.h5"
model.save(name)
### print the keys contained in the history object
print(history.history.keys())

### plot the training and validation loss for each epoch
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.plot(history.history['loss'])
axes.plot(history.history['val_loss'])
axes.set_title('model mean squared error loss')
axes.set_ylabel('mean squared error loss')
axes.set_xlabel('epoch')
axes.legend(['training set', 'validation set'], loc='upper right')
plt.show()
fig.savefig(name + ".jpg")