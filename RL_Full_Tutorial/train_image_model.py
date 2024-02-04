'''
training a CNN model on existing images for RL as above but tweaking model

less epochs - 10
smaller convolutional layers

'''
# building custom generator to process image in batches when training
# this avoids loading all images into RAM at once and stalling the process

import os
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from tensorflow import keras

from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Conv2D, Concatenate, Embedding, Reshape, Flatten, Activation, BatchNormalization
from keras.optimizers import SGD
from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator


SOURCE_IMG_HIGHT = 480
SOURCE_IMG_WIDTH = 640

# scaled down image size before cropping
HEIGHT = 240
WIDTH = 320

HEIGHT_REQUIRED_PORTION = 0.5 #bottom share, e.g. 0.1 is take lowest 10% of rows
WIDTH_REQUIRED_PORTION = 0.9

MAX_STEER_DEGREES = 40
# Define the path to your image data directory
data_dir = 'C:/SelfDrive/GPS with Vision/_img'

#image re-size and crop calcs
height_from = int(HEIGHT * (1 -HEIGHT_REQUIRED_PORTION))
width_from = int((WIDTH - WIDTH * WIDTH_REQUIRED_PORTION) / 2)
width_to = width_from + int(WIDTH_REQUIRED_PORTION * WIDTH)

# image size into model
new_height = HEIGHT - height_from
new_width = width_to - width_from

# Define the parameters for image preprocessing and augmentation
batch_size = 64
image_size = (new_width,new_height)


# Define the position of the label in the image file name
label_position = -5  # Assuming the label is the fifth character from the end

# Create a custom data generator
def custom_data_generator(image_files, batch_size):
    num_samples = len(image_files)
    while True:
        indices = np.random.randint(0, num_samples, batch_size)
        batch_images = []
        batch_input_2 = []
        batch_labels = []
        for idx in indices:
            image_path = image_files[idx]
            label = float(os.path.basename(image_path).split('.png')[0].split('_')[2])
            if label > MAX_STEER_DEGREES:
                label = MAX_STEER_DEGREES
            elif label < -MAX_STEER_DEGREES:
                label = -MAX_STEER_DEGREES
            label = float(label)/MAX_STEER_DEGREES
            input_2 = int(os.path.basename(image_path).split('.png')[0].split('_')[1])
            image = preprocess_image(image_path)
            batch_images.append(image)
            batch_input_2.append(input_2)
            batch_labels.append(label)
        yield [np.array(batch_images), np.array(batch_input_2)], np.array(batch_labels)

# Preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # resize to a smaller starting size before cropping
    image = cv2.resize(image, (WIDTH,HEIGHT))
    
    #cropping application to take lower side of the image
    image = image[height_from:,width_from:width_to]
    image = image / 255.0  # Normalize pixel values between 0 and 1
    return image


def create_model():
    # Image input
    image_input = Input(shape=(new_height, new_width, 3))
    # Integer input
    integer_input = Input(shape=(1,))
    # Preprocess the image input
    x = Conv2D(64, kernel_size=(6, 6), activation='relu',padding='same')(image_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(6, 6), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(6, 6), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(6, 6), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dense(8, activation='relu',activity_regularizer=regularizers.L2(1e-5))(x)
    x = Dropout(0.2)(x)
    x = Dense(4, activation='relu',activity_regularizer=regularizers.L2(1e-5))(x)
    x = Flatten()(x)
    # Concatenate image features with integer input
    concatenated_inputs = Concatenate()([x, integer_input])
    # Dense layers for prediction
    output = Dense(1, activation='linear')(concatenated_inputs)
    # Create the model
    model = Model(inputs=[image_input, integer_input], outputs=output)
    return model



# Get a list of image file paths and labels
image_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.png')]

random.shuffle(image_files)

# Split the data into training and validation sets
split_index = int(len(image_files) * 0.8)  # 80% for training, 20% for validation
train_files, val_files = image_files[:split_index], image_files[split_index:]

# Create data generators for training and validation
train_generator = custom_data_generator(train_files, batch_size)
val_generator = custom_data_generator(val_files, batch_size)

model = create_model()
model.summary()
model.compile(loss='MSE',
              optimizer='adam')

# Train the model
model.fit(train_generator, steps_per_epoch=len(train_files) // batch_size, epochs=10,
          validation_data=val_generator, validation_steps=len(val_files) // batch_size)

#This is what actually saves the model to be used in RL training
desired_layer_output  = model.get_layer('dense').output
model_to_save = Model(inputs=model.input, outputs=desired_layer_output)

# Save the new model - make sure you use your RL env to use this name if you use
model_to_save.save('CNN_image_model.h5')