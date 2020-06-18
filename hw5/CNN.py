import tensorflow as tf
from tensorflow import keras
# Import Keras libraries and packages
from keras.models import Sequential 
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks

# fitting the cNN to images
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
# import PIL for reading img
import PIL

def init_CNN():
    # initializing CNN
    model = Sequential()  
    model.add(Conv2D(32, 3, 3, input_shape = (128, 128, 1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Second convolutional layer
    model.add(Conv2D(32, 3, 3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # # Third convolutional layer
    # model.add(Conv2D(64, 3, 3, activation = 'relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # output is same as our category
    model.add(Dense(output_dim = 15, activation = 'relu'))

    # Use Adam Algorithm
    model.compile(optimizer = 'adam', loss="binary_crossentropy", metrics = ['accuracy'])

    return model



def train_CNN(model):
    # load datasets with ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    # resize to 128 size all image need to be the same size, 128 is also enough
    training_set = train_datagen.flow_from_directory('hw5_data/train/', color_mode = "grayscale",target_size = (128, 128))
    test_set = test_datagen.flow_from_directory('hw5_data/test/', color_mode = "grayscale", target_size = (128, 128))
    # epochs 10 is good enough
    model.fit(training_set, epochs = 10, validation_data = test_set)
    return model,test_set


def test_CNN(model,test_set):
    # evaluate the test_set of the model
    score = model.evaluate(test_set, verbose=0)
    return score


if __name__ == "__main__":
    #  initialize CNN
    model = init_CNN()
    #  train the CNN
    model, test_set = train_CNN(model)
    #  test the images
    score = test_CNN(model,test_set)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])