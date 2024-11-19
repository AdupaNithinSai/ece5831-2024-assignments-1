import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Input
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model
import os

class LeNet:
    def __init__(self, batch_size=32, epochs=20):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self._create_lenet()
        self._compile()
    
    def _create_lenet(self):
        self.model = Sequential([
            Input(shape=(28, 28, 1)),
            Conv2D(filters=6, kernel_size=(5,5), activation='sigmoid', padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),

            Conv2D(filters=16, kernel_size=(5,5), activation='sigmoid', padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),

            Flatten(),

            Dense(120, activation='sigmoid'),
            Dense(84, activation='sigmoid'),
            Dense(10, activation='softmax')
        ])

    def _compile(self):
        if self.model is None:
            print('Error: Create a model first..')
        
        self.model.compile(optimizer='Adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    
    def _preprocess(self):
        # load mnist data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # normalize
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # add channel dim
        self.x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  
        self.x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)  

        # one-hot encoding
        self.y_train = to_categorical(y_train, 10)
        self.y_test = to_categorical(y_test, 10)

    def train(self):
        self._preprocess()
        self.model.fit(self.x_train, self.y_train, 
                       batch_size=self.batch_size, 
                       epochs=self.epochs)

    def save(self, model_path_name):
        if self.model:
            self.model.save(model_path_name)
            print(f'Model saved to {model_path_name}')
        else:
            print('Error: Model is not trained.')

    def load(self, model_path_name):
        self.model = load_model(model_path_name)
        '''if os.path.exists(model_path_name):
            self.model = load_model(model_path_name)
            print(f'Model loaded from {model_path_name}')
        else:
            print(f'Error: Model file {model_path_name} does not exist.')'''

    def predict(self, images):
        if self.model:
            # Ensure input is a 4D tensor
            images = np.array(images).reshape(-1, 28, 28, 1) / 255.0  # Normalize and reshape
            predictions = self.model.predict(images)
            return np.argmax(predictions, axis=1)  # Return predicted labels
        else:
            print('Error: Model is not loaded.')
            return None
