import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers
from keras.utils import to_categorical
from keras.datasets import reuters

class Reuters:
    def __init__(self, num_words=10000):
        self.num_words = num_words
        self.prepare_data()

    def prepare_data(self):
        # Load Reuters dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = reuters.load_data(num_words=self.num_words)
        
        # Load the word index
        word_index = reuters.get_word_index()
        
        # Reverse word index to decode words
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        
        # Decode the first newswire to see the actual text
        decoded_newswire = ' '.join([reverse_word_index.get(i-3, '?') for i in self.x_train[0]])

        print(f"Decoded newswire: {decoded_newswire}")
        
        # Vectorize the sequences (turn the sequences of integers into a binary matrix)
        self.x_train = self.vectorize_sequences(self.x_train)
        self.x_test = self.vectorize_sequences(self.x_test)
        
        # One-hot encode the labels
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    def vectorize_sequences(self, sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1
        return results

    def build_model(self):
        # Build the model with 2 hidden layers and softmax output
        self.model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.num_words,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(46, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, epochs=20, batch_size=512):
        # Split the training data into training and validation sets
        x_val = self.x_train[:1000]
        partial_x_train = self.x_train[1000:]
        y_val = self.y_train[:1000]
        partial_y_train = self.y_train[1000:]

        # Train the model
        self.history = self.model.fit(partial_x_train, partial_y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_data=(x_val, y_val))

    def plot_metrics(self):
        # Extract history metrics
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        # Plot accuracy and loss
        plt.figure(figsize=(8, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()

    def evaluate(self):
        # Evaluate the model on the test dataset
        return self.model.evaluate(self.x_test, self.y_test, verbose=1)
