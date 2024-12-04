import numpy as np
from keras.datasets import imdb
from keras import models, layers, callbacks
import matplotlib.pyplot as plt

class Imdb:
    def __init__(self, num_words=10000):
        self.num_words = num_words
        self.model = None
        self.history = None

    def prepare_data(self):
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=self.num_words)
        self.x_train = self.vectorize_sequences(train_data)
        self.x_test = self.vectorize_sequences(test_data)
        self.y_train = np.asarray(train_labels).astype("float32")
        self.y_test = np.asarray(test_labels).astype("float32")

    def vectorize_sequences(self, sequences):
        results = np.zeros((len(sequences), self.num_words))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.0
        return results

    def build_model(self):
        self.model = models.Sequential([
            layers.Dense(16, activation="relu", input_shape=(self.num_words,)),
            layers.Dropout(0.5),  # Add dropout for regularization
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid")
        ])
        self.model.compile(optimizer="rmsprop",
                           loss="binary_crossentropy",
                           metrics=["accuracy"])

    def train(self, epochs=20, batch_size=512):
        x_val = self.x_train[:10000]
        partial_x_train = self.x_train[10000:]
        y_val = self.y_train[:10000]
        partial_y_train = self.y_train[10000:]

        lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                  factor=0.5, 
                                                  patience=3, 
                                                  verbose=1)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                 patience=5, 
                                                 restore_best_weights=True)

        self.history = self.model.fit(partial_x_train, partial_y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_data=(x_val, y_val),
                                      callbacks=[lr_schedule, early_stopping])


    def plot_metrics(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

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
        return self.model.evaluate(self.x_test, self.y_test)
