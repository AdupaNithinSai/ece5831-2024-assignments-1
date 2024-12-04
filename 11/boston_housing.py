from keras.datasets import boston_housing
from keras import models, layers
import matplotlib.pyplot as plt

class BostonHousing:
    def __init__(self):
        self.model = None
        self.history = None

    def prepare_data(self):
        (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
        self.mean = train_data.mean(axis=0)
        self.std = train_data.std(axis=0)
        self.x_train = (train_data - self.mean) / self.std
        self.x_test = (test_data - self.mean) / self.std
        self.y_train = train_targets
        self.y_test = test_targets

    def build_model(self):
        self.model = models.Sequential([
            layers.Dense(64, activation="relu", input_shape=(self.x_train.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])
        self.model.compile(optimizer="rmsprop",
                           loss="mse",
                           metrics=["mae"])

    def train(self, epochs=20, batch_size=32):
        self.history = self.model.fit(self.x_train, self.y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=0.2)

    def plot_metrics(self):
        mae = self.history.history['mae']
        val_mae = self.history.history['val_mae']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(mae, label='Training MAE')
        plt.plot(val_mae, label='Validation MAE')
        plt.legend(loc='lower right')
        plt.ylabel('MAE')
        plt.title('Training and Validation MAE')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('MSE')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()

    def evaluate(self):
        return self.model.evaluate(self.x_test, self.y_test)
