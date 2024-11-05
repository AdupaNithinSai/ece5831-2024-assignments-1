import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os

class Mnist:
    def __init__(self, model_path):
        """Initialize with the given model path."""
        self.model = tf.keras.models.load_model(model_path)

    def preprocess_image(self, img_resized):
        """Read the image and preprocess for model prediction."""
        img_inverted = np.invert(img_resized) 
        img_normalized = img_inverted.astype('float32') / 255.0
        img_reshaped = np.reshape(img_normalized, (1, 28, 28, 1))
        return img_reshaped
        
    def predict_digit(self, img_resized):
        """Predict the digit from the image path."""      

        preprocessed_image = self.preprocess_image(img_resized)
        prediction = self.model.predict(preprocessed_image)
        predicted_digit = np.argmax(prediction)
        return predicted_digit
    