import sys
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from le_net import LeNet

def display_image(image_path):
    # Read image, resize to 28x28, and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))  # Resize to match input size
    img = img / 255.0  # Normalize
    plt.imshow(img, cmap='gray')
    plt.show()
    return img

def main():
    if len(sys.argv) != 3:
        print('Usage: python module8.py <image_filename> <digit_label>')
        sys.exit(1)

    image_filename = sys.argv[1]
    actual_digit = int(sys.argv[2])

    # Initialize LeNet and load the trained model
    lenet = LeNet()
    lenet.load('08/appikonda_cnn_model.keras')  # Ensure you use the correct model path

    # Display input image
    image = display_image(image_filename)

    # Predict the image
    prediction = lenet.predict([image])  # Predict returns a numpy array of labels
    predicted_digit = prediction[0] if prediction is not None else None

    # Compare and print result
    if predicted_digit == actual_digit:
        print(f"Success: Image {image_filename} is for digit {actual_digit} and is recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image {image_filename} is for digit {actual_digit} but the inference result is {predicted_digit}.")

if __name__ == "__main__":
    main()
