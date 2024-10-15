import argparse
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    Loads and preprocesses the image from the given path.

    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - np.ndarray: The preprocessed image ready for model prediction.
    """
    # Open and convert the image to RGB
    image = Image.open(image_path).convert("RGB")

    # Resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into an array for prediction
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    return data, image

def predict_image(model, data, class_names):
    """
    Predicts the class of the given preprocessed image data using the model.

    Parameters:
    - model: The pre-trained Keras model.
    - data (np.ndarray): The preprocessed image data.
    - class_names (list): List of class names corresponding to model output.

    Returns:
    - tuple: The predicted class name and confidence score.
    """
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score

def main():
    """
    The main function to execute the script. Loads the model, processes the input image,
    predicts the class, and displays the image along with the predicted class and confidence score.
    """
    # Setting up argument parser
    parser = argparse.ArgumentParser(description="Rock-Paper-Scissors image classifier")
    parser.add_argument("--image-path", required=True, help="Path to the image to be classified")
    args = parser.parse_args()

    # Load the pre-trained Keras model
    model = load_model("1/model/keras_model.h5", compile=False)

    # Load the labels
    class_names = open("1/model/labels.txt", "r").readlines()

    # Load and preprocess the image
    data, image = load_image(args.image_path)

    # Predict the class of the image
    class_name, confidence_score = predict_image(model, data, class_names)

    # Print prediction and confidence score
    print(f"Class: {class_name}")
    print(f"Confidence Score: {confidence_score:.4f}")

    # Display the image using matplotlib
    plt.imshow(image)
    plt.title(f"Class: {class_name}\nConfidence Score: {confidence_score:.4f}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    print("\nRock-Paper-Scissors Image Classifier\n")
    print(" If running from cmd: ")
    print("\tpython <PATH_TO_FILE>/rock-paper-scissors.py --image-path <PATH_TO_IMAGE>\n")
    main()