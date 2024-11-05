import argparse
from mnist import Mnist
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST digit recognition")
    parser.add_argument('image_path', type=str, help='Path to the handwritten digit image')
    parser.add_argument('digit', type=int, help='True digit of the image')
    args = parser.parse_args()

    model_path = 'mnist_model.h5'  # Specify your model path
    mnist = Mnist(model_path)

    image_path = args.image_path
    true_digit = args.digit

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    img_resized = cv2.resize(img, (28, 28))
    cv2.imshow("Image",img_resized)
    key = cv2.waitKey(40) & 0xff

    predicted_digit = mnist.predict_digit(img_resized)

    if predicted_digit == true_digit:
        print(f"Success: Image {image_path} for digit {true_digit} is recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image {image_path} for digit {true_digit} but the inference result is {predicted_digit}.")