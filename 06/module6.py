import sys
from PIL import Image
import numpy as np
import pickle
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp
import matplotlib.pyplot as plt

def load_image(image_path):
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)  # resize for MNIST compatibility
    img_array = np.asarray(img).astype(np.float32)
    img_array = img_array.flatten()
    img_array = (255.0 - img_array) / 255.0  # normalize
    return img_array

def main():
    if len(sys.argv) != 3:
        print("Usage: python module6.py <image_file> <true_digit>")
        sys.exit(1)

    image_filename = sys.argv[1]
    true_digit = int(sys.argv[2])

    model = TwoLayerNetWithBackProp(input_size=784, hidden_size=50, output_size=10)
    with open('appikonda_mnist_model.pkl', 'rb') as f:
        model.params = pickle.load(f)
    model.update_layers()

    img = load_image(image_filename)
    y_hat = model.predict(img)
    predicted_digit = np.argmax(y_hat)

    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"True label: {true_digit}, Predicted label: {predicted_digit}")
    plt.show()

    if predicted_digit == true_digit:
        print(f"Success: Image {image_filename} is for digit {true_digit} and recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image {image_filename} is for digit {true_digit} but the inference result is {predicted_digit}.")

if __name__ == "__main__":
    main()

    