import argparse
from mnist1_data import MnistData
import matplotlib.pyplot as plt
import numpy as np
import urllib

def main():
    parser = argparse.ArgumentParser(description='Display MNIST image')
    parser.add_argument('dataset_type', choices=['train', 'test'], help='Specify the dataset type (train/test)')
    parser.add_argument('index', type=int, help='Specify the index of the image to display')
    args = parser.parse_args()

    mnist_data = MnistData()
    (train_images, train_labels), (test_images, test_labels) = mnist_data.load()

    if args.dataset_type == 'train':
        images, labels = train_images, train_labels
    else:
        images, labels = test_images, test_labels

    idx = args.index
    plt.imshow(images[idx].reshape(28, 28))
    plt.title(f"Label: {np.argmax(labels[idx])}")
    plt.show()
    print(f"Label (one-hot): {labels[idx]}")
    print(f"Label: {np.argmax(labels[idx])}")

if __name__ == "__main__":
    print("\nTesting Functionality\n")
    print(" If running from cmd: ")
    print("\tpython <PATH_TO_FILE>/module5-2.py <dataset type> <index>\n")
    
    main()