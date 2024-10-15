import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

def load_labels(label_file):
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def main():
    # Load the model
    model = tf.keras.models.load_model('1/model/keras_model.h5', compile=False)

    # Load the class names
    class_names = load_labels('1/model/labels.txt')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess the image for model prediction (resize and normalize)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.fromarray(frame)

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image.convert("RGB"), size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predict the class
        predictions = model.predict(data)
        class_idx = np.argmax(predictions)
        prediction_label = class_names[class_idx]
        confidence_score = predictions[0][class_idx]

        # Display the resulting frame with prediction
        cv2.putText(frame, f'Prediction: {prediction_label} ({confidence_score:.4f})', (10, 30), 
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Rock Paper Scissors', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\nRock-Paper-Scissors Image Classifier")
    print("   Make  sure to keep your camera unblocked for this.\n")
    print(" If running from cmd: ")
    print("\tpython <PATH_TO_FILE>/rock-paper-scissors-live.py\n")
    main()