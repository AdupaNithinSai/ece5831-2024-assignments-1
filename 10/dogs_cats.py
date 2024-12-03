import pathlib
import tensorflow as tf
#from tensorflow.keras.preprocessing import image_dataset_from_directory
#from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

class DogsCats:
    CLASS_NAMES = ['dog', 'cat']
    IMAGE_SHAPE = (180, 180, 3)
    BATCH_SIZE = 32
    BASE_DIR = pathlib.Path('dogs-vs-cats')
    SRC_DIR = pathlib.Path('dogs-vs-cats-original/train')
    EPOCHS = 20

    def __init__(self):
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.model = None

    def make_dataset_folders(self, subset_name, start_index, end_index):
        subset_dir = self.BASE_DIR / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)
        for class_name in self.CLASS_NAMES:
            class_dir = subset_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

        for category in ("dog", "cat"):
            dir = self.BASE_DIR / subset_name / category
            #print(dir)
            if os.path.exists(dir) is False:
                os.makedirs(dir)
            files = [f'{category}.{i}.jpg' for i in range(start_index, end_index)]
            #print(files)
            for i, file in enumerate(files):
                shutil.copyfile(src=self.SRC_DIR / file, dst=dir / file)
                if i % 500 == 0: # show only once every 100
                    print(f'src:{self.SRC_DIR / file} => dst:{dir / file}')

        for idx in range(start_index, end_index):
            category = 'dog' if 'dog' in os.listdir(self.SRC_DIR)[idx] else 'cat'
            src_file = self.SRC_DIR / f'{category}.{idx}.jpg'
            dst_file = subset_dir / category / f'{category}.{idx}.jpg'
            shutil.copy(src_file, dst_file)

    def _make_dataset(self, subset_name):
        subset_dir = self.BASE_DIR / subset_name
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            subset_dir,
            batch_size=self.BATCH_SIZE,
            image_size=self.IMAGE_SHAPE[:2],
            label_mode='int'  # integer labels
        )
        return dataset

    def make_dataset(self):
        self.train_dataset = self._make_dataset('train')
        self.valid_dataset = self._make_dataset('valid')
        self.test_dataset = self._make_dataset('test')

    def build_network(self, augmentation=True):
        data_augmentation = tf.keras.models.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2)
        ])

        self.model = tf.keras.models.Sequential()
        if augmentation:
            self.model.add(data_augmentation)

        self.model.add(tf.keras.layers.Rescaling(1.0 / 255))
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.IMAGE_SHAPE))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, model_name):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_name,
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            )
        ]

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            epochs=self.EPOCHS,
            callbacks=callbacks
        )
        self.plot_history(history)

    def load_model(self, model_name):
        self.model = tf.keras.models.load_model(model_name)

    def predict(self, image_file):
        img = tf.keras.preprocessing.image.load_img(image_file, target_size=self.IMAGE_SHAPE[:2])
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        score = predictions[0]
        plt.imshow(img)
        plt.title(f'This image is a {self.CLASS_NAMES[int(score > 0.5)]} with a {100 * score:.2f}% confidence.')
        plt.axis('off')
        plt.show()
    
    def plot_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

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