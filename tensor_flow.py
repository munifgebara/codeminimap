import os
import shutil
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import ConfusionMatrixDisplay

# Constants
BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
AUTOTUNE = tf.data.AUTOTUNE


# Function Definitions
def move_image(full_path):
    try:
        img = cv2.imread(full_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_array = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
        predicted_class = class_names[np.argmax(model.predict(img_array)[0])]
        move_file(full_path, predicted_class)
    except Exception as ex:
        print(f'Ignoring {full_path}: {ex}')
        move_file(full_path, 'errors')


def move_file(source_file, destination_folder):
    try:
        os.makedirs(os.path.join(destination, destination_folder), exist_ok=True)
        shutil.move(source_file, os.path.join(destination, destination_folder, os.path.basename(source_file)))
    except Exception as ex:
        print(f'Unable to move file {source_file}: {ex}')


def list_files_recursive(path='.'):
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            list_files_recursive(full_path)
        else:
            move_image(full_path)


def scheduler(epoch, lr):
    return lr if epoch < 5 else lr * np.exp(-0.1)


class Estimator:
    def __init__(self, model, classes):
        self.model = model
        self.classes_ = classes
        self._estimator_type = 'classifier'

    def predict(self, X):
        return self.model.predict(X).argmax(axis=1)


# Load dataset
data_dir = pathlib.Path('/home/munif/PycharmProjects/codeminimap/datasets/all_encrypted_fixed_size')
image_count = len(list(data_dir.glob('*/*.*')))
print(f"Total images: {image_count}")

# Create training and validation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=180875,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

# Optimize dataset performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalize dataset
normalization_layer = layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Define model
num_classes = len(class_names)
model = Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Train model
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
epochs = 40
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[callback])

# Plot training history
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), history.history['accuracy'], label='Training Accuracy')
plt.plot(range(epochs), history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(epochs), history.history['loss'], label='Training Loss')
plt.plot(range(epochs), history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Confusion matrix
classifier = Estimator(model, class_names)
y_true, y_pred = [], []
for images, labels in val_ds:
    y_true.extend(labels.numpy())
    y_pred.extend(classifier.predict(images.numpy()))

y_true, y_pred = np.array(y_true), np.array(y_pred)
conf_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Process new images
# destination = '/media/munif-gebara-junior/munif/model_pre/'
# new_images_dir = '/home/munif-gebara-junior/.tx/nnn/fotos/bb'
# list_files_recursive(new_images_dir)
