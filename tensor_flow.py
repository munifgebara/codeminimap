import math
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

import seaborn as sns
from netaddr.strategy.ipv6 import width

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions


import pathlib

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from yaml import full_load


def plot_confusion_matrix(conf_matrix, class_names):
  plt.figure(figsize=(10, 7))
  sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
              xticklabels=class_names, yticklabels=class_names)
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.title('Confusion Matrix')
  plt.show()


data_dir = pathlib.Path('/media/munif-gebara-junior/Novo volume/doutorado/lbp_example-main/dataset/five').with_suffix('')
data_dir = pathlib.Path('/media/munif-gebara-junior/munif/model').with_suffix('')
image_count = len(list(data_dir.glob('*/*.*')))
print(image_count)


batch_size = 32
img_height = 128
img_width = 128

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=180875,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)



plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break



AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



normalization_layer = layers.Rescaling(1./255)


normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
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


def scheduler(epoch, lr):
  if epoch < 5:
    return lr
  else:
    return lr * np.exp(-0.1)


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

epochs=40
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[callback]
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



class estimator:
  _estimator_type = ''
  classes_=[]
  def __init__(self, model, classes):
    self.model = model
    self._estimator_type = 'classifier'
    self.classes_ = classes
  def predict(self, X):
    y_prob= self.model.predict(X)
    y_pred = y_prob.argmax(axis=1)
    return y_pred

# Instantiate the classifier
classifier = estimator(model, class_names)

# Predict on validation set
y_true = []
y_pred = []

for images, labels in val_ds:
    y_true.extend(labels.numpy())
    predictions = classifier.predict(images.numpy())
    y_pred.extend(predictions)
    print (predictions)

# Convert to numpy arrays for confusion matrix
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute the confusion matrix
conf_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)

# Plot the confusion matrix
plot_confusion_matrix(conf_matrix, class_names)


destination='/media/munif-gebara-junior/munif/model_pre/'

def moveImagem(full_path):
  try:
    img = cv2.imread(full_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_width, img_height))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array = preprocess_input(img_array)  # Preprocess the image
    pred = model.predict(img_array)
    predicted_class_index = np.argmax(pred[0])
    predicted_class_label =  class_names[predicted_class_index]
    move_file(full_path, predicted_class_label)


  except Exception as ex:
    print('ignorando', full_path, ex)
    move_file(full_path, 'errors')


def move_file(source_file, destination_folder):
  try:
    if not os.path.exists(destination + destination_folder):
      os.makedirs(destination + destination_folder)
    destination_file = os.path.join(destination + destination_folder, os.path.basename(source_file))
    shutil.move(source_file, destination_file)
  except Exception as ex:
    print('Imposible move file ', source_file, ex)


def list_files_recursive(path='.'):
  print(path)
  for entry in os.listdir(path):
    full_path = os.path.join(path, entry)
    if os.path.isdir(full_path):
      list_files_recursive(full_path)
    else:
      moveImagem(full_path)

novos='/home/munif-gebara-junior/.tx/nnn/fotos/bb'
list_files_recursive(novos)


novos_dir = pathlib.Path(novos).with_suffix('')

