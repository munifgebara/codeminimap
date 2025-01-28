import os

import numpy as np
import skimage
import sklearn.model_selection
from matplotlib import pyplot as plt

import contr_dissimilarity

from PIL import Image



def converter_grayscale_para_rgb(imagem):
    # Carregar a imagem
    img = Image.open(imagem)

    # Se a imagem for em escala de cinza (modo 'L'), converta para RGB
    if img.mode == 'L':
        img_rgb = img.convert('RGB')
    else:
        img_rgb = img

    return img_rgb

# dataSetPath = "/media/munif-gebara-junior/Novo volume/doutorado/lbp_example-main/dataset/all_encrypted_fixed_size"
dataSetPath = "/home/munif-gebara-junior/Downloads/completo4classes"
# Read the images
X = []
Y = []
min_samples_per_class=500

strings_unicas = set()

for clazz in os.listdir(dataSetPath):
  image_files = list(filter(lambda file: file.lower().endswith((".png", ".jpg", ".jpeg")), os.listdir(f"{dataSetPath}/{clazz}")))
  if len(image_files) <min_samples_per_class:
      continue
  print(f"{clazz}: {len(image_files)}")


  for img_filename in os.listdir(f"{dataSetPath}/{clazz}"):
    complete_filename=f"{dataSetPath}/{clazz}/{img_filename}"
    # img = skimage.io.imread(complete_filename)
    img_rgb = converter_grayscale_para_rgb(complete_filename)
    img = img_rgb

    if img is not None:
        X.append(img)
        Y.append(clazz)


print(strings_unicas)

# Convert to numpy
X = np.array(X, dtype = np.uint8)
Y = sklearn.preprocessing.LabelEncoder().fit_transform(Y)

# Subset the problem to only 10 classes
X = X[Y < 10]
Y = Y[Y < 10]

# Split the data
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, random_state = 18081975, stratify = Y)

# Visualize some images and their corresponding labels.
plt.figure(figsize = (15, 8))
for i in range(30):
  ax = plt.subplot(3, 10, i + 1)
  plt.imshow(X_train[i] / 255.)
  plt.title(Y_train[i])
  plt.axis("off")

model = contr_dissimilarity.train(X_train, Y_train, save_location = "cache/model.pth",
                                  warmup_iterations = 2000, iterations = 2000,
                                  embeddingsize = 128, temperature = 0.5, batch_size = 32,
                                  patch_size = None, projection_head = [128, 64, 32], lr_warmup = 0.01, lr = 0.001)

train_embeddings = contr_dissimilarity.generate_embedding(model, X_train, patch_size = (64, 64), cache = "cache/train_embedding.pkl")
test_embeddings = contr_dissimilarity.generate_embedding(model, X_test, patch_size = (64, 64), cache = "cache/test_embedding.pkl")

contr_dissimilarity.umap_projection(train_embeddings, Y_train)
contr_dissimilarity.umap_projection(test_embeddings, Y_test)

X_prot, Y_prot = contr_dissimilarity.compute_prototypes(train_embeddings, Y_train, n_prototypes = 5, method = "kmeans", cache = "cache/prototypes.pkl")

contr_dissimilarity.umap_projection(X_prot, Y_prot)

contr_space_train = contr_dissimilarity.space_representation(model, train_embeddings, X_prot, cache = "cache/contr-space-train.pkl")
contr_space_test = contr_dissimilarity.space_representation(model, test_embeddings, X_prot, cache = "cache/contr-space-test.pkl")

contr_vector_X_train, contr_vector_Y_train = contr_dissimilarity.vector_representation(model,
                                                                                       X_train, Y_train, X_prot, Y_prot, patch_size = (64, 64), variations = 20,
                                                                                       cache = "cache/contr-vector-train.pkl")

contr_vector_X_test, _ = contr_dissimilarity.vector_representation(model,
                                                                   X_test, Y_test, X_prot, Y_prot, patch_size = (64, 64), variations = 20,
                                                                   cache = "cache/contr-vector-test.pkl")

# Embedding classification
np.random.seed(1234)

# Train a regular classifier
clf = sklearn.linear_model.LogisticRegression()
clf.fit(train_embeddings, Y_train)

# Evaluate the classifier
Y_pred = clf.predict(test_embeddings)
acc = sklearn.metrics.accuracy_score(Y_test, Y_pred)

print(f"Accuracy: {acc * 100:.1f}%")

# Contrastive dissimilarity space classification
np.random.seed(1234)

# Train a regular classifier
clf = sklearn.linear_model.LogisticRegression()
clf.fit(contr_space_train, Y_train)

# Evaluate the classifier
Y_pred = clf.predict(contr_space_test)
acc = sklearn.metrics.accuracy_score(Y_test, Y_pred)

print(f"Accuracy: {acc * 100:.1f}%")

# Contrastive dissimilarity vector classification
np.random.seed(1234)

# Train a regular classifier
clf = sklearn.linear_model.LogisticRegression()
clf.fit(contr_vector_X_train, contr_vector_Y_train)

X_pred_proba = clf.predict_proba(contr_vector_X_test)

# Transform the binary classification back into multiclass
X_pred = contr_dissimilarity.vector_to_class(X_pred_proba, Y_test, Y_prot)

# Evaluate the classifier
acc = sklearn.metrics.accuracy_score(Y_test, Y_pred)

print(f"Accuracy: {acc * 100:.1f}%")