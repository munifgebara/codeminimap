# Contrastive dissimilarity utils

import os
import random
import math
import pickle

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
import seaborn as sns
import umap

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering


#########################################################################################################
#                    Generate patches from an image with optional minimum patch count                   #
#########################################################################################################
def gen_patches(img, patch_size, min_patches = None, regular = True):
  """
  Generates patches from an input image with an optional minimum patch count and patch distribution.

  Parameters
  ----------
  img : numpy.ndarray
      Input image as a NumPy array of shape (height, width, channels).
  patch_size : tuple
      A tuple (patch_height, patch_width) specifying the size of each patch.
  min_patches : int, optional
      The minimum number of patches required. Defaults to None.
  regular : bool, optional
      If True, generates a regular grid of patches. If False, randomly drops some patches to match `min_patches`. Defaults to True.

  Returns
  -------
  numpy.ndarray
      A 4D NumPy array of shape (number_of_patches, patch_height, patch_width, channels) containing the generated patches.
  """

  # Gets the shape of the input image.
  input_shape = img.shape

  # Calculates the minimum number of rows and columns of patches to cover the image.
  n_rows = math.ceil(input_shape[0] / patch_size[0])
  n_cols = math.ceil(input_shape[1] / patch_size[1])

  # Total number of patches.
  n_patches = n_rows * n_cols

  # Adjusts the number of rows and columns to ensure at least 'min_patches' patches are created.
  if min_patches is not None:
    while min_patches > n_patches:
      row_ratio = input_shape[0] / n_rows / patch_size[0]
      col_ratio = input_shape[1] / n_cols / patch_size[1]
      if row_ratio > col_ratio:
        n_rows += 1
      else:
        n_cols += 1
      n_patches = n_rows * n_cols

  # Calculates overlap between patches.
  row_overlap = math.ceil(((patch_size[0] * n_rows) - input_shape[0]) / (n_rows - 1))
  col_overlap = math.ceil(((patch_size[1] * n_cols) - input_shape[1]) / (n_cols - 1))

  # Generate all starting pixels, except the last one.
  row_patches = np.arange(0, input_shape[0], patch_size[0] - row_overlap)[0:(n_rows - 1)]
  col_patches = np.arange(0, input_shape[1], patch_size[1] - col_overlap)[0:(n_cols - 1)]

  # Create the last starting pixel manually to avoid going larger than the input image.
  row_patches = np.append(row_patches, input_shape[0] - patch_size[0])
  col_patches = np.append(col_patches, input_shape[1] - patch_size[1])

  # Generate rows and cols patches.
  row_patches = [(i, i + patch_size[0]) for i in row_patches]
  col_patches = [(i, i + patch_size[1]) for i in col_patches]

  # Combine them
  patches_indices = [(i, j) for i in row_patches for j in col_patches]

  # If not regular, then drop some patches to match min_patches
  if not regular:
    n_drop = n_patches - min_patches
    if n_drop > 0:
      # Generate random indices to delete
      drop_indices = random.sample(range(n_patches), n_drop)
      # Create a new list without the selected elements
      patches_indices = [patches_indices[i] for i in range(n_patches) if i not in drop_indices]
      # Update the number of patches
      n_patches = min_patches

  patches = np.zeros((n_patches, patch_size[0], patch_size[1], input_shape[2]), dtype = np.float32)

  # Extract patches from the image based on calculated indices.
  for patch_i in range(n_patches):
    row, col = patches_indices[patch_i]
    patches[patch_i] = img[row[0]:row[1], col[0]:col[1], :]

  # Normalize the patches if the image data type is 'uint8'.
  if img.dtype == "uint8":
    patches = (patches / 255).astype(np.float32)

  return patches


#########################################################################################################
#     Generate a batch of image pairs and their corresponding classes for training purposes             #
#########################################################################################################
def pair_batch(batch_size, X, Y, augmentations=None, size=None, device=None):
  """
  Generates a batch of image pairs and their corresponding classes for training purposes.

  Parameters
  ----------
  batch_size : int
      The number of image pairs to generate.
  X : numpy.ndarray
      The input image dataset as a NumPy array of shape (num_samples, height, width, channels).
  Y : numpy.ndarray
      The class labels for the input images as a NumPy array of shape (num_samples,).
  augmentations : list, optional
      A list of `albumentations` augmentation operations to apply. Defaults to None.
  device : torch.device, optional
      The device to move the output tensors to (e.g., `torch.device('cuda')` for GPU). Defaults to None.

  Returns
  -------
  list of torch.Tensor
      A list containing three elements:
      - pairs[0]: A tensor representing the first images in the pairs.
      - pairs[1]: A tensor representing the second images in the pairs.
      - pairs[2]: A tensor containing the class labels for each pair.
  """
  
  # Randomly select batch_size number of classes
  classes = np.random.choice(np.unique(Y), size = batch_size, replace = True)

  # Resize the images if a size is provided
  size = size if size is not None else (X.shape[1], X.shape[2])

  # Initialize arrays to store the pairs and their classes
  pairs = [torch.zeros((batch_size, 3, size[0], size[1]), dtype = torch.float32) for _ in range(2)]

  # Store the classes for later filtering of positive and negative samples
  pairs.append(torch.from_numpy(classes))

  # Define the augmentation pipeline
  transform = A.Compose([
    A.RandomCrop(size[0], size[1]),
    A.HorizontalFlip(),
    A.Rotate(),
    A.GaussianBlur(),
    A.RandomBrightnessContrast(),
    ToTensorV2()
  ])

  for i in range(batch_size):
    # Get indices of all samples that belong to the chosen class
    choices = np.where(Y == classes[i])[0]

    # Randomly select two samples of the same class
    idx_A = np.random.choice(choices)
    idx_B = np.random.choice(choices)

    img_A = transform(image = X[idx_A])["image"]
    img_B = transform(image = X[idx_B])["image"]

    # Save the samples to the pair list
    pairs[0][i] = img_A / 255.
    pairs[1][i] = img_B / 255.

  # Move the pairs to the device
  if device is not None:
    pairs[0] = pairs[0].to(device)
    pairs[1] = pairs[1].to(device)
    pairs[2] = pairs[2].to(device)

  return pairs


#########################################################################################################
#     Some utility functions to convert images to PyTorch tensors and create a custom Dataset class     #
#########################################################################################################
def img_to_torch(batch, device=None):
  """
  Converts a batch of images to PyTorch tensors and optionally moves them to a specified device.

  Parameters
  ----------
  batch : list of numpy.ndarray
      A list of images where each image is a NumPy array.
  device : torch.device, optional
      The device to move the output tensors to (e.g., `torch.device('cuda')` for GPU). Defaults to None.

  Returns
  -------
  torch.Tensor
      A batch of images converted to a PyTorch tensor, optionally moved to the specified device.
  """
  
  # Define the transformations
  transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
  ])

  # Apply the transform to the batch of images
  batch = torch.stack([transform(im) for im in batch])

  if device is not None:
    # Move data to device
    batch = batch.to(device)

  return batch

class PatchData(torch.utils.data.Dataset):
  """
  A custom Dataset class for generating and retrieving image patches as PyTorch tensors.

  Parameters
  ----------
  data : numpy.ndarray
      The input dataset containing images from which patches will be generated.
  patch_size : tuple
      A tuple (height, width) indicating the size of the patches to be generated.
  device : torch.device, optional
      The device to move the output tensors to (e.g., `torch.device('cuda')` for GPU). Defaults to None.

  Methods
  -------
  __getitem__(index)
      Generates patches from the image at the specified index and converts them to PyTorch tensors.
  __len__()
      Returns the number of images in the dataset.
  """
  
  def __init__(self, data, patch_size, device = None):
    self.data = data
    self.device = device
    self.patch_size = patch_size
    self.size = self.data.shape[0]

  def __getitem__(self, index):
    patches = gen_patches(self.data[index], self.patch_size)
    return img_to_torch(patches, self.device)

  def __len__(self):
    return self.size
  

#########################################################################################################
#                                 Contrastive dissimilarity loss                                        #
#########################################################################################################
class DissimilarityNTXentLoss(torch.nn.Module):
  """
  Computes the Normalized Temperature-scaled Cross-Entropy (NT-Xent) loss for contrastive dissimilarity.

  Parameters
  ----------
  temperature : float, optional
      The temperature scaling factor for the softmax operation. Defaults to 0.5.

  Methods
  -------
  forward(diss, y)
      Computes the NT-Xent loss given the dissimilarity scores and labels.
  """
  def __init__(self, temperature = 0.5):
    super(DissimilarityNTXentLoss, self).__init__()
    self.temperature = temperature

  def forward(self, diss, y):
    size = diss.shape[0]

    # Mask for positive samples
    y = torch.cat([y, y], dim = 0)
    y1 = torch.tile(y, [size])
    y2 = torch.repeat_interleave(y, size, dim = 0)
    pos_mask = torch.reshape(y1 == y2, (size, size))
    pos_mask.fill_diagonal_(False)

    # Mask for negative samples
    neg_mask = (~torch.eye(size, device = diss.device, dtype = bool)).float()

    # Compute nominator
    nominator = torch.sum(pos_mask * torch.exp(diss / self.temperature), dim = 1)

    # Compute denominator
    denominator = torch.sum(neg_mask * torch.exp(diss / self.temperature), dim = 1)

    # Compute loss
    loss_partial = -torch.log(nominator / denominator)
    loss = torch.mean(loss_partial)

    return loss


#########################################################################################################
#                          Base network, projection head, and contrastive model                         #
#########################################################################################################
class Network(torch.nn.Module):
  """
  A base network for feature extraction using EfficientNetV2-S.

  Parameters
  ----------
  embeddingsize : int
      The size of the output embedding vector.

  Methods
  -------
  forward(x)
      Forward pass through the network.
  """

  def __init__(self, embeddingsize):
    super(Network, self).__init__()

    # Load EfficientNetV2-S as the shared network
    self.network = torchvision.models.efficientnet_v2_s(weights = "DEFAULT")

    # Freeze the base network
    for param in self.network.parameters():
      param.requires_grad = False

    # Replace the last classification layer with a set of custom layers
    self.network.classifier = torch.nn.Sequential(
      torch.nn.Linear(self.network.classifier[1].in_features, 512),
      torch.nn.ReLU(),
      torch.nn.Dropout(p = 0.2),
      torch.nn.Linear(512, 256),
      torch.nn.ReLU(),
      torch.nn.Dropout(p = 0.2),
      torch.nn.Linear(256, embeddingsize)
    )

  def forward(self, x):
    # Pass the input through the shared network
    x = self.network(x)

    # Normalize the embedding vectors
    embedding = torch.nn.functional.normalize(x, p = 2, dim = 1)

    return embedding


class ProjectionHead(torch.nn.Module):
  """
  A projection head for computing the dissimilarity between embedding vectors.

  Parameters
  ----------
  embeddingsize : int
      The size of the input embedding vector.
  hidden_layers : list of int, optional
      List of sizes for hidden layers. Defaults to [128, 64, 32].
  output_size : int, optional
      Size of the output layer. Defaults to 1.

  Methods
  -------
  forward(x1, x2)
      Computes the dissimilarity score between two embedding vectors.
  """

  def __init__(self, embeddingsize, hidden_layers, output_size = 1):
    super(ProjectionHead, self).__init__()

    # Define the projection head architecture dynamically
    layers = []
    input_size = embeddingsize

    for hidden_size in hidden_layers:
      layers.append(torch.nn.Linear(input_size, hidden_size))
      layers.append(torch.nn.ReLU())
      input_size = hidden_size
    
    layers.append(torch.nn.Linear(input_size, output_size))
    self.projection_head = torch.nn.Sequential(*layers)

  def forward(self, x1, x2):
    return self.projection_head(torch.abs(x1 - x2))


class ContrastiveModel(torch.nn.Module):
  """
  A contrastive model for learning dissimilarity between image pairs.

  Parameters
  ----------
  embeddingsize : int
      The size of the output embedding vector.
  projection_head : list of int, optional
      List of sizes for hidden layers in the projection head.
  freeze_base : bool, optional
      Whether to freeze the base network. Defaults to True.

  Methods
  -------
  forward(x1, x2)
      Computes the dissimilarity between pairs of input images.
  freeze_network()
      Freezes the parameters of the base network to prevent them from being updated during training.
  unfreeze_network()
      Unfreezes the parameters of the base network to allow them to be updated during training.
  """

  def __init__(self, embeddingsize, projection_head, freeze_base = True):
    super(ContrastiveModel, self).__init__()
    self.network = Network(embeddingsize)
    self.projection_head = ProjectionHead(embeddingsize, hidden_layers = projection_head)


  def forward(self, x1, x2):
    # Encode the inputs
    x1 = self.network(x1)
    x2 = self.network(x2)

    if self.training:
      batch_size = x1.shape[0]

      # Repeat the elements to match the input expected by the network
      x = torch.cat([x1, x2])
      x1 = torch.tile(x, [batch_size * 2, 1])
      x2 = torch.repeat_interleave(x, batch_size * 2, dim = 0)

      dissimilarity = self.projection_head(x1, x2)
      dissimilarity = torch.reshape(dissimilarity, (batch_size * 2, -1))
    else:
      dissimilarity = self.projection_head(x1, x2)

    return dissimilarity

  def freeze_network(self):
    for param in self.network.parameters():
      param.requires_grad = False

  def unfreeze_network(self):
    for param in self.network.parameters():
      param.requires_grad = True


def train(X, Y, save_location, embeddingsize=128, temperature=0.5, batch_size=32, 
          warmup_iterations=1000, iterations=10000, patch_size=None, projection_head=[128, 64, 32], lr_warmup=0.01, lr=0.001):
  """
  Trains a contrastive dissimilarity model using the NT-Xent loss.

  Parameters
  ----------
  X : numpy.ndarray
      The input image dataset as a NumPy array of shape (num_samples, height, width, channels).
  Y : numpy.ndarray
      The class labels for the input images as a NumPy array of shape (num_samples,).
  save_location : str
      An identifier for the model to be saved or loaded.
  embeddingsize : int, optional
      The size of the embedding vector. Defaults to 128.
  temperature : float, optional
      The temperature scaling factor for the NT-Xent loss. Defaults to 0.5.
  batch_size : int, optional
      The number of image pairs in each batch. Defaults to 32.
  warmup_iterations : int, optional
      The number of iterations for the warm-up phase. Defaults to 1000.
  iterations : int, optional
      The number of training iterations. Defaults to 10000.
  patch_size : tuple, optional
      A tuple (height, width) indicating the size of the patches to be generated. If None, the entire image is used. Defaults to None.
  projection_head : list of int, optional
      A list of sizes for hidden layers in the projection head. Defaults to [128, 64, 32].
  lr_warmup : float, optional
      The learning rate for the warm-up phase. Defaults to 0.01.
  lr : float, optional
      The learning rate for the main training phase. Defaults to 0.001.

  Returns
  -------
  ContrastiveModel
      The trained contrastive model.
  """
  
  # Define the model filename based on model_id
  print(f"Model file: {save_location}")

  # Define computation device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  contr_model = None

  # Load pre-trained model if it exists
  if os.path.isfile(save_location):
    print("Loading pre-trained model...")
    contr_model = ContrastiveModel(embeddingsize, projection_head = projection_head)
    contr_model.load_state_dict(torch.load(save_location, weights_only = True))
    contr_model.to(device)

  # Train a new model if not loaded
  if contr_model is None:
    print("Training a new model...")

    contr_model = ContrastiveModel(embeddingsize, projection_head = projection_head)
    contr_model.to(device)

    # Initialize loss function, optimizer, and set model to training mode
    contr_loss = DissimilarityNTXentLoss(temperature)
    optimizer = torch.optim.SGD(contr_model.parameters(), lr = lr_warmup, momentum = 0.9)
    contr_model.train()

    print("Warmup Phase")

    # Warm-up phase: freeze most layers and train the classifier layers only
    train_loss = 0
    for epoch in range(warmup_iterations // 100):
      for _ in range(100):
        # x1, x2, y = pair_batch(batch_size, X, Y, size=patch_size, device=device)
        x1, x2, y = pair_batch(batch_size, X, Y, size = patch_size, device = device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = contr_model(x1, x2)

        # Compute loss
        loss = contr_loss(outputs, y)
        train_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

      # Compute average loss for the epoch
      print(f"Epoch {epoch + 1}, Warmup Loss: {train_loss / 100:.4f}")
      train_loss = 0

    print("Training Phase")

    # Unfreeze the network for full training
    contr_model.unfreeze_network()

    # Reinitialize optimizer for the training phase with a lower learning rate
    optimizer = torch.optim.SGD(contr_model.parameters(), lr = lr, momentum = 0.9)

    # Training phase
    train_loss = 0
    for epoch in range(iterations // 100):
      for _ in range(100):
        x1, x2, y = pair_batch(batch_size, X, Y, size = patch_size, device = device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = contr_model(x1, x2)

        # Compute loss
        loss = contr_loss(outputs, y)
        train_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

      # Compute average loss for the epoch
      print(f"Epoch {epoch + 1}, Training Loss: {train_loss / 100:.4f}")
      train_loss = 0

    # Save the trained model
    print("Saving the trained model...")
    # Extract the directory from the save location
    dir = os.path.dirname(save_location)
    if dir:
      os.makedirs(dir, exist_ok = True)
    torch.save(contr_model.state_dict(), save_location)

  # Freeze the model parameters and set to evaluation mode
  contr_model.freeze_network()
  contr_model.eval()

  print("Model is ready for evaluation.")

  return contr_model


#########################################################################################################
#                                       Embedding generation                                            #
#########################################################################################################
def generate_embedding(model, data, patch_size, cache="embedding.pkl"):
  """
  Generate embeddings for a given dataset and caches them.

  Parameters
  ----------
  model : torch.nn.Module
      The trained model to be used for generating embeddings.
  data : numpy.ndarray
      The data as a NumPy array.
  patch_size : tuple of int
      The size of the patches to be generated from the images.
  cache : str, optional
      The file path for caching and loading precomputed embeddings. Defaults to 'embedding.pkl'.

  Returns
  -------
  np.ndarray
      The computed embeddings for the specified dataset.
  """

  # Check if the embeddings file exists
  if os.path.isfile(cache):
    with open(cache, "rb") as f:
      embedding = pickle.load(f)
  else:
    # Define computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the patch loader
    patch_data = PatchData(data, patch_size = patch_size, device = device)
    patch_dataloader = torch.utils.data.DataLoader(dataset = patch_data, batch_size = None, shuffle = False)

    # Extract patches and generate embeddings
    embedding = torch.stack([
      model.network(batch_data) for _, batch_data in enumerate(patch_dataloader)
    ])

    # Convert to numpy and compute the mean for each sample
    embedding = np.array(embedding.cpu(), dtype = np.float32)
    embedding = np.mean(embedding, axis=1)

    # Store the embeddings and save them to the cache file
    with open(cache, "wb") as f:
      pickle.dump(embedding, f, protocol=pickle.HIGHEST_PROTOCOL)

  return embedding


#########################################################################################################
#                                            UMAP                                                       #
#########################################################################################################
def umap_projection(encoded_X, Y, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
  """
  Visualizes high-dimensional encodings using UMAP for dimensionality reduction.

  Parameters
  ----------
  encoded_X : np.ndarray
      The high-dimensional encodings to be visualized, of shape (num_samples, num_features).
  Y : np.ndarray or list
      The labels or classes corresponding to each sample in encoded_X.
  n_neighbors : int, optional
      The number of neighbors to consider for UMAP. Controls local versus global structure. Defaults to 15.
  min_dist : float, optional
      The minimum distance between points in the low-dimensional UMAP representation. Defaults to 0.1.
  n_components : int, optional
      The number of dimensions for the UMAP output. Typically 2 for 2D visualization. Defaults to 2

.
  random_state : int, optional
      The random seed for reproducibility. Defaults to 42.

  Returns
  -------
  None
  """

  # Check input dimensions
  assert len(encoded_X) == len(Y), "The number of samples in encoded_X and Y must be the same."

  # Initialize UMAP reducer
  reducer = umap.UMAP(
    n_neighbors = n_neighbors,
    min_dist = min_dist,
    n_components = n_components,
    random_state = random_state,
    n_jobs = 1
  )
  
  # Fit and transform the data
  trn = reducer.fit_transform(encoded_X)

  # Plotting
  plt.figure(figsize=(10, 6))
  sns.scatterplot(
    x=trn[:, 0], 
    y=trn[:, 1], 
    hue = Y, 
    palette = sns.color_palette("bright", len(np.unique(Y))), 
    legend = False
  )
  plt.title("UMAP Projection")
  plt.xlabel("UMAP-1")
  plt.ylabel("UMAP-2")
  plt.show()


#########################################################################################################
#                                             Prototype selection                                       #
#########################################################################################################
def compute_centroids(X, Y, K):
  """
  Computes centroids for each cluster.

  Parameters
  ----------
  X : np.ndarray
      The input data as a NumPy array of shape (num_samples, num_features).
  Y : np.ndarray
      The cluster labels corresponding to each sample in X.
  K : int
      The number of clusters.

  Returns
  -------
  np.ndarray
      A NumPy array of shape (K, num_features) containing the centroids of each cluster.
  """

  m, n = X.shape
  centroids = np.zeros((K, n))
  for k in range(K):
    x = X[Y == k]
    centroids[k, :] = np.mean(x, axis = 0)
  return centroids

def compute_prototypes(embeddings, Y, n_prototypes=5, method="kmeans", cache="prototypes.pkl"):
  """
  Computes prototypes for each class using specified clustering methods.

  Parameters
  ----------
  embeddings : np.ndarray
      The embeddings data as a NumPy array of shape (num_samples, num_features).
  Y : np.ndarray
      The class labels corresponding to each sample in the embeddings array.
  n_prototypes : int, optional
      The number of prototypes to compute for each class. Defaults to 5.
  method : str, optional
      The clustering method to use for computing prototypes. Options are 'kmeans', 'spectral', and 'hierarchical'. Defaults to 'kmeans'.
  cache : str, optional
      The file path for caching precomputed prototypes. Defaults to 'prototypes.pkl'.

  Returns
  -------
  np.ndarray
      A NumPy array of shape (total_prototypes, num_features) containing the computed prototypes for each class.
  np.ndarray
      A NumPy array of shape (total_prototypes,) containing the class label for each prototype.
  """

  # Check if the prototypes file exists
  if os.path.isfile(cache):
    with open(cache, "rb") as f:
      prototypes, classes = pickle.load(f)
  else:
    # Count the number of unique classes in the data
    uniq_classes = np.unique(Y)
    n_classes = len(uniq_classes)

    # Compute the total number of prototypes
    total_prototypes = n_classes * n_prototypes
    prototypes = np.zeros((total_prototypes, embeddings.shape[1]), dtype = np.float32)
    classes = np.zeros(total_prototypes, dtype = np.int32)

    # Find prototypes in each class
    for idx, cls in enumerate(uniq_classes):
      X_embedding = embeddings[np.where(Y == cls)]
      start, end = n_prototypes * (idx + 1) - n_prototypes, n_prototypes * (idx + 1)

      # Clustering based on the chosen method
      if method == "kmeans":
        # K-means clustering to find prototypes
        clustering = KMeans(n_clusters = n_prototypes, init = "k-means++", n_init = "auto", random_state = 1234).fit(X_embedding)
        centroids = clustering.cluster_centers_
      elif method == "spectral":
        # Spectral clustering
        clustering = SpectralClustering(n_clusters = n_prototypes, affinity = "nearest_neighbors", random_state = 1234).fit(X_embedding)
        labels = clustering.labels_
        centroids = compute_centroids(X_embedding, labels, n_prototypes)
      elif method == "hierarchical":
        # Hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters = n_prototypes).fit(X_embedding)
        labels = clustering.labels_
        centroids = compute_centroids(X_embedding, labels, n_prototypes)
      else:
        raise ValueError("Unsupported clustering method. Choose from 'kmeans', 'spectral', or 'hierarchical'.")

      # Store the computed centroids and assign class labels to each prototype
      prototypes[start:end, :] = centroids
      classes[start:end] = cls  # Assign the class label to each prototype

    # Save the computed prototypes and their classes to a file
    with open(cache, "wb") as f:
      pickle.dump((prototypes, classes), f, protocol = pickle.HIGHEST_PROTOCOL)

  return prototypes, classes


#########################################################################################################
#                                  Dissimilarity representation                                         #
#########################################################################################################
def space_representation(model, encoded, X_prot, cache="contr-space.pkl"):
  """
  Computes the contrastive dissimilarity space for a given dataset.

  Parameters
  ----------
  model : torch.nn.Module
      The trained model with a projection head for computing dissimilarity.
  encoded : np.ndarray
      A NumPy array containing the encoded data for the specific dataset to compute dissimilarity for.
  X_prot : np.ndarray
      A NumPy array of shape (num_prototypes, num_features) containing the prototypes.
  cache : str, optional
      The file path for caching precomputed dissimilarity space. Defaults to 'contr-space.pkl'.

  Returns
  -------
  np.ndarray
      A NumPy array containing the dissimilarity space representation.
  """

  # Check if the contrastive space file exists
  if os.path.isfile(cache):
    with open(cache, "rb") as f:
      contr_space = pickle.load(f)
  else:
    # Convert prototypes to PyTorch tensor and move to GPU
    X_prot = torch.from_numpy(X_prot).to("cuda")
    n_prot = X_prot.shape[0]

    # Initialize the contrastive space list for the given set
    contr_space = []

    # Loop through each data point in the dataset
    for idx in range(encoded.shape[0]):
      local_x = encoded[[idx], :]
      local_x = np.repeat(local_x, n_prot, axis = 0)  # Repeat to match the number of prototypes
      local_x = torch.from_numpy(local_x).to("cuda")
      
      # Compute dissimilarity using the model's projection head
      diss = model.projection_head(local_x, X_prot).squeeze().cpu().detach().numpy()
      contr_space.append(diss)

    # Convert the list to a NumPy array for consistency
    contr_space = np.array(contr_space)

    # Save the computed contrastive dissimilarity space to a file
    with open(cache, "wb") as f:
      pickle.dump(contr_space, f, protocol = pickle.HIGHEST_PROTOCOL)

  return contr_space


def vector_representation(model, X, Y, X_prot, Y_prot, patch_size, variations=20, cache="contr-vector.pkl"):
  """
  Computes the contrastive dissimilarity vector for given datasets using a projection head model.

  Parameters
  ----------
  model : torch.nn.Module
      The trained model with a projection head for computing dissimilarity.
  X : np.ndarray
      The input data as a NumPy array.
  Y : np.ndarray
      The class labels for the input data.
  X_prot : np.ndarray
      The prototypes data as a NumPy array.
  Y_prot : np.ndarray
      The class labels for the prototype data.
  patch_size : tuple of int
      The size of the patches to generate.
  variations : int, optional
      The minimum number of variations to generate per input image. Defaults to 20.
  cache : str, optional
      The file path for caching precomputed dissimilarity vector. Defaults to 'contr-vector.pkl'.

  Returns
  -------
  dict
      A dictionary containing the contrastive dissimilarity vectors and corresponding labels for input data.
  """

  # Check if the contrastive vector file exists
  if os.path.isfile(cache):
    with open(cache, "rb") as f:
      X_contr_vector, Y_contr_vector = pickle.load(f)
  else:
    # Generate label pairs for training data
    Y_contr_vector = np.transpose([np.repeat(Y, len(Y_prot)), np.tile(Y_prot, len(Y))])
    Y_contr_vector = Y_contr_vector[:, 0] == Y_contr_vector[:, 1]

    X_contr_vector = []

    # Loop through each data point in the dataset
    for idx in range(X.shape[0]):
      # Prepare patches and prototypes for the projection head
      local_patches = img_to_torch(gen_patches(X[idx], patch_size, min_patches = variations * 5, regular = False), device = "cuda")
      patch_encodings = model.network(local_patches)

      # Get the mean of a set of patches to create more stable encodings
      patch_encodings = torch.mean(torch.stack(patch_encodings.split(5)), dim = 1, dtype = torch.float32)

      number_patches = patch_encodings.shape[0]
      number_prototypes = X_prot.shape[0]

      patch_encodings = torch.tile(patch_encodings, [number_prototypes, 1])
      local_prototypes = np.repeat(X_prot, number_patches, axis = 0)
      local_prototypes = torch.from_numpy(local_prototypes).to("cuda")

      diss_vec = model.projection_head(patch_encodings, local_prototypes)
      diss_vec = np.array(np.split(diss_vec.cpu().detach().numpy(), number_prototypes))
      X_contr_vector.append(diss_vec)

    # Reshape to match the labels
    X_contr_vector = np.reshape(X_contr_vector, (len(Y) * number_prototypes, -1))

    # Save the computed contrastive dissimilarity vectors to a file
    with open(cache, "wb") as f:
      pickle.dump((X_contr_vector, Y_contr_vector), f, protocol = pickle.HIGHEST_PROTOCOL)

  return X_contr_vector, Y_contr_vector


def vector_to_class(X_proba, Y, Y_prot):
  """
  Transforms contrastive dissimilarity vectors back into multiclass classification.

  Parameters
  ----------
  X_proba : np.ndarray
      The predicted probabilities for the test data as an array of shape (n_samples * n_prototypes, 2).
  Y : np.ndarray
      The true class labels for the test data.
  Y_prot : np.ndarray
      The prototype labels used to determine the number of prototypes per class.

  Returns
  -------
  np.ndarray
      The predicted class labels for the test data.
  """
  
  # Reshape to match the number of test samples
  X_proba = np.reshape(X_proba[:, 1], (Y.shape[0], -1))

  # Find the number of prototypes per class, a single int value
  prot_per_class = np.bincount(Y_prot).max()

  # Average the prediction probabilities across all prototypes
  X_proba = np.reshape(X_proba, (Y.shape[0], -1, prot_per_class))
  X_proba = np.max(X_proba, axis = -1)

  # Get the class with the highest probability for each test sample
  X_pred = np.argmax(X_proba, axis = 1)

  return X_pred