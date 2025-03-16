import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import numpy as np
import shutil
from sklearn.metrics import confusion_matrix
import os
import hashlib


def gerar_nome_modelo(dataset_path):
    hash_nome = hashlib.md5(dataset_path.encode()).hexdigest()[:8]  # Hash curto
    nome_limpo = dataset_path.replace("/", "_").replace(" ", "_")  # Nome seguro
    return f"{nome_limpo}_{hash_nome}"


# Diretório das imagens
DATASET_PATH = '/home/munif-gebara-junior/ml/learn/codeminimap/dataset/novo_encrypted'
data_dir = pathlib.Path(DATASET_PATH)

# Diretório filtrado
filtered_data_dir = pathlib.Path('/home/munif-gebara-junior/ml/learn/codeminimap/dataset/novo_encrypted/filtered')

# Remover a pasta filtrada se existir para evitar lixo
if filtered_data_dir.exists():
    shutil.rmtree(filtered_data_dir)
filtered_data_dir.mkdir(parents=True, exist_ok=True)

# Filtrar apenas as pastas com mais de 500 arquivos
filtered_dirs = [d for d in data_dir.iterdir() if d.is_dir() and len(list(d.glob('*'))) > 500]

# Copiar imagens das pastas filtradas para o novo diretório
for d in filtered_dirs:
    target_dir = filtered_data_dir / d.name
    target_dir.mkdir(parents=True, exist_ok=True)
    for img_file in d.glob('*'):
        shutil.copy(img_file, target_dir)

# Parâmetros do dataset
BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
VALIDATION_SPLIT = 0.2
SEED = 42

# Carregar dataset a partir das pastas filtradas
train_ds = tf.keras.utils.image_dataset_from_directory(
    filtered_data_dir,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    filtered_data_dir,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

# Normalização dos dados
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Construção do modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names))
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Treinar o modelo
epochs = 5
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Diretório onde os modelos serão salvos
MODELOS_PATH = "modelos"
os.makedirs(MODELOS_PATH, exist_ok=True)

nome_modelo = gerar_nome_modelo(DATASET_PATH)
caminho_modelo = os.path.join(MODELOS_PATH, nome_modelo + ".keras")
model.save(caminho_modelo)



# Avaliação no conjunto de teste
test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print(f'Accuracy on validation set: {test_acc:.4f}')

# Plotar a acurácia do treinamento
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Gerar previsões para a matriz de confusão
y_true, y_pred = [], []
for images, labels in val_ds:
    y_true.extend(labels.numpy())
    predictions = model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_true, y_pred)

# Plotar a matriz de confusão
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


