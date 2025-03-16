import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.keras.preprocessing import image

# Caminho do modelo salvo
MODEL_PATH = "modelos/_home_munif-gebara-junior_ml_learn_codeminimap_dataset_novo_encrypted_68b36a01.keras"  # Ajuste conforme o nome do modelo salvo

# Carregar o modelo treinado
model = tf.keras.models.load_model(MODEL_PATH)
model_h5 = tf.keras.models.load_model(
    "/home/munif-gebara-junior/ml/learn/codeminimap/modelos/_home_munif-gebara-junior_ml_learn_codeminimap_dataset_novo_encrypted_68b36a01.h5")

import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.keras.preprocessing import image

# Caminho do modelo salvo
MODEL_PATH = "modelos/_home_munif-gebara-junior_ml_learn_codeminimap_dataset_novo_encrypted_68b36a01.keras"
MODEL_H5_PATH = "modelos/_home_munif-gebara-junior_ml_learn_codeminimap_dataset_novo_encrypted_68b36a01.h5"

# Carregar os modelos
model = tf.keras.models.load_model(MODEL_PATH)
model_h5 = tf.keras.models.load_model(MODEL_H5_PATH)

# Definir as classes (ajuste conforme necessário)
class_names = ['html', 'javaunittest', 'json', 'other', 'sql', 'svg', 'xml']


def predict_image(image_path, model_used, model_type="Keras"):
    try:
        # Carregar e processar a imagem
        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalização

        # Fazer a previsão
        predictions = model_used.predict(img_array)
        probabilities = tf.nn.softmax(predictions[0]).numpy()  # Aplica softmax para obter probabilidades

        predicted_class_index = np.argmax(probabilities)
        confidence = np.max(probabilities) * 100  # Converter para %

        predicted_class: str = class_names[predicted_class_index]

        print(f"📌 Imagem: {os.path.basename(image_path)} ({model_type})")
        print(f"🧐 Classe prevista: {class_names[predicted_class_index]} (Índice: {predicted_class_index})")
        print(f"🔢 Confiança: {confidence:.2f}%\n")
        return predicted_class

    except Exception as e:
        print(f"❌ Erro ao processar a imagem {image_path}: {e}")


def process_directory(directory):
    """Percorre todas as imagens na pasta e executa a previsão"""
    if not os.path.isdir(directory):
        print(f"❌ O caminho '{directory}' não é um diretório válido.")
        return

    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("⚠️ Nenhuma imagem encontrada na pasta.")
        return

    tamanho = len(image_files)
    print(f"📂 Encontradas {tamanho} imagens na pasta '{directory}'")
    erros = 0
    for file in image_files:
        image_path = os.path.join(directory, file)
        classe = predict_image(image_path, model, "Keras")
        if classe != 'json':
            erros += 1
    print(erros)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict.py <caminho_da_imagem_ou_pasta>, usando pasta padrão")

        path = '/home/munif-gebara-junior/ml/learn/codeminimap/dataset/novo_encrypted/json'
    else:
        path = sys.argv[1]
    if os.path.isdir(path):
        process_directory(path)
    else:
        predict_image(path, model, "Keras")
        # predict_image(path, model_h5, "H5")
