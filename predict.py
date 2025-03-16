import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.preprocessing import image

# Caminho do modelo salvo
MODEL_PATH = "modelos/_home_munif-gebara-junior_ml_learn_codeminimap_dataset_novo_encrypted_68b36a01.keras"  # Ajuste conforme o nome do modelo salvo

# Carregar o modelo treinado
model = tf.keras.models.load_model(MODEL_PATH)
model_h5 = tf.keras.models.load_model("/home/munif-gebara-junior/ml/learn/codeminimap/modelos/_home_munif-gebara-junior_ml_learn_codeminimap_dataset_novo_encrypted_68b36a01.h5")


# Definir as classes (deve ser o mesmo que foi usado no treinamento)
class_names = ['html', 'javaunittest', 'json', 'other', 'sql', 'svg', 'xml'] # Ajuste conforme necessário


def predict_image(image_path):
    try:
        # Carregar e processar a imagem
        img = image.load_img(image_path, target_size=(128, 128))  # Ajuste o tamanho conforme necessário
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalização, se necessário

        # Fazer a previsão
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        print(f"A imagem pertence à classe: {class_names[predicted_class]} (Confiança: {confidence:.2f})")
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")


def predict_image_h5(image_path):
    try:
        # Carregar e processar a imagem
        img = image.load_img(image_path, target_size=(128, 128))  # Ajuste o tamanho conforme necessário
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalização, se necessário

        # Fazer a previsão
        predictions = model_h5.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        print(f"A imagem pertence à classe: {class_names[predicted_class]} (Confiança: {confidence:.2f})")
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict.py <caminho_da_imagem>")
    else:
        predict_image(sys.argv[1])
        predict_image_h5(sys.argv[1])


# python predict.py /home/munif-gebara-junior/ml/learn/codeminimap/dataset/novo_encrypted/filtered/javaunittest/0001416391772903.png
