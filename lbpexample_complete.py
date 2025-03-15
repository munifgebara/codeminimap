import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
from skimage.feature import graycomatrix, graycoprops
import pickle
import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import joblib
import torch.nn.functional as F
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops
import os
from torch.utils.data import DataLoader
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from sklearn.metrics import precision_score, recall_score, f1_score
from array import array
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.utils import to_categorical
from keras import models, layers


# LBP
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = local_binary_pattern(
            image, self.numPoints, self.radius, method="nri_uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist

# Classe para descritor Gabor


class GaborFeatures:
    def __init__(self, kernel_size=(21, 21), sigma=4.0, lambd=10.0, gamma=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.lambd = lambd
        self.gamma = gamma

    def describe(self, image):
        gabor_kernels = []
        for theta in range(4):  # Variação angular
            theta = theta / 4. * np.pi
            kernel = cv2.getGaborKernel(
                self.kernel_size, self.sigma, theta, self.lambd, self.gamma, 0, ktype=cv2.CV_32F)
            gabor_kernels.append(kernel)

        # Aplicar os filtros Gabor e calcular a média como feature
        features = [cv2.filter2D(image, cv2.CV_8UC3, kernel).mean()
                    for kernel in gabor_kernels]
        return np.array(features)

# Classe para descritor GLCM


class GLCMFeatures:
    def __init__(self, distances=[1], angles=[0], levels=256):
        self.distances = distances
        self.angles = angles
        self.levels = levels

    def describe(self, image):
        glcm = graycomatrix(image, distances=self.distances, angles=self.angles,
                            levels=self.levels, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        return np.array([contrast, homogeneity])

# Arquiterura da rede


class DenseNetwork:
    def __init__(self, config, X_train, y_train, X_val, y_val):
        self.input_shape = tuple(config['model']['input_shape'])
        self.num_classes = config['model']['num_classes']
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']

        # Definindo n_features com base no tamanho do vetor de características
        self.n_features = X_train.shape[1]
        print(f'Número de características: {self.n_features}')

        self.model = self.build_model()

        # Mapeando as classes para índices
        self.label_map = {label: idx for idx, label in enumerate(set(y_train))}
        y_train = [self.label_map[label] for label in y_train]
        y_val = [self.label_map[label] for label in y_val]

        self.X_train, self.y_train, self.X_val, self.y_val = self.load_data(
            X_train, y_train, X_val, y_val)

        # Class weights inicializados (se necessário)
        self.class_weights = None

    def build_model(self):
        model = models.Sequential()

        # Camada de entrada
        model.add(layers.Input(shape=(self.n_features,)))

        # Adicionando 5 camadas densas
        for i in range(5):
            model.add(layers.Dense(128, activation='relu'))
            # Normalização para estabilizar o treinamento
            model.add(layers.BatchNormalization())
            # Regularização para evitar overfitting
            model.add(layers.Dropout(0.1))

        # Camada de saída
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Compilação
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def load_data(self, X_train, y_train, X_val, y_val):
        # Normalizando os dados (se necessário)
        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')

        # Aplicando one-hot encoding nas labels
        y_train = to_categorical(y_train, self.num_classes)
        y_val = to_categorical(y_val, self.num_classes)

        return X_train, y_train, X_val, y_val

    def train(self):
        # Callbacks para monitorar o treinamento
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)

        # Treinando o modelo
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.X_val, self.y_val),
            class_weight=self.class_weights,
            callbacks=[early_stopping]
        )
        return history

    def evaluate(self):
        # Avaliando o modelo
        test_loss, test_acc = self.model.evaluate(self.X_val, self.y_val)

        # Fazendo previsões
        y_pred_probs = self.model.predict(self.X_val)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Convertendo y_val de volta para rótulos originais
        y_val_labels = np.argmax(self.y_val, axis=1)

        # Calculando métricas de precisão, recall e F1
        precision = precision_score(
            y_val_labels, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val_labels, y_pred, average='weighted')
        f1 = f1_score(y_val_labels, y_pred, average='weighted')

        print("Classes reais:", np.unique(y_val_labels))
        print("Classes previstas:", np.unique(y_pred))

        # Retornando as métricas em um dicionário
        return {
            "accuracy": test_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def predict(self, X):
        # Fazendo previsões no conjunto de dados de entrada
        return np.argmax(self.model.predict(X), axis=1)
# Função para salvar matriz de confusão e relatório


def salvar_matriz_confusao(all_labels, all_preds, results_dir, nome_modelo, extractor_name, all_matriz, name_matriz):
    # Identificar todas as classes únicas
    unique_labels = np.unique(all_labels)  # Determina os rótulos únicos

    if len(all_labels) == len(all_preds):
        # Gerar a matriz de confusão
        cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)

        # Configurar o gráfico
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title(f"Matriz de Confusão_NN_{extractor_name}")
        plt.xlabel('Predição')
        plt.ylabel('Verdadeiro')

        # Salvar a matriz de confusão
        output_path = os.path.join(
            results_dir, f"confusion_matrix_{nome_modelo}.png")
        plt.savefig(output_path)
        plt.close()

        print(f"Matriz de confusão salva em: {output_path}")
        all_matriz.append(cm)
        name_matriz.append(f'NN_{extractor_name}')
    else:
        print(f"Erro: as listas de labels e predições têm tamanhos diferentes: {
              len(all_labels)} vs {len(all_preds)}")


def salvar_modelo(model, model_path):
    # Se for um modelo Keras, usa o método de salvar do Keras
    if hasattr(model, 'save'):
        model.save(model_path)
    # Se for um modelo PyTorch, usa o método de salvar do PyTorch
    else:
        torch.save(model.state_dict(), model_path)


def plotar_matrizes_confusao(matrizes_confusao, class_labels, output_path):
    """
    Plota 12 matrizes de confusão em uma única figura.

    Args:
        matrizes_confusao (list): Lista de 12 matrizes de confusão (np.ndarray).
        class_labels (list): Lista dos nomes das classes para os eixos das matrizes.
        output_path (str): Caminho para salvar a figura gerada.
    """
    # Configurar a figura e os subplots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))  # 3 linhas x 4 colunas
    axes = axes.flatten()  # Acessa os subplots em um formato iterável

    for i, cm in enumerate(matrizes_confusao):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=class_labels, yticklabels=class_labels)
        axes[i].set_title(f'Matriz {i+1}')
        axes[i].set_xlabel('Predição')
        axes[i].set_ylabel('Verdadeiro')

    # Remover eixos extras (se houver menos de 12 matrizes)
    for ax in axes[len(matrizes_confusao):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Figura com 12 matrizes de confusão salva em: {output_path}")


# Configurações de saída
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

classSizes = [500]
all_matriz = []
name_matriz = []

for classSize in classSizes:

    feature_extractor = []
    contf = 0
    feature_extractor = [
        LocalBinaryPatterns(8, 2),
        GaborFeatures(),
        GLCMFeatures(),
    ]

    for contf in range(len(feature_extractor)):

        images = []
        labels = []
        textLabels = []
        filenames = []

        baseFolder = "/home/munif/PycharmProjects/codeminimap/datasets/all_encrypted_fixed_size/"

        folders = os.listdir(baseFolder)
        l = 0
        for folder in folders:
            print('Loading folder ', folder)
            c = 0

            if len(os.listdir(baseFolder + folder)) < classSize:
                print("não usando ", folder, " por ter somente ", len(os.listdir(baseFolder + folder)), "amostras")
                continue

            textLabels.append(folder)

            for image_path in os.listdir(baseFolder + folder):
                image = cv2.imread(os.path.join(
                    baseFolder + folder, image_path), cv2.IMREAD_GRAYSCALE)
                # image = cv2.resize(image, (256, 256))
                images.append(image)
                labels.append(l)
                filenames.append(image_path)

                c += 1
                if c >= classSize:
                    break
            l += 1

        t = 0
        for textLabel in textLabels:
            print(t, textLabel)
            t += 1

        # Inicialize a lista de características
        data = []

        # Extração de características baseada no valor de contf
        if contf == 0:  # LBP
            for image in images:
                hist = feature_extractor[contf].describe(image)
                data.append(hist)
        elif contf == 1:  # Gabor
            for image in images:
                gabor_features = feature_extractor[contf].describe(image)
                data.append(gabor_features)
        elif contf == 2:  # GLCM
            for image in images:
                glcm_features = feature_extractor[contf].describe(image)
                data.append(glcm_features)
        else:
            raise ValueError(f"Valor inesperado para contf: {contf}")

        data = np.array(data)
        labels = np.array(labels)

        # Divisão do dataset
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=180875)

        # SVM com GridSearchCV
        param_grid_svm = {'C': [1000, 2000, 10000], 'gamma': [
            2, 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
        grid_svm = GridSearchCV(SVC(), param_grid_svm, refit=True, verbose=3)
        grid_svm.fit(X_train, y_train)

        print("Best parameters for SVM: ", grid_svm.best_params_)
        print("Best estimator for SVM: ", grid_svm.best_estimator_)

        # RandomForest com GridSearchCV
        param_grid_rf = {'n_estimators': [
            10, 50, 100, 200, 500, 1000], 'max_depth': [None, 10, 20, 30, 100]}
        grid_rf = GridSearchCV(RandomForestClassifier(),
                               param_grid_rf, refit=True, verbose=3)
        grid_rf.fit(X_train, y_train)

        print("Best parameters for RandomForest: ", grid_rf.best_params_)
        print("Best estimator for RandomForest: ", grid_rf.best_estimator_)

        # KNeighbors com GridSearchCV
        param_grid_knn = {'n_neighbors': [
            3, 5, 7, 9, 11, 13, 15, 17, 19], 'weights': ['uniform', 'distance']}
        grid_knn = GridSearchCV(KNeighborsClassifier(),
                                param_grid_knn, refit=True, verbose=3)
        grid_knn.fit(X_train, y_train)

        print("Best parameters for KNeighbors: ", grid_knn.best_params_)
        print("Best estimator for KNeighbors: ", grid_knn.best_estimator_)

        # Rede Neural treinamento

        print(f"Iniciando treinamento da rede totalmente conectada para {
              feature_extractor}...")
        input_size = X_train.shape[1]  # Número de características extraídas
        hidden_size = 128
        num_classes = len(np.unique(textLabels))

        # Criação e treinamento do modelo FC
        model_fc = DenseNetwork(config={'model': {'input_shape': (input_size,), 'num_classes': num_classes},
                                        'training': {'batch_size': 32, 'epochs': 100}},
                                X_train=X_train, y_train=y_train,
                                X_val=X_test, y_val=y_test)

        history = model_fc.train()

        # Nome do extrator de características
        extractor_name = ['LBP', 'Gabor', 'GLCM'][contf]

        # Salvando o modelo treinado
        salvar_modelo(model_fc.model, os.path.join(output_folder, f'NN_{extractor_name}{
                      baseFolder.replace("/", "_").replace("\\", "_").replace(":", "_")}{classSize}_model.h5'))

        # Avaliação do modelo treinado
        print(f"Validando a rede totalmente conectada para {
              feature_extractor}...")
        eval_metrics = model_fc.evaluate()

        print(f"Métricas para {feature_extractor} - FC: {eval_metrics}")
        
        # Predição
        

        predictions = model_fc.predict(X_test)
        print("Predições do modelo:", predictions[:10])  # Inspecione as primeiras predições
        
        # Ajustar as predições de acordo com as dimensões
        if predictions.ndim == 2:  # Múltiplas classes
            y_pred = np.argmax(predictions, axis=1)
        elif predictions.ndim == 1:  # Problema binário
            y_pred = (predictions > 0.5).astype(int)
        else:
            raise ValueError("Formato inesperado na saída do modelo.")
       
       
        print("Dimensão de y_test:", y_test.shape)
        print("Dimensão de y_pred:", y_pred.shape)
        print("Valores únicos em y_test:", np.unique(y_test))
        print("Valores únicos em y_pred:", np.unique(y_pred))

        # Chamada corrigida
        salvar_matriz_confusao(
            y_test,
            y_pred,
            output_folder,
            f'NN_{extractor_name}{baseFolder.replace("/", "_").replace("\\", "_").replace(":", "_")}{classSize}.png',
            extractor_name,
            all_matriz,
            name_matriz
        )

        # Predição e avaliação com o melhor modelo de cada classificador
        classifiers = {
            'SVM': grid_svm,
            'RandomForest': grid_rf,
            'KNeighbors': grid_knn
        }

        for name, clf in classifiers.items():
            print(f"Results for {name} for dataset {baseFolder}:")
            y_pred = clf.predict(X_test)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            print(report)
            print(cm)

            # Salvar relatório
            report_path = os.path.join(output_folder, f'{name}_{extractor_name}{
                                       baseFolder.replace("/", "_").replace("\\", "_").replace(":", "_")}{classSize}')
            with open(report_path + '.txt', "w") as file:
                file.write(report)
            print(f"Relatório salvo: {report_path}")

            # Salvar matriz de confusão
            cm_path = report_path + 'confusion_matrix.png'
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix ({name}_{extractor_name})")
            plt.savefig(cm_path)
            plt.close()
            print(f"Matriz de confusão salva: {cm_path}")
            all_matriz.append(cm)
            name_matriz.append(f'{name}_{extractor_name}')

            # Salvar modelo treinado
            model_path = os.path.join(output_folder, report_path+'_model.pkl')
            with open(model_path, "wb") as model_file:
                pickle.dump(clf, model_file)
            print(f"Modelo salvo: {model_path}")


if all_matriz:
    unique_labels = np.unique(y_test)

    # Configurar o número de colunas e calcular o número de linhas
    num_colunas = 4  # Escolha um valor apropriado para o número de colunas
    # Calcula o número de linhas necessário (arredonda para cima)
    num_linhas = -(-len(all_matriz) // num_colunas)

    fig, axes = plt.subplots(num_linhas, num_colunas,
                             figsize=(5 * num_colunas, 5 * num_linhas))

    # Flatten `axes` para garantir que funcione com qualquer número de subplots
    axes = axes.flatten()

    for idx, (ax, cm) in enumerate(zip(axes, all_matriz)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=unique_labels, yticklabels=unique_labels)
        ax.set_title(name_matriz[idx])
        ax.set_xlabel('Predição')
        ax.set_ylabel('Verdadeiro')

    # Esconde eixos vazios se houver menos matrizes que subplots
    for ax in axes[len(all_matriz):]:
        ax.axis('off')

    # Salvar a figura final
    output_path_final = os.path.join(output_folder, "todas_matrizes.png")
    plt.tight_layout()
    plt.savefig(output_path_final)
    plt.close()
    print(f"Imagem única com todas as matrizes salva em: {output_path_final}")
