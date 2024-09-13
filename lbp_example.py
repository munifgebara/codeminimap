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


# LBP
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = local_binary_pattern(image, self.numPoints, self.radius, method="nri_uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist


classSizes = [500]

for classSize in classSizes:

    # Configurações do LBP
    feature_extractor = LocalBinaryPatterns(8, 2)

    images = []
    labels = []
    textLabels = []
    filenames = []

    baseFolder = 'dataset/all_encrypted_fixed/'

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
            image = cv2.imread(os.path.join(baseFolder + folder, image_path), cv2.IMREAD_GRAYSCALE)
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




    # Extração das características LBP
    data = []
    for image in images:
        hist = feature_extractor.describe(image)
        data.append(hist)

    i=0
    for d in data:
        if len(d)!=59:
            print (i,filenames[i],labels[i],textLabels[labels[i]],len(d))
        i=i+1




    data = np.array(data)
    labels = np.array(labels)

    # Divisão do dataset
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=180875)

    # SVM com GridSearchCV
    param_grid_svm = {'C': [1000, 2000, 10000], 'gamma': [2, 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
    grid_svm = GridSearchCV(SVC(), param_grid_svm, refit=True, verbose=3)
    grid_svm.fit(X_train, y_train)

    print("Best parameters for SVM: ", grid_svm.best_params_)
    print("Best estimator for SVM: ", grid_svm.best_estimator_)

    # RandomForest com GridSearchCV
    param_grid_rf = {'n_estimators': [10, 50, 100, 200, 500, 1000], 'max_depth': [None, 10, 20, 30, 100]}
    grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, refit=True, verbose=3)
    grid_rf.fit(X_train, y_train)

    print("Best parameters for RandomForest: ", grid_rf.best_params_)
    print("Best estimator for RandomForest: ", grid_rf.best_estimator_)

    # KNeighbors com GridSearchCV
    param_grid_knn = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19], 'weights': ['uniform', 'distance']}
    grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, refit=True, verbose=3)
    grid_knn.fit(X_train, y_train)

    print("Best parameters for KNeighbors: ", grid_knn.best_params_)
    print("Best estimator for KNeighbors: ", grid_knn.best_estimator_)

    # Predição e avaliação com o melhor modelo de cada classificador
    classifiers = {
        'SVM': grid_svm,
        'RandomForest': grid_rf,
        'KNeighbors': grid_knn
    }

    for name, clf in classifiers.items():
        print(f"Results for {name} for dataset {baseFolder}:")
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        arquivoResultados = f'results/{name}_{baseFolder.replace("/", "_")}{classSize}.txt'

        file = open(arquivoResultados, "w")
        file.write(f"Results for {name} for dataset {baseFolder} {classSize}:\n")
        file.write(classification_report(y_test, y_pred))

        t = 0

        file.write('\n\nconfusion_matrix\n')

        file.write(str(confusion_matrix(y_test, y_pred)))

        file.write('\n\nLabels\n')

        for textLabel in textLabels:
            file.write(f'Label {t}  -> {textLabel}\n')
            t += 1

        file.write('\n\n\n\n\n')

        file.close()

        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {name} {baseFolder} classSize: {classSize}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f'results/confusion_matrix_{name}_{baseFolder.replace("/", "_")}{classSize}.png')
        plt.close()
