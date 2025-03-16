# Example of using LBP with different classifiers

Example image classification using local binary patterns (LBP) for feature extraction and different classifiers, including SVM, Random Forest, and KNeighbors, using GridSearchCV to find the best hyperparameters for each classifier.

## Requirements

The following packages are required

- `numpy`
- `opencv-python`
- `scikit-image`
- `scikit-learn`
- `matplotlib`
- `seaborn`


The dataset must be a folder with a subfolder for each class containing images


To create a dataset from the source code of a project, you use the minimaps.py


npm install @tensorflow/tfjs-node


pip install --upgrade tensorflowjs
tensorflowjs_converter --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --skip_op_check \
    /home/munif-gebara-junior/ml/learn/codeminimap/modelos/_home_munif-gebara-junior_ml_learn_codeminimap_dataset_novo_encrypted_68b36a01/saved_model \
    modelos_js/



tensorflowjs_converter --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    /home/munif-gebara-junior/ml/learn/codeminimap/modelos/_home_munif-gebara-junior_ml_learn_codeminimap_dataset_novo_encrypted_68b36a01/saved_model \
    modelos_js/


tensorflowjs_converter --input_format keras --skip_op_check modelos/meu_modelo.h5 modelos_js/
