import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import xgboost as xgb

# calcular métricas de desempenho
def get_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# caminhos
FEATURES_CSV_PATH = 'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic-haralick/etc/features.csv'
MODEL_FILE_PATH = 'C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic-haralick/etc/best_xgboost_model.dat'

# carregar dados do arquivo CSV
if os.path.exists(FEATURES_CSV_PATH):
    data = pd.read_csv(FEATURES_CSV_PATH, delimiter=';')
else:
    raise FileNotFoundError(f"O arquivo CSV não foi encontrado em: {FEATURES_CSV_PATH}")

# SMOTE para lidar com desbalanceamento
X = data.drop('Label', axis=1)
y = data['Label'].values
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# conjunto de treinamento e teste 
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Parâmetros do modelo XGBoost
param_grid = {
    'max_depth': [10, 20, 30],
    'min_child_weight': [20, 40],
    'gamma': [0.2, 0.4],
    'subsample': [0.9, 1.5],
    'colsample_bytree': [0.9, 1.5],
    'eta': [0.1, 0.5],
    'n_estimators': [400, 600, 800],
}

# configurar e realizar busca em grade
xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='merror', random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=0, n_jobs=-1)

# melhores parâmetros para treinar o modelo
grid_search.fit(X_train, y_train)
best_xgb_model = grid_search.best_estimator_

# avaliar o modelo
y_pred_best_xgb = best_xgb_model.predict(X_test)
metrics_best_xgb = get_metrics(y_test, y_pred_best_xgb)

# exibir métricas
print("\nMétricas para XGBoost:")
for metric, value in metrics_best_xgb.items():
    print(f"{metric}: {value}")

print("\nTreinamento e avaliação do modelo concluídos com sucesso.")

# salvar o modelo
with open(MODEL_FILE_PATH, 'wb') as model_file:
    pickle.dump(best_xgb_model, model_file)
