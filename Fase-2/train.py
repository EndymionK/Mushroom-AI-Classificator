import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
import joblib
from datetime import datetime
import os
import argparse

# Configurar el manejo de argumentos de línea de comandos
parser = argparse.ArgumentParser(description='Entrena un modelo XGBoost usando el archivo CSV proporcionado.')
parser.add_argument('train_csv_path', type=str, help='Ruta del archivo CSV de entrenamiento')
args = parser.parse_args()

# Cargar el dataset de entrenamiento
df_train = pd.read_csv(args.train_csv_path)

# Eliminar la columna 'id'
df_train = df_train.drop(columns=['id'])

# Eliminar columnas con más del 95% de valores faltantes
missing_threshold = 0.95
high_missing_columns = df_train.columns[df_train.isnull().mean() > missing_threshold]
df_train = df_train.drop(columns=high_missing_columns)
gc.collect()

# Imputación de valores faltantes usando KNN
def knn_impute(df, n_neighbors=5):
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(knn_imputer.fit_transform(df_encoded), columns=df_encoded.columns)
    for col in df.select_dtypes(include='object').columns:
        df_imputed[col] = df_imputed[col].round().astype(int).map(
            dict(enumerate(df[col].astype('category').cat.categories)))
    return df_imputed

df_train_imputed = knn_impute(df_train, n_neighbors=5)

# Codificación ordinal para variables categóricas
cat_cols_train = df_train_imputed.select_dtypes(include=['object']).columns
cat_cols_train = cat_cols_train[cat_cols_train != 'class']
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df_train_imputed[cat_cols_train] = ordinal_encoder.fit_transform(df_train_imputed[cat_cols_train].astype(str))

# Codificación de la variable objetivo
le = LabelEncoder()
df_train_imputed['class'] = le.fit_transform(df_train_imputed['class'])

# Separación en características y objetivo
y = df_train_imputed['class']
X = df_train_imputed.drop(['class'], axis=1)

# División del conjunto de datos en entrenamiento y prueba
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entrenamiento del modelo XGBoost
model = XGBClassifier(
    alpha=0.1,
    subsample=0.8,
    colsample_bytree=0.6,
    objective='binary:logistic',
    max_depth=14,
    min_child_weight=7,
    gamma=1e-6,
    n_estimators=100
)
XGB = model.fit(train_X, train_y, eval_set=[(test_X, test_y)], verbose=False)

# Guardar el modelo entrenado en la misma carpeta que el archivo de entrenamiento
output_dir = os.path.dirname(args.train_csv_path)
model_filename = os.path.join(output_dir, f'Fase-2/xgb_model.joblib')
joblib.dump(XGB, model_filename)

print(f'Modelo guardado como {model_filename}')
