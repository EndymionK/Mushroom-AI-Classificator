import argparse
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer

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

def preprocess_data(df, categorical_columns):
    df_imputed = knn_impute(df)
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_imputed[categorical_columns] = ordinal_encoder.fit_transform(df_imputed[categorical_columns].astype(str))
    return df_imputed

def main(csv_file):
    # Cargar el modelo
    model_path = './Fase-2/xgb_model_20240824_1820.joblib'
    model = joblib.load(model_path) 

    # Leer el archivo CSV
    df = pd.read_csv(csv_file)

    # Eliminar la columna 'id' si existe y almacenar 'id' para el archivo de salida
    id_column = None
    if 'id' in df.columns:
        id_column = df['id']
        df = df.drop(columns=['id'])
    
    # Preprocesar datos
    categorical_columns = df.select_dtypes(include=['object']).columns
    df_processed = preprocess_data(df, categorical_columns)
    
    # Realizar predicciones
    y_pred = model.predict(df_processed)
    
    # Crear el archivo de salida con 'id' y 'prediction'
    if id_column is not None:
        output_df = pd.DataFrame({'id': id_column, 'prediction': y_pred})
    else:
        output_df = pd.DataFrame({'prediction': y_pred})
    
    output_file = 'predictions.csv'
    output_df.to_csv(output_file, index=False)
    print(f'Predicciones guardadas en {output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicción usando un modelo previamente almacenado.')
    parser.add_argument('csv_file', type=str, help='Ruta al archivo CSV de entrada.')
    args = parser.parse_args()
    main(args.csv_file)
