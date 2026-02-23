import pandas as pd
import joblib
from pathlib import Path
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

def run_preprocessing():
    # Configuracion de rutas
    DATA_DIR = Path('./data/processed')
    MODEL_DIR = Path('./models')
    MODEL_DIR.mkdir(exist_ok=True)

    # Cargar datos
    train_df = pd.read_parquet(DATA_DIR / 'train.parquet')
    test_df = pd.read_parquet(DATA_DIR / 'test.parquet')

    # Definir columnas
    cols_numeric = ['quilate', 'profundidad', 'tabla', 'x', 'y', 'z']
    cols_categoric = ['corte', 'color', 'claridad']
    
    # Ordenes jerarquicos
    cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

    # Construir el Preprocessor
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[cut_order, color_order, clarity_order])),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipe, cols_numeric),
        ("categoric_ordinales", categorical_pipe, cols_categoric),
    ])

    # Ajuste y transformacion
    logging.info("Ajustando y transformando datos con Scikit-Learn Pipeline...")
    # El preprocesador aprende del set de entrenamiento solamente
    X_train_transformed = preprocessor.fit_transform(train_df.drop(columns=['precio']))
    X_test_transformed = preprocessor.transform(test_df.drop(columns=['precio']))

    # Guardar el objeto preprocessor para la prediccion manual
    joblib.dump(preprocessor, MODEL_DIR / 'preprocessor.joblib')
    
    # Guardar los datos transformados para el entrenamiento
    train_processed = pd.DataFrame(X_train_transformed)
    train_processed['precio'] = train_df['precio'].values # Recuperamos el target
    
    train_processed.to_parquet(DATA_DIR / 'train_preprocessed.parquet', index=False)
    logging.info("Preprocesador guardado y datos listos para el entrenamiento.")

if __name__ == "__main__":
    run_preprocessing()