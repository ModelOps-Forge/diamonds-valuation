import pandas as pd
import joblib
import logging
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Configuracion de logs para trazabilidad
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Semilla para reproduccion
SEED = 123

def run_training():
    # Configuracion de rutas
    DATA_DIR = Path('./data/processed')
    MODEL_DIR = Path('./models')
    MODEL_DIR.mkdir(exist_ok=True)

    # Cargar datos preprocesados
    logging.info("Cargando datos preprocesados...")
    train_df = pd.read_parquet(DATA_DIR / 'train_preprocessed.parquet')
    test_df = pd.read_parquet(DATA_DIR / 'test_preprocessed.parquet')

    # Separar Features (X) y Target (y)
    X_train = train_df.drop(columns=['precio'])
    y_train = train_df['precio']
    X_test = test_df.drop(columns=['precio'])
    y_test = test_df['precio']

    # Inicializar y Entrenar XGBoost (Configuracion Base)
    logging.info("Entrenando modelo XGBRegressor base...")
    model = XGBRegressor(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=6, 
        random_state=SEED,
        n_jobs=-1  # Usa todos los núcleos disponibles
    )
    
    model.fit(X_train, y_train)

    # Predicciones y Evaluacion
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Reporte de Metricas
    logging.info("-" * 30)
    logging.info(f"Metricas del Modelo:")
    logging.info(f"MAE:  {mae:.2f}")
    logging.info(f"RMSE: {rmse:.2f}")
    logging.info(f"R2:   {r2:.4f}")
    logging.info("-" * 30)

    # 6. Guardar el modelo
    model_path = MODEL_DIR / 'trained_model.joblib'
    joblib.dump(model, model_path)
    logging.info(f"Modelo guardado en: {model_path}")

if __name__ == "__main__":
    run_training()