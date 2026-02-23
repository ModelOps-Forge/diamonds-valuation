from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import sys


def run_ingestion():
    """Prepara el dataset de diamantes (CSV) y lo guarda en Parquet con tipos fijos."""
    # Configuración de logs para trazabilidad
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # ruta del directorio de datos
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / 'data' / 'raw'
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ruta del directorio de datos procesados
    DATA_DIR_PRO = BASE_DIR / 'data' / 'processed'
    DATA_DIR_PRO.mkdir(parents=True, exist_ok=True)

    # semilla para reproduccion
    SEED = 42

    # Ruta al CSV de entrada (fall-back si cambia el nombre)
    url_data = DATA_DIR / 'diamonds_train.csv'
    if not url_data.exists():
        print(f"No se encontró {url_data}. Asegúrate de colocar el CSV en {DATA_DIR}/")
        sys.exit(1)

    logging.info("Iniciando ingesta de datos...")
    diamonds_df = pd.read_csv(url_data, low_memory=False)

    logging.info("Realizando division de datos (80/20)...")
    train_df, test_df = train_test_split(diamonds_df, test_size=0.2, random_state=SEED)

    # Guardar ambos en Parquet
    train_df.to_parquet(DATA_DIR_PRO / 'train.parquet', index=False, engine='pyarrow', compression='snappy')
    test_df.to_parquet(DATA_DIR_PRO / 'test.parquet', index=False, engine='pyarrow', compression='snappy')

    logging.info(f"Archivos creados: train.parquet ({len(train_df)} filas) y test.parquet ({len(test_df)} filas)")

if __name__ == "__main__":
    run_ingestion()