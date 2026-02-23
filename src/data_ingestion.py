from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import sys

# Configuracion de logs para trazabilidad
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Ruta del directorio de datos
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Ruta del directorio de datos procesados
DATA_DIR_PRO = BASE_DIR / 'data' / 'processed'
DATA_DIR_PRO.mkdir(parents=True, exist_ok=True)

# Semilla para reproduccion
SEED = 42

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Logica de limpieza tecnica y tipado inicial."""
    logging.info("Iniciando limpieza tecnica y renombrado...")
    
    # Renombrar
    new_names = {
        'carat': 'quilate', 'cut': 'corte', 'color': 'color', 
        'clarity': 'claridad', 'depth': 'profundidad', 'table': 'tabla', 
        'price': 'precio', 'x': 'x', 'y': 'y', 'z': 'z'
    }
    df = df.rename(columns=new_names)

    # Duplicados
    initial_rows = len(df)
    df = df.drop_duplicates()
    rows_lost = initial_rows - len(df)
    if rows_lost > 0:
        logging.warning(f"Se eliminaron {rows_lost} filas duplicadas.")

    # Dimensiones validas
    mask_invalid = (df[['x', 'y', 'z']].isna().any(axis=1)) | (df[['x', 'y', 'z']] <= 0).any(axis=1)
    if mask_invalid.any():
        logging.warning(f"Eliminando {mask_invalid.sum()} filas con dimensiones invalidas.")
        df = df.loc[~mask_invalid].copy()

    # Tipado optimizado
    cols_numeric_float = ['quilate', 'profundidad', 'tabla', 'x', 'y', 'z']
    df[cols_numeric_float] = df[cols_numeric_float].astype('float32')
    df['precio'] = df['precio'].astype('int32')
    
    return df

def run_ingestion():
    """Prepara el dataset de diamantes (CSV) y lo guarda en Parquet con tipos fijos."""
    # Ruta al CSV de entrada (fall-back si cambia el nombre)
    url_data = DATA_DIR / 'diamonds_train.csv'
    if not url_data.exists():
        print(f"No se encontró {url_data}. Asegúrate de colocar el CSV en {DATA_DIR}/")
        sys.exit(1)

    logging.info("Iniciando ingesta de datos...")
    diamonds_df = pd.read_csv(url_data, low_memory=False)

    # Aplicar limpieza
    diamonds_df = clean_data(diamonds_df)

    logging.info("Realizando division de datos (80/20)...")
    train_df, test_df = train_test_split(diamonds_df, test_size=0.2, random_state=SEED)

    # Guardar ambos en Parquet
    train_df.to_parquet(DATA_DIR_PRO / 'train.parquet', index=False, engine='pyarrow', compression='snappy')
    test_df.to_parquet(DATA_DIR_PRO / 'test.parquet', index=False, engine='pyarrow', compression='snappy')

    logging.info(f"Archivos creados: train.parquet ({len(train_df)} filas) y test.parquet ({len(test_df)} filas)")

if __name__ == "__main__":
    run_ingestion()