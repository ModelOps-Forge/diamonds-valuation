import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configuracion de logs para trazabilidad
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_evaluation():
    # Configuracion de rutas
    DATA_DIR = Path('./data/processed')
    MODEL_DIR = Path('./models')
    # Carpeta de reportes
    REPORT_DIR = Path('./reports/figures')
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar modelo y datos preprocesados de test
    logging.info("Cargando artefactos para evaluacion...")
    model = joblib.load(MODEL_DIR / 'trained_model.joblib')
    test_df = pd.read_parquet(DATA_DIR / 'test_preprocessed.parquet')

    X_test = test_df.drop(columns=['precio'])
    y_test = test_df['precio']

    # Generar Predicciones
    y_pred = model.predict(X_test)

    # Grafica 1: Prediccion vs Real
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
    plt.title(f'Comparativa: Real vs Predicho\n(R2: {model.score(X_test, y_test):.4f})')
    plt.xlabel('Precio Real (USD)')
    plt.ylabel('Prediccion (USD)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(REPORT_DIR / 'real_vs_predicho.png')
    plt.close()

    # Grafica 2: Distribucion de Errores (Residuales)
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, bins=50, color='purple')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title('Distribucion de Residuales (Errores de Prediccion)')
    plt.xlabel('Error (Precio Real - Predicho)')
    plt.ylabel('Frecuencia')
    plt.savefig(REPORT_DIR / 'distribucion_errores.png')
    plt.close()

    logging.info(f"Evaluacion finalizada. Graficas disponibles en: {REPORT_DIR}")

if __name__ == "__main__":
    run_evaluation()