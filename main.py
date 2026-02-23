import logging
import sys
from src.data_ingestion import run_ingestion

# from src.feature_engineering import run_feature_engineering
from src.preprocessing import run_preprocessing
from src.model_training import run_training

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    try:
        logging.info("--- Iniciando Pipeline ---")
        # 1. Ingesta
        logging.info("Etapa 1: Ingesta de datos...")
        run_ingestion()
        # 2. Ingenieria de caracteristicas
        logging.info("Etapa 2: Feature Engineering... [PENDIENTE]")
        # run_feature_engineering()
        # 3. Preprocesamiento
        logging.info("Etapa 3: Preprocesamiento...")
        run_preprocessing()
        # 4. Optimizacion o Tuneo
        logging.info("Etapa 4: Optimizacion... [PENDIENTE]")
        # run_tuning()
        ### Los parametros no se asignan automatico
        # 5. Entrenamiento
        logging.info("Etapa 5: Entrenamiento...")
        run_training()
        # 6. Evaluacion
        logging.info("Etapa 6: Evaluacion... [PENDIENTE]")
        # run_evaluation()
        logging.info("--- Pipeline ejecutado con éxito ---")

    except Exception as e:
        logging.error(f"El pipeline falló en la etapa crítica: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()