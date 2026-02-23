import logging
import sys
from src.data_ingestion import run_ingestion

# from src.feature_engineering import run_feature_engineering

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

        # 2. Feature Engineering (Placeholder para tu próximo commit)
        logging.info("Etapa 2: Feature Engineering... [PENDIENTE]")
        # run_feature_engineering()

        logging.info("--- Pipeline ejecutado con éxito ---")

    except Exception as e:
        logging.error(f"El pipeline falló en la etapa crítica: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()