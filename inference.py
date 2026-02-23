import pandas as pd
import joblib
from pathlib import Path

def format_output(data, price):
    """Crea una salida estética en consola."""
    print("\n" + "="*45)
    print(" 💎 RESUMEN DE TASACIÓN DE DIAMANTE 💎 ")
    print("="*45)
    
    # Formateo de tabla simple para los inputs
    print(f"{'CARACTERÍSTICA':<20} | {'VALOR':<20}")
    print("-" * 45)
    for key, value in data.items():
        # Capitalizamos y quitamos guiones bajos para que se vea mejor
        label = key.replace('_', ' ').capitalize()
        print(f"{label:<20} | {value:<20}")
    
    print("-" * 45)
    # Resultado final resaltado
    print(f"{'PRECIO ESTIMADO:':<20} | \033[1;32m${price:,.2f} USD\033[0m")
    print("="*45 + "\n")

def predict_price(data_dict):
    MODEL_DIR = Path('./models')
    try:
        preprocessor = joblib.load(MODEL_DIR / 'preprocessor.joblib')
        model = joblib.load(MODEL_DIR / 'trained_model.joblib')
    except FileNotFoundError:
        print("\n❌ Error: No se encontraron los modelos. Ejecuta 'python main.py' primero.")
        return None

    df_input = pd.DataFrame([data_dict])
    X_processed = preprocessor.transform(df_input)
    prediction = model.predict(X_processed)
    
    return prediction[0]

if __name__ == "__main__":
    # Datos de prueba
    nuevo_diamante = {
        'quilate': 0.7,
        'corte': 'Ideal',
        'color': 'E',
        'claridad': 'SI1',
        'profundidad': 61.5,
        'tabla': 55.0,
        'x': 5.72, 'y': 5.75, 'z': 3.52
    }
    
    precio = predict_price(nuevo_diamante)
    if precio:
        format_output(nuevo_diamante, precio)