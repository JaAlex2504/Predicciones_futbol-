import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. CARGA DE DATOS
URL_DATOS_GITHUB = "https://github.com/JaAlex2504/Predicciones_futbol-/blob/e2c6e1a3c336ffe4b1e7e35071124ff7ef24a701/data/melb_data.csv"

try:
    df = pd.read_csv(URL_DATOS_GITHUB)

    # SELECCIÓN DE VARIABLES MÚLTIPLES
    # Elegimos columnas numéricas que influyen en el precio
    columnas_input = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]

    # Limpieza: Eliminamos filas que tengan valores vacíos en estas columnas
    df_limpio = df[columnas_input + ["Price"]].dropna()

    X = df_limpio[columnas_input]
    y = df_limpio["Price"]

except Exception as e:
    print(f"Error: {e}")
    exit()

# 2. DIVISIÓN Y ENTRENAMIENTO
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo_multiple = LinearRegression()
modelo_multiple.fit(X_train, y_train)

# 3. EVALUACIÓN
predicciones = modelo_multiple.predict(X_test)
nuevo_r2 = r2_score(y_test, predicciones)

print(f"--- RESULTADOS DEL MODELO MEJORADO ---")
print(f"Variables usadas: {columnas_input}")
print(f"Nuevo coeficiente R²: {nuevo_r2:.4f}")

# 4. COMPARATIVA INTERESANTE
# Vamos a ver la predicción vs la realidad para la primera casa del examen
print(f"\nEjemplo de predicción:")
print(f"Precio Real: ${y_test.iloc[0]:,.2f}")
print(f"Precio Predicho: ${predicciones[0]:,.2f}")
