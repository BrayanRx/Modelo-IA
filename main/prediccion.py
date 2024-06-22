import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Cargar el modelo
model = load_model('Modelo-IA/main/parametros/modelo_mlp.keras')

# Cargar el scaler y el encoder
scaler = joblib.load('Modelo-IA/main/parametros/scaler.save')
encoder = joblib.load('Modelo-IA/main/parametros/encoder.save')

# Cargar nuevos datos para predicción
df_nuevos_datos = pd.read_csv('Modelo-IA/data/nuevos_datos.csv')

# Asegurarse de que los nombres de las columnas coincidan con los del entrenamiento
df_nuevos_datos.columns = df_nuevos_datos.columns.str.strip()  # Eliminar espacios en blanco en los nombres de columnas

# Seleccionar solo las columnas necesarias
df_nuevos_datos = df_nuevos_datos[['alumno', 'curso-sec', 'Promedio', 'Calculo', 'Estadistica', 'Fisica', 'Quimica', 'Programacion', 'GestionEmp', 'Economia', 'area-interes', 'nota-esperada']]

# Comprobar y agregar categorías desconocidas al encoder
def add_categories(encoder, df, column_name):
    categories = encoder.categories_[0]
    unique_values = df[column_name].unique()
    new_categories = np.setdiff1d(unique_values, categories)
    if new_categories.size > 0:
        encoder.categories_[0] = np.append(categories, new_categories)
        print(f"Añadidas nuevas categorías al encoder: {new_categories}")

# Añadir nuevas categorías al encoder si es necesario
add_categories(encoder, df_nuevos_datos, 'area-interes')
add_categories(encoder, df_nuevos_datos, 'curso-sec')
# Codificar las variables categóricas
encoded_area_interes = encoder.transform(df_nuevos_datos[['area-interes']])
encoded_curso_sec = encoder.transform(df_nuevos_datos[['curso-sec']])

# Concatenar las características numéricas y las codificadas
X_nuevos = np.concatenate([df_nuevos_datos[['Promedio', 'Calculo', 'Estadistica', 'Fisica', 'Quimica', 'Programacion', 'GestionEmp', 'Economia', 'nota-esperada']].values, 
                           encoded_area_interes, encoded_curso_sec], axis=1)

# Escalar las características
X_nuevos = scaler.transform(X_nuevos)

# Predicciones
predicciones = model.predict(X_nuevos)

# Mostrar las predicciones
for i, pred in enumerate(predicciones):
    alumno = df_nuevos_datos.iloc[i]['alumno']
    curso_sec = df_nuevos_datos.iloc[i]['curso-sec']
    print(f'La nota esperada del {alumno} en la sección {curso_sec} es {pred[0]:.2f}')