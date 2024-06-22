import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
import joblib

# Cargar los datos
df = pd.read_csv('Modelo-IA/data/entrenamiento_final.csv')

# Separar las características y la etiqueta
X = df.drop(columns=['nota-final'])
y = df['nota-final']

# Codificar las variables categóricas
encoder = OneHotEncoder(sparse_output=False)
encoded_area_interes = encoder.fit_transform(X[['area-interes']])
encoded_curso_sec = encoder.fit_transform(X[['curso-sec']])

# Guardar el encoder
joblib.dump(encoder, 'Modelo-IA/main/parametros/encoder.save')

# Concatenar las características numéricas y las codificadas
X = np.concatenate([X[['Promedio', 'Calculo', 'Estadistica', 'Fisica', 'Quimica', 'Programacion', 'GestionEmp', 'Economia', 'nota-esperada']].values,
                    encoded_area_interes, encoded_curso_sec], axis=1)
# Escalar las características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Guardar el scaler
joblib.dump(scaler, 'Modelo-IA/main/parametros/scaler.save')

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo MLP con Dropout para regularización la tasa de aprendizaje 0.001
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
optimizer = SGD(learning_rate=0.001, momentum=0.9)
model.compile(optimizer, loss='mse')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=2, callbacks=[early_stopping, reduce_lr])

# Guardar el modelo
model.save('Modelo-IA/main/parametros/modelo_mlp.keras')

# Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error: {loss}')

# Realizar predicciones de prueba
predicciones = model.predict(X_test)
for i in range(len(predicciones)):
    if predicciones[i][0] < 0:
        predicciones[i][0] = 0
    elif predicciones[i][0] > 20:
        predicciones[i][0] = 20
        
    print(f'Predicción: {predicciones[i][0]}, Real: {y_test.iloc[i]}')