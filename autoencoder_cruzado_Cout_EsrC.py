#CRUZADO COUT - ESR_C
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib

# === RUTAS DE DATOS ===
carpeta_entrenamiento = "D:/alejandro/plantas/buck/buck_cruzado_Cout_EsrC_entrenamiento"
carpeta_verificacion  = "D:/alejandro/plantas/buck/buck_cruzado_Cout_EsrC_predecir"
modelo_path =           "D:/alejandro/plantas/buck/modelo/autoencoder_Cout_EsrC.keras"
scaler_path =           "D:/alejandro/plantas/buck/modelo/autoencoder_Cout_EsrC.pkl"

# === FUNCIONES ===
def procesar_archivo_complejo(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]
        frecuencias, ganancia_db, fase_deg = [], [], []
        for line in lines:
            valores = line.strip().split()
            if len(valores) == 3:
                frecuencias.append(float(valores[0]))
                ganancia_db.append(float(valores[1]))
                fase_deg.append(float(valores[2]))
        ganancia_lineal = 10 ** (np.array(ganancia_db) / 20)
        fase_rad = np.radians(fase_deg)
        Z_complex = ganancia_lineal * np.exp(1j * fase_rad)
        return pd.DataFrame({
            "Frecuencia": frecuencias,
            "Re(Z)": np.real(Z_complex),
            "Im(Z)": np.imag(Z_complex)
        })

def cargar_datos_carpeta_complejo(carpeta):
    datos, nombres = [], []
    for archivo in sorted(os.listdir(carpeta)):
        if archivo.endswith(".txt"):
            path = os.path.join(carpeta, archivo)
            datos.append(procesar_archivo_complejo(path))
            nombres.append(archivo)
    return datos, nombres

def escalar_datos_complejos(data_list):
    all_data = np.concatenate([d[["Re(Z)", "Im(Z)"]].values for d in data_list], axis=0)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(all_data)
    datos_escalados = [scaler.transform(d[["Re(Z)", "Im(Z)"]].values) for d in data_list]
    return datos_escalados, scaler

def construir_autoencoder(input_dim):
    inp = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(1024, activation='relu')(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    encoded = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(encoded)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x) 
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    decoded = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    model = tf.keras.models.Model(inp, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

# === CARGA Y ENTRENAMIENTO ===
datos_entrenamiento, _ = cargar_datos_carpeta_complejo(carpeta_entrenamiento)
datos_escalados_entrenamiento, scaler = escalar_datos_complejos(datos_entrenamiento)
X = np.array([curva.reshape(-1) for curva in datos_escalados_entrenamiento])
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

autoencoder = construir_autoencoder(X.shape[1])
#callback = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
autoencoder.fit(X_train, X_train,
                epochs=1000,
                batch_size=64,
                validation_data=(X_val, X_val),
                #callbacks=[callback],
                shuffle=True)

autoencoder.save(modelo_path)
joblib.dump(scaler, scaler_path)