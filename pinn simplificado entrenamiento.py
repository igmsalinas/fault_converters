# CRUZADO COUT - ESR_C (Autoencoder + PINN)
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
carpeta_entrenamiento = r"D:/alejandro/plantas/buck/buck_cruzado_Cout_Rds_1_entrenamiento"
carpeta_verificacion  = r"D:/alejandro/plantas/buck/buck_cruzado_Cout_Rds_1_predecir"
modelo_path =           r"D:/alejandro/plantas/buck/modelo/autoencoder_Cout_Rds_1_simplificado.keras"
scaler_path =           r"D:/alejandro/plantas/buck/modelo/autoencoder_Cout_Rds_1_simplificado.pkl"

# =======================
# LECTURA Y PREPROCESADO
# =======================
def procesar_archivo_complejo(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]
        frecuencias, ganancia_db, fase_deg = [], [], []
        for line in lines:
            vals = line.strip().split()
            if len(vals) == 3:
                frecuencias.append(float(vals[0]))
                ganancia_db.append(float(vals[1]))
                fase_deg.append(float(vals[2]))
    ganancia_lineal = 10 ** (np.array(ganancia_db) / 20.0)
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
    if not datos:
        raise RuntimeError(f"No hay .txt en {carpeta}")
    # comprobación rápida de que todas las curvas tienen misma longitud
    n0 = len(datos[0])
    if any(len(d)!=n0 for d in datos):
        raise ValueError("Las curvas no tienen la misma malla de frecuencias. "
                         "Exporta con el mismo AC sweep o unificar la malla.")
    return datos, nombres

def escalar_datos_complejos(data_list):
    all_data = np.concatenate([d[["Re(Z)", "Im(Z)"]].values for d in data_list], axis=0)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(all_data)
    datos_escalados = [scaler.transform(d[["Re(Z)", "Im(Z)"]].values) for d in data_list]
    return datos_escalados, scaler

# =======================
# DECODER FÍSICO (PINN)
# =======================
@tf.keras.utils.register_keras_serializable()
class BuckDecoder(tf.keras.layers.Layer):
    """
    Decoder físico para Buck en dominio frecuencia:
      Z_L(jw) = ESR_L + j w L
      Z_C(jw) = ESR_C + 1/(j w C)
      Z_par   = Z_C || Rout
      Rs      = a1*Rds1 + a2*Rds2
      Z_out   = Rs + Z_L + Z_par
    Produce [Re(Z_out), Im(Z_out)]
    """
    def __init__(self, omega, alpha1=0.5, alpha2=0.5, name="buck_decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        omega_np = np.array(omega, dtype=np.float32).reshape(1, -1)
        self._omega_np = omega_np
        self.omega = tf.constant(omega_np, tf.float32)
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "omega": self._omega_np.tolist(),
            "alpha1": self.alpha1,
            "alpha2": self.alpha2,
            "name": self.name,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        omega_list = config.pop("omega")
        return cls(omega=np.array(omega_list, dtype=np.float32), **config)

    def call(self, params):
        # Θ = [L, C, ESR_L, ESR_C, Rds1, Rds2, Rout]
        L, C, EsrL, EsrC, Rds1, Rds2, Rout = tf.split(params, 7, axis=-1)

        # Positividad (softplus) para parámetros físicos
        soft = tf.nn.softplus
        L    = soft(L)    + 1e-12
        C    = soft(C)    + 1e-12
        EsrL = soft(EsrL) + 1e-9
        EsrC = soft(EsrC) + 1e-9
        Rds1 = soft(Rds1) + 1e-9
        Rds2 = soft(Rds2) + 1e-9
        Rout = soft(Rout) + 1e-9

        def to_cplx(x): return tf.complex(x, tf.zeros_like(x))

        w  = self.omega                      # (1, n_freq)
        jw = tf.complex(tf.zeros_like(w), w)

        Z_L = to_cplx(EsrL) + jw * to_cplx(L)
        Z_C = to_cplx(EsrC) + 1.0 / (jw * to_cplx(C))
        Z_R = to_cplx(Rout)
        Z_par = 1.0 / (1.0 / Z_C + 1.0 / Z_R)

        Rs = to_cplx(self.alpha1) * to_cplx(Rds1) + to_cplx(self.alpha2) * to_cplx(Rds2)

        Z_out = Rs + Z_L + Z_par
        ReZ = tf.math.real(Z_out)
        ImZ = tf.math.imag(Z_out)
        return tf.concat([ReZ, ImZ], axis=-1)  # (batch, 2*n_freq)

# =======================
# MODELO: ENCODER + Θ + DECODER FÍSICO + (AFÍN MinMax)
# =======================
def construir_pinn_autoencoder(input_dim, omega, scale_vec, min_vec, latent_dim=128):
    n_freq = input_dim // 2
    inp = tf.keras.layers.Input(shape=(input_dim,), name="x_in")

    # Encoder
    x = tf.keras.layers.Dense(1024, activation='relu')(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    z = tf.keras.layers.Dense(latent_dim, activation='relu', name='z')(x)

    # Capa de parámetros físicos Θ
    params = tf.keras.layers.Dense(7, activation='linear', name='params')(z)

    # Decoder físico
    dec = BuckDecoder(omega=omega, alpha1=0.5, alpha2=0.5)
    y_unscaled = dec(params)  # (batch, 2*n_freq)

    # Mapear al espacio MinMax(-1,1) con coeficientes del scaler
    scale_tf = tf.constant(scale_vec.reshape(1, -1).astype(np.float32))
    min_tf   = tf.constant(min_vec.reshape(1, -1).astype(np.float32))
    y_scaled = tf.keras.layers.Lambda(lambda t: t * scale_tf + min_tf, name="to_minmax")(y_unscaled)
    output_shape=(input_dim,)
    model = tf.keras.models.Model(inp, y_scaled, name="AE_PINN_Buck")
    model.compile(optimizer='adam', loss='mse')
    return model

# =======================
# CARGA Y ENTRENAMIENTO
# =======================
datos_entrenamiento, _ = cargar_datos_carpeta_complejo(carpeta_entrenamiento)

# Vector de frecuencias de referencia y omega
freq_ref = datos_entrenamiento[0]["Frecuencia"].to_numpy(dtype=float)
omega = 2.0 * np.pi * freq_ref.astype(np.float32)
# Construye arrays escalados y scaler
datos_escalados_entrenamiento, scaler = escalar_datos_complejos(datos_entrenamiento)

# Matriz X (apilando y vectorizando Re/Im)
X = np.array([curva.reshape(-1) for curva in datos_escalados_entrenamiento], dtype=np.float32)
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

# Vectores del MinMaxScaler para llevar salida a [-1,1]:
# X_scaled = X * scale_ + min_
scale_vec = scaler.scale_   # shape = (2*n_freq,)
min_vec   = scaler.min_     # shape = (2*n_freq,)

# Construir modelo
autoencoder = construir_pinn_autoencoder(
    
    input_dim=X.shape[1],
    omega=omega,
    scale_vec=scale_vec,
    min_vec=min_vec,
    latent_dim=128
)

# callback = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
autoencoder.fit(X_train, X_train,
                epochs=1000,
                batch_size=64,
                validation_data=(X_val, X_val),
                # callbacks=[callback],
                shuffle=True)

# Guardar modelo y scaler
autoencoder.save(modelo_path)
joblib.dump(scaler, scaler_path)

print("Modelo guardado en:", modelo_path)
print("Scaler guardado en:", scaler_path)
