import os, glob, re, warnings
import numpy as np
import pandas as pd
import tensorflow as tf

# =======================
# CONFIGURACIÓN
# =======================
CARPETA_TRAIN = r"D:/alejandro/plantas/buck/buck_cruzado_Cout_EsrC_entrenamiento"
MODELO_PATH   = r"D:/alejandro/plantas/buck/modelo/pinn_buck_duty.keras"
LATENT_DIM    = 16
HIDDEN        = [512, 256]
EPOCHS        = 1000
BATCH         = 128
LR            = 1e-3
DUTY_DEFAULT  = 0.50  # lo dejo por si lo usas más adelante

tf.keras.utils.set_random_seed(42)

# =======================
# UTILIDADES
# =======================
def listar_txt(carpeta):
    return sorted(glob.glob(os.path.join(carpeta, "*.txt")))

def leer_curva(path):
    # Se asume: columnas = [freq_Hz, mag_dB, fase_grados]
    df = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, engine="python")
    freq = df.iloc[:, 0].to_numpy().astype(np.float64)
    mag_db = df.iloc[:, 1].to_numpy().astype(np.float64)
    ph_deg = df.iloc[:, 2].to_numpy().astype(np.float64)

    mag_lin = 10.0 ** (mag_db / 20.0)
    ph_rad  = np.deg2rad(ph_deg)

    re = mag_lin * np.cos(ph_rad)
    im = mag_lin * np.sin(ph_rad)
    return freq, np.column_stack([re, im]).astype(np.float64)

def cargar_carpeta(carpeta):
    rutas = listar_txt(carpeta)
    if not rutas:
        raise FileNotFoundError(f"No se han encontrado .txt en: {carpeta}")
    X_list, F_list, names = [], [], []
    for p in rutas:
        f, mat = leer_curva(p)
        X_list.append(mat)
        F_list.append(f)
        names.append(os.path.basename(p))
    return F_list, X_list, names

def reinterpolar_a_malla(X_list, F_list, F_ref):
    Xr = []
    for F, X in zip(F_list, X_list):
        if len(F) != len(F_ref) or np.max(np.abs(F - F_ref)) > 1e-12:
            re = np.interp(F_ref, F, X[:, 0])
            im = np.interp(F_ref, F, X[:, 1])
            Xr.append(np.column_stack([re, im]))
        else:
            Xr.append(X)
    return np.stack(Xr, axis=0)

# =======================
# Capa física (Buck duty->salida)
# =======================
@tf.keras.utils.register_keras_serializable(package="pinns")
class BuckDutyDecoder(tf.keras.layers.Layer):
    def __init__(self, omega, name="buck_duty_decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        # Guardamos una copia serializable y otra tensorial para cálculo
        omega_np = np.array(omega, dtype=np.float32).reshape(1, -1)
        self._omega_list = omega_np.tolist()               # para get_config
        self.omega = tf.constant(omega_np, dtype=tf.float32)  # para call

    def call(self, params):
        L, C, Rs, Vin, Rload = tf.split(params, 5, axis=-1)
        soft = tf.nn.softplus
        L     = soft(L)     + 1e-12
        C     = soft(C)     + 1e-12
        Rs    = soft(Rs)    + 1e-7
        Vin   = soft(Vin)   + 1e-6
        Rload = soft(Rload) + 1e-6

        w = self.omega                       # (1, n_freq)
        jω = tf.complex(tf.zeros_like(w), w) # (1, n_freq)

        Lc, Cc = tf.cast(L, tf.complex64), tf.cast(C, tf.complex64)
        Rsc, Vinc = tf.cast(Rs, tf.complex64), tf.cast(Vin, tf.complex64)
        Rloadc = tf.cast(Rload, tf.complex64)

        ZL  = jω * Lc
        ZRs = Rsc
        ZC  = 1.0 / (jω * Cc + 1e-30)
        ZR  = Rloadc
        Zpar = (ZC * ZR) / (ZC + ZR + 1e-30)
        Zseries = ZRs + ZL + Zpar
        Gvd = Vinc * (Zpar / (Zseries + 1e-30))

        Re = tf.math.real(Gvd)
        Im = tf.math.imag(Gvd)
        return tf.concat([Re, Im], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "omega": self._omega_list
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Keras pasará 'omega' como lista; la volvemos a np.array en __init__
        return cls(**config)


# =======================
# Construcción PINN
# =======================
def build_pinn_buck_duty(n_freq, latent_dim, omega, duty=DUTY_DEFAULT):
    inp = tf.keras.Input(shape=(2 * n_freq,), name="x_in")
    x = tf.keras.layers.Dense(HIDDEN[0], activation='relu')(inp)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(HIDDEN[1], activation='relu')(x)
    z = tf.keras.layers.Dense(latent_dim, activation='relu', name="z")(x)
    # params = [L, C, Rs, Vin, Rload]
    params = tf.keras.layers.Dense(5, activation='linear', name="params")(z)
    out = BuckDutyDecoder(omega)(params)
    model = tf.keras.Model(inputs=inp, outputs=out, name="PINN_Buck_Duty")
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mse')
    return model

# =======================
# Entrenamiento
# =======================
if __name__ == "__main__":
    F_list, X_list, names = cargar_carpeta(CARPETA_TRAIN)
    F_ref = F_list[0]
    X_train = reinterpolar_a_malla(X_list, F_list, F_ref)  # (N, n_freq, 2)
    n_freq = X_train.shape[1]
    omega = (2 * np.pi * F_ref).astype(np.float32)  # 1D

    # Normalización simple por magnitud (percentil 95)
    mag = np.sqrt(X_train[..., 0]**2 + X_train[..., 1]**2)
    scale = np.percentile(mag, 95)
    X_train_scaled = (X_train.reshape(len(X_train), -1) / (scale if scale > 0 else 1.0)).astype(np.float32)

    model = build_pinn_buck_duty(n_freq, LATENT_DIM, omega, duty=DUTY_DEFAULT)

    # Split 80/20
    n_tr = int(0.8 * len(X_train_scaled))
    X_tr, X_val = X_train_scaled[:n_tr], X_train_scaled[n_tr:]

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True)
    ]

    hist = model.fit(
        X_tr, X_tr,
        validation_data=(X_val, X_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    model.save(MODELO_PATH)
    print("[OK] Modelo guardado:", MODELO_PATH)
