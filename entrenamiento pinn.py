import os, re, glob, warnings
import numpy as np
import pandas as pd
import tensorflow as tf

# =======================
# RUTAS Y CONFIG
# =======================
CARPETA_TRAIN = r"D:/alejandro/plantas/buck/buck_cruzado_Cout_Rds_1_entrenamiento"

MODELO_PATH   = r"D:/alejandro/plantas/buck/modelo/pinn_buck.keras"
ESCALA_PATH   = r"D:/alejandro/plantas/buck/modelo/pinn_buck_scale.txt"
FREF_PATH     = r"D:/alejandro/plantas/buck/modelo/pinn_buck_fref.npy"
COVINV_PATH   = r"D:/alejandro/plantas/buck/modelo/pinn_buck_cov_inv.npy"
PMU_PATH      = r"D:/alejandro/plantas/buck/modelo/pinn_buck_params_mu.npy"
PSD_PATH      = r"D:/alejandro/plantas/buck/modelo/pinn_buck_params_sd.npy"

GOOD_TOL_PCT  = 5.0
LATENT_DIM    = 16
HIDDEN        = [512, 256]
EPOCHS        = 300
BATCH         = 64
LR            = 1e-3
LAMBDA_PASS   = 1e-2
LAMBDA_SMOOTH = 1e-4
DUTY_DEFAULT  = 0.50   # ~ Vout/Vin

np.set_printoptions(suppress=True, linewidth=140)
tf.keras.utils.set_random_seed(42)

# =======================
# UTILIDADES I/O
# =======================
def listar_txt(carpeta):
    patrones = ["*.txt", "*.TXT", "**/*.txt", "**/*.TXT"]
    archivos = []
    for pat in patrones:
        archivos.extend(glob.glob(os.path.join(carpeta, pat), recursive=True))
    archivos = sorted(list({os.path.abspath(p) for p in archivos if os.path.isfile(p)}))
    return archivos

def leer_curva_formato_amp_rad(path):
    try:
        df = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < 3:
        df = df[pd.to_numeric(df[0], errors="coerce").notnull()]
        if df.shape[1] < 3:
            raise ValueError(f"Formato no esperado en {path}")
    freq = df.iloc[:,0].astype(float).to_numpy()
    amp  = df.iloc[:,1].astype(float).to_numpy()
    ph   = df.iloc[:,2].astype(float).to_numpy()
    re = amp * np.cos(ph)
    im = amp * np.sin(ph)
    mat = np.column_stack([re, im])
    return freq, mat

def cargar_carpeta(carpeta):
    rutas = listar_txt(carpeta)
    if len(rutas) == 0:
        raise RuntimeError(f"No se encontraron .txt en {carpeta}.")
    X_list, F_list, names = [], [], []
    for p in rutas:
        f, mat = leer_curva_formato_amp_rad(p)
        X_list.append(mat); F_list.append(f); names.append(os.path.basename(p))
    return F_list, X_list, names

def reinterpolar_a_malla(X_list, F_list, F_ref):
    Xr = []
    for (F, X) in zip(F_list, X_list):
        if len(F) != len(F_ref) or np.max(np.abs(F - F_ref)) > 1e-12:
            re = np.interp(F_ref, F, X[:,0]); im = np.interp(F_ref, F, X[:,1])
            Xr.append(np.column_stack([re, im]))
        else:
            Xr.append(X)
    return np.stack(Xr, axis=0)  # (N, n_freq, 2)

# =======================
# PARSEO DE NOMBRES
# =======================
COMPONENT_KEYS = {
    "Rds_1":"Rds_1", "Rds1":"Rds_1",
    "Rds_2":"Rds_2", "Rds2":"Rds_2",
    "Lout":"Lout", "Rout":"Rout", "Cout":"Cout",
    "Esr_L":"Esr_L", "ESR_L":"Esr_L",
    "Esr_C":"Esr_C", "ESR_C":"Esr_C",
}
def parse_porcentajes(nombre):
    base = os.path.splitext(nombre)[0]
    partes = base.split("__"); out = {}
    for p in partes:
        m = re.match(r"([A-Za-z0-9]+(?:_[0-9])?)_([+\-]?\d+(?:\.\d+)?)%", p)
        if m:
            key = COMPONENT_KEYS.get(m.group(1), None)
            if key is not None:
                out[key] = float(m.group(2))
    return out

def etiqueta_bueno(nombre, tol=GOOD_TOL_PCT):
    d = parse_porcentajes(nombre)
    if len(d) == 0: return 0
    return 0 if all(abs(v) <= tol for v in d.values()) else 1

# =======================
# ESCALADO Y OMEGA
# =======================
def robust_scale_factor(X):  # X: (N, n_freq, 2)
    mag = np.sqrt(X[...,0]**2 + X[...,1]**2)
    s = np.percentile(mag, 95)
    return float(max(s, 1e-6))

def compute_omega(freq_hz):
    return 2.0*np.pi*freq_hz.astype(np.float32)

# =======================
# CAPA DECODER FÍSICO
# =======================
@tf.keras.utils.register_keras_serializable()
class BuckDecoder(tf.keras.layers.Layer):
    def __init__(self, omega, alpha1=0.5, alpha2=0.5,
                 lambda_pass=0.0, lambda_smooth=0.0, name="buck_decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        omega_np = np.array(omega, dtype=np.float32).reshape(1, -1)
        self._omega_np = omega_np
        self.omega = tf.constant(omega_np, tf.float32)
        self.alpha1 = float(alpha1); self.alpha2 = float(1.0 - alpha1) if alpha2 is None else float(alpha2)
        self.lambda_pass = float(lambda_pass); self.lambda_smooth = float(lambda_smooth)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"omega": self._omega_np.tolist(),
                    "alpha1": self.alpha1, "alpha2": self.alpha2,
                    "lambda_pass": self.lambda_pass, "lambda_smooth": self.lambda_smooth})
        return cfg
    @classmethod
    def from_config(cls, config):
        omega_list = config.pop("omega")
        return cls(omega=np.array(omega_list, dtype=np.float32), **config)
    def compute_output_shape(self, input_shape):
        n_freq = int(self._omega_np.shape[1]); return (input_shape[0], 2*n_freq)
    def call(self, params):
        L, C, EsrL, EsrC, Rds1, Rds2, Rout = tf.split(params, 7, axis=-1)
        soft = tf.nn.softplus
        L=soft(L)+1e-12; C=soft(C)+1e-12
        EsrL=soft(EsrL)+1e-9; EsrC=soft(EsrC)+1e-9
        Rds1=soft(Rds1)+1e-9; Rds2=soft(Rds2)+1e-9; Rout=soft(Rout)+1e-9
        def to_cplx(x): return tf.complex(x, tf.zeros_like(x))
        w=self.omega; jw=tf.complex(tf.zeros_like(w), w)
        Z_L = to_cplx(EsrL) + jw*to_cplx(L)
        Z_C = to_cplx(EsrC) + 1.0/(jw*to_cplx(C))
        Z_R = to_cplx(Rout)
        Z_par = 1.0/(1.0/Z_C + 1.0/Z_R)
        Rs = to_cplx(self.alpha1)*to_cplx(Rds1) + to_cplx(self.alpha2)*to_cplx(Rds2)
        Z_out = Rs + Z_L + Z_par
        ReZ = tf.math.real(Z_out); ImZ = tf.math.imag(Z_out)
        out = tf.concat([ReZ, ImZ], axis=-1)
        if self.lambda_pass>0.0: self.add_loss(self.lambda_pass*tf.reduce_mean(tf.nn.relu(-ReZ)))
        if self.lambda_smooth>0.0:
            def d2(x): return x[:,2:] - 2.0*x[:,1:-1] + x[:,:-2]
            self.add_loss(self.lambda_smooth*(tf.reduce_mean(tf.square(d2(ReZ)))+tf.reduce_mean(tf.square(d2(ImZ)))))
        return out

def build_pinn_buck(n_freq, latent_dim, omega, duty=DUTY_DEFAULT):
    alpha1 = float(np.clip(duty, 0.0, 1.0)); alpha2 = 1.0 - alpha1
    inp = tf.keras.Input(shape=(2*n_freq,), name="x_in")
    x = tf.keras.layers.Dense(HIDDEN[0], activation='relu')(inp)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(HIDDEN[1], activation='relu')(x)
    z = tf.keras.layers.Dense(latent_dim, activation='relu', name="z")(x)
    params = tf.keras.layers.Dense(7, activation='linear', name="params")(z)
    out = BuckDecoder(omega, alpha1=alpha1, alpha2=alpha2,
                      lambda_pass=LAMBDA_PASS, lambda_smooth=LAMBDA_SMOOTH)(params)
    model = tf.keras.Model(inputs=inp, outputs=out, name="PINN_Buck_PD")
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mse')
    return model

# =======================
# ENTRENAMIENTO
# =======================
if __name__ == "__main__":
    # Carga TRAIN
    F_train_list, X_train_list, names_train = cargar_carpeta(CARPETA_TRAIN)
    print(f"[TRAIN] {len(names_train)} curvas leídas.")
    F_ref = F_train_list[0]
    X_train = reinterpolar_a_malla(X_train_list, F_train_list, F_ref)  # (N, n_freq, 2)
    n_freq = X_train.shape[1]
    omega = compute_omega(F_ref)

    # Filtrar SANOS por nombre
    y_train_bin = np.array([etiqueta_bueno(n, GOOD_TOL_PCT) for n in names_train], dtype=int)
    mask_good = (y_train_bin == 0)
    if np.sum(mask_good) == 0:
        warnings.warn("No hay curvas 'buenas' en TRAIN; se usará todo.")
        mask_good = np.ones_like(y_train_bin, dtype=bool)
    X_train_good = X_train[mask_good]

    # Escala global robusta
    scale = robust_scale_factor(X_train_good)
    np.savetxt(ESCALA_PATH, np.array([scale]), fmt="%.6e")
    np.save(FREF_PATH, F_ref.astype(np.float32))
    print(f"[INFO] Escala guardada en {ESCALA_PATH} | F_ref guardada en {FREF_PATH}")

    # Vectorizar / escalar
    Xtr_all = (X_train_good.reshape(len(X_train_good), -1)) / scale

    # Modelo y entrenamiento
    model = build_pinn_buck(n_freq, LATENT_DIM, omega, duty=DUTY_DEFAULT)
    X_tr, X_val = np.split(Xtr_all, [int(0.8*len(Xtr_all))])  # split simple
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True, verbose=1),
    ]
    hist = model.fit(X_tr, X_tr, validation_data=(X_val, X_val),
                     epochs=EPOCHS, batch_size=BATCH, shuffle=True,
                     callbacks=callbacks, verbose=1)
    model.save(MODELO_PATH)
    print("[OK] Modelo guardado:", MODELO_PATH)

    # Artefactos para verificación: Sigma^{-1}, mu/sd de parámetros
    Xtr_pred = model.predict(Xtr_all, verbose=0)
    E_tr     = Xtr_pred - Xtr_all
    cov      = np.cov(E_tr, rowvar=False)
    cov_inv  = np.linalg.pinv(cov + 1e-6*np.eye(cov.shape[0]))
    np.save(COVINV_PATH, cov_inv.astype(np.float32))

    params_model = tf.keras.Model(model.input, model.get_layer("params").output)
    P_tr = params_model.predict(Xtr_all, verbose=0)
    P_mu = P_tr.mean(axis=0); P_sd = P_tr.std(axis=0) + 1e-9
    np.save(PMU_PATH, P_mu.astype(np.float32))
    np.save(PSD_PATH, P_sd.astype(np.float32))
    print(f"Modelo guardado: {COVINV_PATH}, {PMU_PATH}, {PSD_PATH}")