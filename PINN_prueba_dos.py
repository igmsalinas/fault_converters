import os, glob, re, warnings
import numpy as np
import pandas as pd
import tensorflow as tf

# Métricas
try:
    from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_fscore_support, accuracy_score
    SK_OK = True
except Exception:
    SK_OK = False
    warnings.warn("scikit-learn no disponible. Se omiten métricas supervisadas.")

import matplotlib.pyplot as plt

# =======================
# CONFIG
# =======================
CARPETA_TRAIN = r"D:/alejandro/plantas/buck/buck_cruzado_Cout_EsrC_entrenamiento"
CARPETA_TEST  = r"D:/alejandro/plantas/buck/buck_cruzado_Cout_EsrC_predecir"  # cambia si tienes test separado
MODELO_PATH   = r"D:/alejandro/plantas/buck/modelo/pinn_buck_duty.keras"
RESULTS_CSV   = r"D:/alejandro/plantas/buck/modelo/eval_resultados.csv"
SEED          = 42

tf.keras.utils.set_random_seed(SEED)

# =======================
# Capa física (para cargar el modelo)
# =======================
@tf.keras.utils.register_keras_serializable(package="pinns")
class BuckDutyDecoder(tf.keras.layers.Layer):
    def __init__(self, omega, name="buck_duty_decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        omega_np = np.array(omega, dtype=np.float32).reshape(1, -1)
        self._omega_list = omega_np.tolist()
        self.omega = tf.constant(omega_np, dtype=tf.float32)

    def call(self, params):
        L, C, Rs, Vin, Rload = tf.split(params, 5, axis=-1)
        soft = tf.nn.softplus
        L     = soft(L)     + 1e-12
        C     = soft(C)     + 1e-12
        Rs    = soft(Rs)    + 1e-7
        Vin   = soft(Vin)   + 1e-6
        Rload = soft(Rload) + 1e-6

        w = self.omega
        jω = tf.complex(tf.zeros_like(w), w)

        Lc     = tf.cast(L,     tf.complex64)
        Cc     = tf.cast(C,     tf.complex64)
        Rsc    = tf.cast(Rs,    tf.complex64)
        Vinc   = tf.cast(Vin,   tf.complex64)
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
        cfg = super().get_config()
        cfg.update({"omega": self._omega_list})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# =======================
# Utilidades I/O y preproceso
# =======================
def listar_txt(carpeta):
    return sorted(glob.glob(os.path.join(carpeta, "*.txt")))

def leer_curva(path):
    # columnas = [freq_Hz, mag_dB, fase_grados]; primera fila cabecera (skiprows=1)
    df = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, engine="python")
    freq  = df.iloc[:, 0].to_numpy().astype(np.float64)
    magdb = df.iloc[:, 1].to_numpy().astype(np.float64)
    phdeg = df.iloc[:, 2].to_numpy().astype(np.float64)
    mag   = 10.0 ** (magdb / 20.0)
    ph    = np.deg2rad(phdeg)
    re = mag * np.cos(ph)
    im = mag * np.sin(ph)
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

def flatten_ri(X):
    N, n, two = X.shape
    assert two == 2
    return X.reshape(N, 2 * n)

# =======================
# Tu criterio de etiquetado
# =======================
PATRON = re.compile(r"Cout_([+-]?\d+(?:\.\d+)?)%__Esr_C_([+-]?\d+(?:\.\d+)?)%\.txt", re.IGNORECASE)

def parse_cout_esrc(fn):
    m = PATRON.search(fn)  # search (más robusto que match)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))

def construir_BUENOS_desde_train(carpeta_train):
    # 1) Si existe BUENOS.txt, usarlo
    path_txt = os.path.join(carpeta_train, "BUENOS.txt")
    if os.path.isfile(path_txt):
        with open(path_txt, "r", encoding="utf-8") as f:
            return set([ln.strip() for ln in f if ln.strip()])

    # 2) Si no, construir por regla ±5% para Cout y Esr_C
    buenos = set()
    for fn in listar_txt(carpeta_train):
        name = os.path.basename(fn)
        c,e = parse_cout_esrc(name)
        if c is None:
            continue
        if abs(c) <= 5.0 and abs(e) <= 5.0:
            buenos.add(name)
    return buenos

def clasificar_archivo(fn, BUENOS):
    # => 0: bueno, 1: leve (uno fuera), 2: crítico (ambos fuera)
    name = os.path.basename(fn)
    if name in BUENOS:
        return 0
    c,e = parse_cout_esrc(name)
    if c is not None:
        fuera_cout = abs(c) > 5.0
        fuera_esrc = abs(e) > 5.0
        if fuera_cout and fuera_esrc:
            return 2
        elif fuera_cout or fuera_esrc:
            return 1
    return 1  # por defecto

# =======================
# Clasificación por umbrales desde MSE
# =======================
def assign_3class(err, t1, t2):
    y = np.zeros_like(err, dtype=int)
    y[err > t1] = 1
    y[err > t2] = 2
    return y

def grid_search_3class(err, y_true, p1_grid=None, p2_grid=None):
    if not SK_OK:
        return None, None, None, None
    if p1_grid is None:
        p1_grid = np.linspace(60, 99.5, 40)
    if p2_grid is None:
        p2_grid = np.linspace(70, 99.9, 50)
    best = (-1.0, None, None, None, None)  # score, t1, t2, y_pred, cm
    for p1 in p1_grid:
        t1 = np.percentile(err, p1)
        for p2 in p2_grid:
            t2 = np.percentile(err, p2)
            if t2 <= t1:
                continue
            y_pred = assign_3class(err, t1, t2)
            score = f1_score(y_true, y_pred, average="macro")
            if score > best[0]:
                cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
                best = (score, t1, t2, y_pred, cm)
    _, t1, t2, y_pred, cm = best
    if t1 is None:
        return None, None, None, None
    return t1, t2, y_pred, cm

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    # 1) Cargar datos train para F_ref y escala
    F_train_list, X_train_list, names_train = cargar_carpeta(CARPETA_TRAIN)
    F_ref = F_train_list[0]
    X_train = reinterpolar_a_malla(X_train_list, F_train_list, F_ref)  # (Ntr, n_freq, 2)
    n_freq = X_train.shape[1]

    # Escala por percentil 95 (igual que en entrenamiento)
    mag_tr = np.sqrt(X_train[...,0]**2 + X_train[...,1]**2)
    scale = np.percentile(mag_tr, 95)
    if scale <= 0: scale = 1.0
    X_train_flat   = flatten_ri(X_train).astype(np.float32)
    X_train_scaled = (X_train_flat / scale).astype(np.float32)

    # 2) Cargar test y reinterpolar a la malla de train
    F_test_list, X_test_list, names_test = cargar_carpeta(CARPETA_TEST)
    X_test = reinterpolar_a_malla(X_test_list, F_test_list, F_ref)
    X_test_flat   = flatten_ri(X_test).astype(np.float32)
    X_test_scaled = (X_test_flat / scale).astype(np.float32)

    # 3) Cargar modelo
    model = tf.keras.models.load_model(MODELO_PATH)

    # 4) Inferencia
    Xhat_tr_scaled = model.predict(X_train_scaled, verbose=0)
    Xhat_te_scaled = model.predict(X_test_scaled,  verbose=0)
    Xhat_tr = (Xhat_tr_scaled * scale).astype(np.float32)
    Xhat_te = (Xhat_te_scaled * scale).astype(np.float32)

    # 5) Error MSE por muestra
    def mse_per_sample(y_true_flat, y_pred_flat, nfreq):
        y_true = y_true_flat.reshape(-1, nfreq, 2)
        y_pred = y_pred_flat.reshape(-1, nfreq, 2)
        err = (y_true - y_pred)**2
        err = err.sum(axis=2).mean(axis=1)
        return err

    err_tr = mse_per_sample(X_train_flat, Xhat_tr, n_freq)
    err_te = mse_per_sample(X_test_flat,  Xhat_te,  n_freq)

    # 6) Construir BUENOS e inferir y_true con TU REGLA
    BUENOS = construir_BUENOS_desde_train(CARPETA_TRAIN)
    y_true = np.array([clasificar_archivo(nm, BUENOS) for nm in names_test], dtype=int)

    # 7) Ajuste de umbrales para 3 clases y métricas
    t1, t2, y_pred, cm = grid_search_3class(err_te, y_true)
    if t1 is None:
        # Sin sklearn: usa percentiles de train como fallback
        t1 = np.percentile(err_tr, 95.0)
        t2 = np.percentile(err_tr, 99.0)
        y_pred = assign_3class(err_te, t1, t2)
        cm = None
        print("Aviso: scikit-learn no disponible. Umbrales por p95/p99 de train.")

    print(f"\nUmbrales óptimos (3 clases): t1={t1:.4e}, t2={t2:.4e}")
    if SK_OK:
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2], zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        print("\nMatriz de confusión (filas=real, cols=pred):")
        print(cm)
        print("\nMétricas por clase [0,1,2]:")
        for i, (p, r, f, s) in enumerate(zip(prec, rec, f1, sup)):
            print(f"Clase {i}: P={p:.3f}  R={r:.3f}  F1={f:.3f}  Soporte={s}")
        print(f"\nAccuracy={acc:.3f}  Macro-F1={macro_f1:.3f}")

    # 8) También binario (bueno vs no-bueno), por si te interesa
    if SK_OK:
        y_true_bin = (y_true == 0).astype(int)
        # barrido de percentiles para umbral único
        best = (-1, None, None)
        for p in np.linspace(60, 99.9, 80):
            t = np.percentile(err_te, p)
            y_pred_bin = (err_te <= t).astype(int)  # 1=bueno
            f1b = f1_score(y_true_bin, y_pred_bin)
            if f1b > best[0]:
                best = (f1b, t, y_pred_bin)
        f1b, tb, y_pred_bin = best
        cmb = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1])
        accb = accuracy_score(y_true_bin, y_pred_bin)
        print(f"\nBinario (bueno vs no-bueno): umbral={tb:.4e}, F1={f1b:.3f}, Acc={accb:.3f}")
        print("Matriz de confusión binaria [no-bueno, bueno]:")
        print(cmb)

    # 9) Exportar CSV con parsings y resultados
    cout_list, esrc_list = zip(*[parse_cout_esrc(nm) for nm in names_test])
    df_out = pd.DataFrame({
        "filename": names_test,
        "Cout_pct": cout_list,
        "Esr_C_pct": esrc_list,
        "mse": err_te,
        "y_true": y_true,
        "y_pred": y_pred
    })
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df_out.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"\nResultados guardados en: {RESULTS_CSV}")

    # 10) Gráficas
    try:
        plt.figure()
        plt.hist(err_tr, bins=40, alpha=0.6, label="train")
        plt.hist(err_te, bins=40, alpha=0.6, label="test")
        plt.axvline(np.percentile(err_tr,95), linestyle="--", label="p95 train")
        plt.axvline(np.percentile(err_tr,99), linestyle="--", label="p99 train")
        plt.xlabel("MSE de reconstrucción")
        plt.ylabel("Recuentos")
        plt.legend()
        plt.title("Distribución del error de reconstrucción")
        plt.tight_layout()
        plt.show()

        if SK_OK:
            fig = plt.figure()
            im = plt.imshow(cm, interpolation="nearest")
            plt.title("Matriz de confusión (3 clases)")
            plt.colorbar(im)
            tick_marks = np.arange(3)
            plt.xticks(tick_marks, [0,1,2])
            plt.yticks(tick_marks, [0,1,2])
            plt.xlabel("Predicción")
            plt.ylabel("Real")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        warnings.warn(f"No se pudieron mostrar gráficas: {e}")
