

#ejecutar primero mover a entrenamiento
#copiar los archivos de entrenamiento a predecir para que estén todos en la misma carpeta
#modificar las rutas
#modificar los nombres de archivos p.e "Cout__Esr_C" por "Cout__RDS_1"

#
import os, re, glob, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve,
    roc_auc_score, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =======================
# CONFIGURACIÓN
# =======================
MODELO_PATH        = r"D:/alejandro/plantas/buck/modelo/pinn_buck.keras"
ESCALA_PATH        = r"D:/alejandro/plantas/buck/modelo/pinn_buck_scale.txt"
FREF_PATH          = r"D:/alejandro/plantas/buck/modelo/pinn_buck_fref.npy"
COVINV_PATH        = r"D:/alejandro/plantas/buck/modelo/pinn_buck_cov_inv.npy"
PMU_PATH           = r"D:/alejandro/plantas/buck/modelo/pinn_buck_params_mu.npy"
PSD_PATH           = r"D:/alejandro/plantas/buck/modelo/pinn_buck_params_sd.npy"

CARPETA_VERIFICAR  = r"D:/alejandro/plantas/buck/buck_cruzado_Cout_Rds_1_predecir"
CARPETA_ENTRENAMIENTO = r"D:/alejandro/plantas/buck/buck_cruzado_Cout_Rds_1_entrenamiento"

GOOD_TOL_PCT   = 5.0
BETA_UMBRAL    = 0.8
GAMMA_PARAM_Z  = 0.10  # peso de distancia en parámetros dentro del score combinado

np.set_printoptions(suppress=True, linewidth=140)

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

    #Formato: Frequency  amp(Vo1)  phase(Vo1)
    #amp lineal, fase en radianes y pasa a freq (Hz) y matriz (n_freq,2) Re/Im(Z)

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
    return freq, np.column_stack([re, im])

def cargar_carpeta(carpeta):
    rutas = listar_txt(carpeta)
    if len(rutas) == 0:
        raise RuntimeError(f"No se encontraron .txt en {carpeta}")
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
# PARSEO DE NOMBRES Y ETIQUETAS
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
    partes = base.split("__")
    out = {}
    for p in partes:
        m = re.match(r"([A-Za-z0-9]+(?:_[0-9])?)_([+\-]?\d+(?:\.\d+)?)%", p)
        if m:
            key_raw = m.group(1)
            val = float(m.group(2))
            key = COMPONENT_KEYS.get(key_raw, None)
            if key is not None:
                out[key] = val
    return out

def clasificar_archivo_multinivel(fn, tol=GOOD_TOL_PCT):
    d = parse_porcentajes(fn)
    k = sum(1 for v in d.values() if abs(v) > tol)
    if k == 0: return 0  # Bueno
    if k == 1: return 1  # Leve
    return 2             # Grave

def etiqueta_bueno_bin(fn, tol=GOOD_TOL_PCT):
    return 0 if clasificar_archivo_multinivel(fn, tol) == 0 else 1

# =======================
# CARGA DE ARTEFACTOS PINN
# =======================
scale   = float(np.loadtxt(ESCALA_PATH))
F_ref   = np.load(FREF_PATH)
cov_inv = np.load(COVINV_PATH)
P_mu    = np.load(PMU_PATH)
P_sd    = np.load(PSD_PATH)

# =======================
# CARGA MODELO (con BuckDecoder registrado)
# =======================
@tf.keras.utils.register_keras_serializable()
class BuckDecoder(tf.keras.layers.Layer):
    def __init__(self, omega, alpha1=0.5, alpha2=0.5,
                 lambda_pass=0.0, lambda_smooth=0.0, name="buck_decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        omega_np = np.array(omega, dtype=np.float32).reshape(1, -1)
        self._omega_np = omega_np
        self.omega = tf.constant(omega_np, tf.float32)
        self.alpha1 = float(alpha1); self.alpha2 = float(alpha2)
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

print("Cargando modelo PINN…")
model = tf.keras.models.load_model(MODELO_PATH, compile=False)
params_model = tf.keras.Model(model.input, model.get_layer("params").output)

# =======================
# PROCESAR DATOS
# =======================
print("Procesando archivos de verificación…")
F_list, X_list, nombres = cargar_carpeta(CARPETA_VERIFICAR)
X_mat = reinterpolar_a_malla(X_list, F_list, F_ref)              # (N, n_freq, 2)
Xv = (X_mat.reshape(len(X_mat), -1)) / scale                      # (N, 2*n_freq)
n_samples, flat_dim = Xv.shape
n_freq = flat_dim // 2

BUENOS = set(f for f in os.listdir(CARPETA_ENTRENAMIENTO) if f.endswith(".txt"))
print("Ejemplos buenos (nombres en train):", len(BUENOS))
print("Archivos a verificar:", len(nombres))

# =======================
# INFERENCIA
# =======================
print("Ejecutando inferencia…")
recon = model.predict(Xv, verbose=0)

# Errores y features (como tu verificador cruzado)
errs_mse = np.mean((recon - Xv) ** 2, axis=1)

recon_mat = recon.reshape(n_samples, n_freq, 2)
orig_mat  = Xv.reshape(n_samples, n_freq, 2)

err_d_re = np.mean(np.abs(np.diff(recon_mat[:, :, 0], axis=1) - np.diff(orig_mat[:, :, 0], axis=1)), axis=1)
err_d_im = np.mean(np.abs(np.diff(recon_mat[:, :, 1], axis=1) - np.diff(orig_mat[:, :, 1], axis=1)), axis=1)
err_std  = np.std(recon - Xv, axis=1)

# =======================
# SCORE PINN (Mahalanobis + γ·param_z)
# =======================
E_te = recon - Xv
score_maha = np.einsum("bi,ij,bj->b", E_te, cov_inv, E_te)

P_te     = params_model.predict(Xv, verbose=0)
param_z  = np.sum(((P_te - P_mu) / (P_sd + 1e-9))**2, axis=1)

score = score_maha + GAMMA_PARAM_Z * param_z   # <-- score combinado principal

# =======================
# ETIQUETAS (binario y 3 clases)
# =======================
y_true     = np.array([clasificar_archivo_multinivel(fn, GOOD_TOL_PCT) for fn in nombres], dtype=int)
y_true_bin = (y_true > 0).astype(int)

# =======================
# UMBRAL BINARIO POR Fβ SOBRE score COMBINADO
# =======================
def umbral_por_fbeta(y, s, beta=BETA_UMBRAL):
    p, r, thr = precision_recall_curve(y, s)
    fbeta = (1+beta**2)*p*r/np.maximum(beta**2*p + r, 1e-12)
    if len(thr) == 0:
        return float(np.percentile(s, 90))
    i = int(np.nanargmax(fbeta))
    return thr[max(0, i-1)]

thr_opt = umbral_por_fbeta(y_true_bin, score, beta=BETA_UMBRAL)
y_pred_bin = (score > thr_opt).astype(int)

print("\n--- Métricas (PINN Score COMBINADO) ---")
print(f"Accuracy : {accuracy_score(y_true_bin, y_pred_bin):.4f}")
print(f"Precision: {precision_score(y_true_bin, y_pred_bin, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_true_bin, y_pred_bin):.4f}")
print(f"F1-score : {f1_score(y_true_bin, y_pred_bin):.4f}")
print(f"ROC–AUC  : {roc_auc_score(y_true_bin, score):.4f}")
print(f"PR–AUC   : {average_precision_score(y_true_bin, score):.4f}")
print(f"Umbral(F{BETA_UMBRAL}) : {thr_opt:.3e}")

# =======================
# CLASIFICADOR LOGÍSTICO
# =======================
X_feat = np.column_stack([
    errs_mse,
    err_d_re,
    err_d_im,
    np.max(np.abs(recon - Xv), axis=1),
    err_std
])
scaler_feat = StandardScaler()
X_scaled = scaler_feat.fit_transform(X_feat)

clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000, class_weight="balanced")
clf.fit(X_scaled, y_true_bin)
y_proba_log = clf.predict_proba(X_scaled)[:, 1]

# Búsqueda del umbral óptimo (max F1)
umbral_opt_lr, mejor_f1 = 0.0, -1.0
for thr in np.linspace(0, 1, 1000):
    y_pred_thr = (y_proba_log > thr).astype(int)
    f1 = f1_score(y_true_bin, y_pred_thr)
    if f1 > mejor_f1:
        mejor_f1 = f1
        umbral_opt_lr = thr
y_pred_final = (y_proba_log > umbral_opt_lr).astype(int)

print("\n--- Métricas de la regresión logística binaria ---")
print(f"Umbral óptimo: {umbral_opt_lr:.6f}")
print(f"Accuracy     : {accuracy_score(y_true_bin, y_pred_final):.4f}")
print(f"Precision    : {precision_score(y_true_bin, y_pred_final):.4f}")
print(f"Recall       : {recall_score(y_true_bin, y_pred_final):.4f}")
print(f"F1-score     : {f1_score(y_true_bin, y_pred_final):.4f}")

# =======================
# CONVERTIR A MULTICLASE 
# =======================
y_pred_multi = []
for pred, name in zip(y_pred_final, nombres):
    if pred == 0:
        y_pred_multi.append(0)
    else:
        y_pred_multi.append(clasificar_archivo_multinivel(name, GOOD_TOL_PCT))
y_pred_multi = np.array(y_pred_multi)

# =======================
# MATRIZ DE CONFUSIÓN E INFORME
# =======================
cm = confusion_matrix(y_true, y_pred_multi, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bueno", "Leve", "Crítico"])
plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format="d")
plt.title("Matriz de Confusión Extendida")
plt.grid(False)
plt.tight_layout()
plt.show()

print("\n--- Informe clasificación extendida ---")
print(classification_report(
    y_true, y_pred_multi,
    labels=[0, 1, 2],
    target_names=["Bueno", "Leve", "Crítico"],
    zero_division=0
))

# =======================
# ROC y PR (regresión logística)
# =======================
RocCurveDisplay.from_predictions(y_true_bin, y_proba_log)
plt.title("Curva ROC")
plt.grid(True)
plt.show()

PrecisionRecallDisplay.from_predictions(y_true_bin, y_proba_log)
plt.title("Curva Precisión vs Recall")
plt.grid(True)
plt.show()

# =======================
# Importancia de características (LR)
# =======================
importances = np.abs(clf.coef_[0])
labels = ["MSE", "derr_re", "derr_im", "máximo error", "desviación"]
sns.barplot(x=importances, y=labels)
plt.title("Importancia de características")
plt.xlabel("Peso absoluto")
plt.tight_layout()
plt.show()

# =======================
# Distribución del score combinado
# =======================
df_score = pd.DataFrame({"score": score, "clase": y_true})
sns.histplot(data=df_score, x="score", hue="clase", bins=30, palette="deep", kde=True)
plt.title("Distribución del score combinado (PINN)")
plt.xlabel("Score")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()

# =======================
# Matriz de confusión binaria
# =======================
cm_bin = confusion_matrix(y_true_bin, (score > thr_opt).astype(int), labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm_bin, display_labels=["Bueno", "Anómalo"])
disp.plot(cmap="Greens")
plt.title("Matriz de Confusión Binaria (PINN Score)")
plt.grid(False)
plt.tight_layout()
plt.show()

# =======================
# ERRORES POR CLASE (MSE, derivadas y STD)
# =======================
df_box = pd.DataFrame({
    "MSE": errs_mse,
    "d_Re": err_d_re,
    "d_Im": err_d_im,
    "STD": err_std,
    "Clase": y_true_bin
})
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_box.melt(id_vars="Clase"), x="variable", y="value", hue="Clase")
plt.title("Distribución de Errores según la Clase")
plt.grid(True)
plt.tight_layout()
plt.show()

# =======================
# DISPERSIÓN SCORE vs VARIACIÓN DEL COMPONENTE
# =======================
variaciones = []
for name in nombres:
    d = parse_porcentajes(name)
    variaciones.append(max((abs(v) for v in d.values()), default=0.0))

plt.figure(figsize=(8, 5))
plt.scatter(variaciones, score, c=y_true_bin, cmap='coolwarm', alpha=0.7)
plt.axhline(thr_opt, color='gray', linestyle='--', label="Umbral óptimo (Fβ)")
plt.xlabel("Máxima variación porcentual declarada (%)")
plt.ylabel("Score combinado (PINN)")
plt.title("Score vs variación de componentes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =======================
# HISTOGRAMA DE PROBABILIDADES (LR)
# =======================
sns.histplot(x=y_proba_log, hue=y_true_bin, bins=30, kde=True, palette="Set2")
plt.axvline(umbral_opt_lr, color="red", linestyle="--", label="Umbral óptimo")
plt.title("Distribución de probabilidades (Reg. Logística)")
plt.xlabel("Probabilidad clase 'anómalo'")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =======================
# COMPARATIVA DE MÉTRICAS (BARRAS)
# =======================
metrics_comb = {
    "Accuracy": accuracy_score(y_true_bin, (score > thr_opt).astype(int)),
    "Precision": precision_score(y_true_bin, (score > thr_opt).astype(int), zero_division=0),
    "Recall": recall_score(y_true_bin, (score > thr_opt).astype(int)),
    "F1-score": f1_score(y_true_bin, (score > thr_opt).astype(int))
}
metrics_log = {
    "Accuracy": accuracy_score(y_true_bin, y_pred_final),
    "Precision": precision_score(y_true_bin, y_pred_final, zero_division=0),
    "Recall": recall_score(y_true_bin, y_pred_final),
    "F1-score": f1_score(y_true_bin, y_pred_final)
}
labels_m = list(metrics_comb.keys())
vals_comb = [metrics_comb[k] for k in labels_m]
vals_log  = [metrics_log[k]  for k in labels_m]
x = np.arange(len(labels_m)); width = 0.35
plt.figure(figsize=(9, 5))
plt.bar(x - width/2, vals_comb, width, label='Score PINN (Combinado)')
plt.bar(x + width/2, vals_log,  width, label='Reg. Logística')
plt.ylim(0, 1.05)
plt.xticks(x, labels_m)
plt.ylabel("Valor")
plt.title("Comparativa de Métricas")
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# =======================
# GRÁFICOS RADAR
# =======================
from math import pi
def radar_chart(metrics_dict, title=""):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values()) + [list(metrics_dict.values())[0]]
    angles = [n/float(len(labels)) * 2 * pi for n in range(len(labels))] + [0]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(title, y=1.08)
    plt.tight_layout()
    plt.show()

radar_chart(metrics_comb, title="Score PINN (Combinado)")
radar_chart(metrics_log,  title="Regresión Logística")

# =======================
# RECONSTRUCCIONES (índice seleccionable)
# =======================
idx = min(400, len(nombres)-1)  # evita desbordar si hay menos de 401 muestras
nombre_archivo = nombres[idx]
print(f"Mostrando curva: {nombre_archivo}")
curva_orig  = Xv[idx].reshape(n_freq, 2)
curva_recon = recon[idx].reshape(n_freq, 2)
frecuencias = np.arange(n_freq)

plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.plot(frecuencias, curva_orig[:, 0], label='Re original')
plt.plot(frecuencias, curva_recon[:, 0], '--', label='Re reconstruido')
plt.xlabel("Punto de muestreo"); plt.ylabel("Parte Real")
plt.title("Reconstrucción de la parte real"); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(frecuencias, curva_orig[:, 1], label='Im original')
plt.plot(frecuencias, curva_recon[:, 1], '--', label='Im reconstruido')
plt.xlabel("Punto de muestreo"); plt.ylabel("Parte Imaginaria")
plt.title("Reconstrucción de la parte imaginaria"); plt.legend(); plt.grid(True)

plt.suptitle(f"Comparativa curva original vs reconstruida\nArchivo: {nombre_archivo}")
plt.tight_layout(); plt.show()

plt.figure(figsize=(6, 5))
plt.plot(curva_orig[:, 0], curva_orig[:, 1], label="Original", linewidth=2)
plt.plot(curva_recon[:, 0], curva_recon[:, 1], '--', label="Reconstruido", linewidth=2)
plt.xlabel("Parte Real"); plt.ylabel("Parte Imaginaria")
plt.title("Trayectoria en el plano complejo (Z)")
plt.legend(); plt.grid(True); plt.axis("equal"); plt.tight_layout(); plt.show()

# =======================
# MATRIZ DE CONFUSIÓN (REG. LOGÍSTICA)
# =======================
def plot_confusion_matrix(y_true_b, y_pred_b, title):
    cm = confusion_matrix(y_true_b, y_pred_b)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Sano", "Anómalo"],
                yticklabels=["Sano", "Anómalo"])
    plt.xlabel("Predicción"); plt.ylabel("Real"); plt.title(title); plt.tight_layout(); plt.show()

plot_confusion_matrix(y_true_bin, y_pred_final, "Matriz de Confusión (Regresión Logística)")
