# ejecutar primero mover a entrenamiento
# copiar los archivos de entrenamiento a predecir para que estén todos en la misma carpeta
# modificar las rutas
# modificar los nombres de archivos p.e "Cout__Esr_C" por "Cout__RDS_1"

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# --- IMPORT 3D ---
from mpl_toolkits.mplot3d import Axes3D  

plt.rcParams.update({
    "figure.dpi": 120,      # opcional, que se vea más nítido
    "axes.titlesize": 18,   # título
    "axes.labelsize": 15,   # etiquetas de ejes (xlabel/ylabel)
    "xtick.labelsize": 13,  # ticks de eje X (abajo)
    "ytick.labelsize": 13,  # ticks de eje Y (lateral)
    "legend.fontsize": 13   # leyenda
})

# === CONFIGURACIÓN ===
MODELO_PATH        = "D:/alejandro/plantas/buck/modelo/autoencoder_Cout_EsrC.keras"
SCALER_PATH        = "D:/alejandro/plantas/buck/modelo/autoencoder_Cout_EsrC.pkl"
CARPETA_VERIFICAR  = "D:/alejandro/plantas/buck/buck_cruzado_Cout_EsrC_predecir"
CARPETA_ENTRENAMIENTO = "D:/alejandro/plantas/buck/buck_cruzado_Cout_EsrC_entrenamiento"

BUENOS = set(f for f in os.listdir(CARPETA_ENTRENAMIENTO) if f.endswith(".txt"))

# === FUNCIONES ===
def procesar_archivo_complejo(path):
    df = pd.read_csv(path, sep=r"\s+", skiprows=1, names=["Frecuencia", "Ganancia", "Fase"])
    mag_db_arr = df["Ganancia"].to_numpy()
    phase_deg_arr = df["Fase"].to_numpy()
    mag_lin = 10 ** (mag_db_arr / 20)
    phase_rad = np.radians(phase_deg_arr)
    Z = mag_lin * np.exp(1j * phase_rad)
    return np.column_stack([Z.real, Z.imag])

def clasificar_archivo(fn):
    if fn in BUENOS:
        return 0  # Bueno
    match = re.match(r"Cout_([+-]?\d+\.\d+)%__Esr_C_([+-]?\d+\.\d+)%\.txt", fn) #esta línea es la que menciono en el word, intenta automatizar este proceo pasando los nombres
    #de los componentes a variables para cambiarlo todo desde el encabezado
    if match:
        pct_cout = float(match.group(1))
        pct_esrc = float(match.group(2))
        fuera_cout = abs(pct_cout) > 5
        fuera_esrc = abs(pct_esrc) > 5
        if fuera_cout and fuera_esrc:
            return 2  # Crítico
        elif fuera_cout or fuera_esrc:
            return 1  # Leve
    return 1  # Por defecto, leve si no se puede determinar

# --- PARSER ROBUSTO PARA VARIACIONES EN NOMBRES + PLOT 3D ---
_CANON = {
    "Rds_1":"Rds_1","RDS_1":"Rds_1","Rds1":"Rds_1",
    "Rds_2":"Rds_2","RDS_2":"Rds_2","Rds2":"Rds_2",
    "Cout":"Cout","COUT":"Cout",
    "Lout":"Lout","LOUT":"Lout",
    "Rout":"Rout","ROUT":"Rout",
    "Esr_L":"Esr_L","ESR_L":"Esr_L","EsrL":"Esr_L",
    "Esr_C":"Esr_C","ESR_C":"Esr_C","EsrC":"Esr_C",
}
_pat = re.compile(r'([A-Za-z]+(?:_[0-9])?)_([+\-]?\d+(?:\.\d+)?)%')

def _variaciones_por_nombre(fname):
    d = {}
    for key, val in _pat.findall(fname):
        key = _CANON.get(key, None)
        if key is not None:
            d[key] = float(val)
    return d  # dict: {"Cout": +1.0, "Esr_C": +0.5, ...}

def plot_score_3d(nombres, score, comp_x, comp_y,
                  labels=None, thr=None, cmap='viridis',
                  elev=25, azim=-30, s=35, fontsize=13):
    """
    nombres : lista de nombres de archivo
    score   : array (N,) con el score de cada archivo (p.ej. score óptimo)
    comp_x  : 'Cout', 'Rds_2', 'Esr_C', ...
    comp_y  : idem
    labels  : opcional array (N,) 0/1/2... para colorear por clase (usa y_true_bin si quieres binario)
    thr     : opcional float para dibujar plano z=thr (umbral)
    """
    X, Y, Z, idx = [], [], [], []
    for i, name in enumerate(nombres):
        d = _variaciones_por_nombre(name)
        if comp_x in d and comp_y in d:
            X.append(d[comp_x]); Y.append(d[comp_y]); Z.append(score[i]); idx.append(i)
    if len(X) == 0:
        print(f"[WARN] No hay archivos con ambos componentes '{comp_x}' y '{comp_y}' en el nombre.")
        return

    X, Y, Z = np.array(X), np.array(Y), np.array(Z)

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection='3d')

    if labels is not None:
        labels = np.asarray(labels)[idx]
        palette = np.array(['tab:blue','tab:red','tab:orange','tab:green'])
        ax.scatter(X, Y, Z, c=palette[np.clip(labels,0,3)], s=s, depthshade=False)
        import matplotlib.patches as mpatches
        patches = []
        uniq = np.unique(labels)
        names = {0:"Sano", 1:"Anómalo", 2:"Leve", 3:"Grave"}
        for u in uniq:
            patches.append(mpatches.Patch(color=palette[int(u)], label=names.get(int(u), str(u))))
        ax.legend(handles=patches, loc='upper left', fontsize=fontsize)
    else:
        p = ax.scatter(X, Y, Z, c=Z, cmap=cmap, s=s, depthshade=False)
        cbar = fig.colorbar(p, ax=ax, shrink=0.75, pad=0.05)
        cbar.set_label("Score combinado", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    if thr is not None:
        xg = np.linspace(np.min(X), np.max(X), 2)
        yg = np.linspace(np.min(Y), np.max(Y), 2)
        XX, YY = np.meshgrid(xg, yg)
        ZZ = np.full_like(XX, thr, dtype=float)
        ax.plot_surface(XX, YY, ZZ, alpha=0.20, color='gray', rstride=1, cstride=1)
        ax.text(np.max(X), np.max(Y), thr, f'Umbral={thr:.2e}',
                fontsize=fontsize, ha='right', va='bottom')

    ax.set_xlabel(f"{comp_x} (%)", fontsize=fontsize)
    ax.set_ylabel(f"{comp_y} (%)", fontsize=fontsize)
    ax.set_zlabel("Score", fontsize=fontsize)
    ax.set_title("Score frente a variación de dos componentes", fontsize=fontsize+2, pad=10)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.zaxis.set_tick_params(labelsize=fontsize)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# === CARGAR MODELO Y SCALER ===
print("Cargando modelo y scaler...")
autoencoder = tf.keras.models.load_model(MODELO_PATH)
scaler = joblib.load(SCALER_PATH)

# === PROCESAR DATOS ===
print("Procesando archivos de verificación...")
archivos = sorted(f for f in os.listdir(CARPETA_VERIFICAR) if f.endswith(".txt"))
Xv, nombres = [], []
for fn in archivos:
    path = os.path.join(CARPETA_VERIFICAR, fn)
    mat = procesar_archivo_complejo(path)
    Xv.append(scaler.transform(mat).flatten())
    nombres.append(fn)
Xv = np.array(Xv)

# === DETECTAR BUENOS ===
print("Ejemplos buenos detectados:", sorted(BUENOS))
print("Archivos a verificar:", sorted(archivos))

# === INFERENCIA ===
print("Ejecutando inferencia...")
recon = autoencoder.predict(Xv)
errs_mse = np.mean((recon - Xv) ** 2, axis=1)

n_samples, flat_dim = Xv.shape
n_freq = flat_dim // 2
recon_mat = recon.reshape(n_samples, n_freq, 2)
orig_mat = Xv.reshape(n_samples, n_freq, 2)

err_d_re = np.mean(np.abs(np.diff(recon_mat[:, :, 0], axis=1) - np.diff(orig_mat[:, :, 0], axis=1)), axis=1)
err_d_im = np.mean(np.abs(np.diff(recon_mat[:, :, 1], axis=1) - np.diff(orig_mat[:, :, 1], axis=1)), axis=1)
err_std  = np.std(recon - Xv, axis=1)

# === GENERAR ETIQUETAS ===
y_true = np.array([clasificar_archivo(fn) for fn in nombres])
y_true_bin = (y_true > 0).astype(int)

# === SCORE ÓPTIMO ===
ALPHAS = np.arange(0.05, 1, 0.05)
BETAS  = np.arange(0.05, 1, 0.05)
best = {"alpha": None, "beta": None, "threshold": None, "f1": -1.0}

print(f"{'α':>5} {'β':>5} {'Umbral':>12} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
for alpha in ALPHAS:
    for beta in BETAS:
        score_tmp = errs_mse + alpha * (err_d_re + err_d_im) / 2 + beta * err_std
        # búsqueda de umbral sobre score_tmp
        for thr in np.unique(score_tmp):
            y_pred = (score_tmp > thr).astype(int)
            f1 = f1_score(y_true_bin, y_pred)
            if f1 > best["f1"]:
                best.update({"alpha": alpha, "beta": beta, "threshold": thr, "f1": f1})
        # imprimir solo métricas en la mejor combinación hasta ahora
        if best["alpha"] == alpha and best["beta"] == beta:
            y_pred_best = (score_tmp > best['threshold']).astype(int)
            print(f"{alpha:5.2f} {beta:5.2f} {best['threshold']:12.2e} "
                  f"{accuracy_score(y_true_bin, y_pred_best):6.3f} "
                  f"{precision_score(y_true_bin, y_pred_best):6.3f} "
                  f"{recall_score(y_true_bin, y_pred_best):6.3f} "
                  f"{f1_score(y_true_bin, y_pred_best):6.3f}")

# se construye el score ÓPTIMO con α y β ganadores
alpha_opt = best["alpha"]
beta_opt  = best["beta"]
thr_opt   = best["threshold"]
score_opt = errs_mse + alpha_opt * (err_d_re + err_d_im) / 2 + beta_opt * err_std

# Para no romper el flujo posterior que usa 'score', se iguala al score óptimo:
score = score_opt

# === CLASIFICADOR LOGÍSTICO ===
X_feat = np.column_stack([
    errs_mse, err_d_re, err_d_im, np.max(np.abs(recon - Xv), axis=1), err_std
])
scaler_feat = StandardScaler()
X_scaled = scaler_feat.fit_transform(X_feat)

clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
clf.fit(X_scaled, y_true_bin)
y_proba_log = clf.predict_proba(X_scaled)[:, 1]

umbral_opt = 0.0
mejor_f1 = 0.0
for thr in np.linspace(0, 1, 1000):
    y_pred_thr = (y_proba_log > thr).astype(int)
    f1 = f1_score(y_true_bin, y_pred_thr)
    if f1 > mejor_f1:
        mejor_f1 = f1
        umbral_opt = thr

y_pred_final = (y_proba_log > umbral_opt).astype(int)

print("\n--- Métricas de la regresión logística binaria ---")
print(f"Umbral óptimo: {umbral_opt:.6f}")
print(f"Accuracy     : {accuracy_score(y_true_bin, y_pred_final):.4f}")
print(f"Precision    : {precision_score(y_true_bin, y_pred_final):.4f}")
print(f"Recall       : {recall_score(y_true_bin, y_pred_final):.4f}")
print(f"F1-score     : {f1_score(y_true_bin, y_pred_final):.4f}")

# === CONVERTIR A MULTICLASE ===
y_pred_multi = []
for pred, name in zip(y_pred_final, nombres):
    if pred == 0:
        y_pred_multi.append(0)
    else:
        y_pred_multi.append(clasificar_archivo(name))
y_pred_multi = np.array(y_pred_multi)

# === MATRIZ DE CONFUSIÓN E INFORME ===
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

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
RocCurveDisplay.from_predictions(y_true_bin, y_proba_log)
plt.title("Curva ROC")
plt.grid(True)
plt.show()

PrecisionRecallDisplay.from_predictions(y_true_bin, y_proba_log)
plt.title("Curva Precisión vs Recall")
plt.grid(True)
plt.show()

importances = np.abs(clf.coef_[0])
labels = ["MSE", "derr_re", "derr_im", "máximo error", "desviación"]
sns.barplot(x=importances, y=labels)
plt.title("Importancia de características")
plt.xlabel("Peso absoluto")
plt.tight_layout()
plt.show()

df_score = pd.DataFrame({
    "score": score,   # usamos score_opt (alias score)
    "clase": y_true
})
sns.histplot(data=df_score, x="score", hue="clase", bins=30, palette="deep", kde=True)
plt.title("Distribución del score combinado")
plt.xlabel("Score")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()

# MATRIZ DE CONFUSIÓN BINARIA (score combinado con umbral óptimo)
cm_bin = confusion_matrix(y_true_bin, (score > best['threshold']).astype(int), labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm_bin, display_labels=["Bueno", "Anómalo"])
disp.plot(cmap="Greens")
plt.title("Matriz de Confusión Binaria")
plt.grid(False)
plt.tight_layout()
plt.show()

# ERRORES POR CLASE (MSE, derivadas y STD)
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

# DISPERSIÓN SCORE vs VARIACIÓN DEL COMPONENTE (2D)
variaciones = []
for name in nombres:
    m = re.search(r'[-+]?\d+\.\d+', name)
    variaciones.append(float(m.group()) if m else 0.0)

plt.figure(figsize=(8, 5))
plt.scatter(variaciones, score, c=y_true_bin, cmap='coolwarm', alpha=0.7)
plt.axhline(best['threshold'], color='gray', linestyle='--', label="Umbral óptimo")
plt.xlabel("Variación porcentual del componente (%)")
plt.ylabel("Score combinado")
plt.title("Score frente a variación del componente")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Usa el score óptimo y el umbral óptimo encontrados
plot_score_3d(nombres, score, comp_x='Cout', comp_y='_Esr_C',
              labels=y_true_bin, thr=best['threshold'], elev=28, azim=-50, s=35, fontsize=13)

# HISTOGRAMA DE PROBABILIDADES PREDICHAS (REG. LOGÍSTICA)
sns.histplot(x=y_proba_log, hue=y_true_bin, bins=30, kde=True, palette="Set2")
plt.axvline(umbral_opt, color="red", linestyle="--", label="Umbral óptimo")
plt.title("Distribución de probabilidades predichas")
plt.xlabel("Probabilidad clase 'anómalo'")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# COMPARATIVA DE MÉTRICAS (BARRAS)
metrics_comb = {
    "Accuracy": accuracy_score(y_true_bin, (score > best['threshold']).astype(int)),
    "Precision": precision_score(y_true_bin, (score > best['threshold']).astype(int)),
    "Recall": recall_score(y_true_bin, (score > best['threshold']).astype(int)),
    "F1-score": f1_score(y_true_bin, (score > best['threshold']).astype(int))
}
metrics_log = {
    "Accuracy": accuracy_score(y_true_bin, y_pred_final),
    "Precision": precision_score(y_true_bin, y_pred_final),
    "Recall": recall_score(y_true_bin, y_pred_final),
    "F1-score": f1_score(y_true_bin, y_pred_final)
}
labels_m = list(metrics_comb.keys())
vals_comb = [metrics_comb[k] for k in labels_m]
vals_log  = [metrics_log[k]  for k in labels_m]
x = np.arange(len(labels_m)); width = 0.35
plt.figure(figsize=(9, 5))
plt.bar(x - width/2, vals_comb, width, label='Score Combinado')
plt.bar(x + width/2, vals_log,  width, label='Reg. Logística')
plt.ylim(0, 1.05)
plt.xticks(x, labels_m)
plt.ylabel("Valor")
plt.title("Comparativa de Métricas de Evaluación")
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# GRÁFICOS RADAR PARA AMBOS MÉTODOS
from math import pi
def radar_chart(metrics_dict, title=""):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    values += values[:1]
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(title, y=1.08)
    plt.tight_layout()
    plt.show()

radar_chart(metrics_comb, title="Score Combinado")
radar_chart(metrics_log,  title="Regresión Logística")

# RECONSTRUCCIONES (ÍNDICE SELECCIONABLE)
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

# HEATMAP DE MATRIZ DE CONFUSIÓN (REG. LOGÍSTICA)
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Sano", "Anómalo"],
                yticklabels=["Sano", "Anómalo"])
    plt.xlabel("Predicción"); plt.ylabel("Real"); plt.title(title); plt.tight_layout(); plt.show()

plot_confusion_matrix(y_true_bin, y_pred_final, "Matriz de Confusión (Regresión Logística)")
