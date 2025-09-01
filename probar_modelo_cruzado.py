

#ejecutar primero mover a entrenamiento
#copiar los archivos de entrenamiento a predecir para que estén todos en la misma carpeta
#modificar las rutas
#modificar los nombres de archivos p.e "Cout__Esr_C" por "Cout__RDS_1"

#
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

# === CONFIGURACIÓN ===
MODELO_PATH        = "D:/alejandro/plantas/buck/modelo/autoencoder_Cout_Rout.keras"
SCALER_PATH        = "D:/alejandro/plantas/buck/modelo/autoencoder_Cout_Rout.pkl"
CARPETA_VERIFICAR  = "D:/alejandro/plantas/buck/buck_cruzado_Cout_Rout_predecir"
CARPETA_ENTRENAMIENTO = "D:/alejandro/plantas/buck/buck_cruzado_Cout_Rout_entrenamiento"

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
    match = re.match(r"Cout_([+-]?\d+\.\d+)%__Rout_([+-]?\d+\.\d+)%\.txt", fn)
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
        score = errs_mse + alpha * (err_d_re + err_d_im) / 2 + beta * err_std
        for thr in np.unique(score):
            y_pred = (score > thr).astype(int)
            f1 = f1_score(y_true_bin, y_pred)
            if f1 > best["f1"]:
                best.update({"alpha": alpha, "beta": beta, "threshold": thr, "f1": f1})
        if best["alpha"] == alpha and best["beta"] == beta:
            y_pred_best = (score > best['threshold']).astype(int)
            print(f"{alpha:5.2f} {beta:5.2f} {best['threshold']:12.2e} "
                  f"{accuracy_score(y_true_bin, y_pred_best):6.3f} "
                  f"{precision_score(y_true_bin, y_pred_best):6.3f} "
                  f"{recall_score(y_true_bin, y_pred_best):6.3f} "
                  f"{f1_score(y_true_bin, y_pred_best):6.3f}")

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
    "score": score,
    "clase": y_true
})
sns.histplot(data=df_score, x="score", hue="clase", bins=30, palette="deep", kde=True)
plt.title("Distribución del score combinado")
plt.xlabel("Score")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()

#MATRIZ DE CONFUSIÓN BINARIA (score combinado con umbral óptimo) ===
cm_bin = confusion_matrix(y_true_bin, (score > best['threshold']).astype(int), labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm_bin, display_labels=["Bueno", "Anómalo"])
disp.plot(cmap="Greens")
plt.title("Matriz de Confusión Binaria")
plt.grid(False)
plt.tight_layout()
plt.show()

#ERRORES POR CLASE (MSE, derivadas y STD) ===
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

#DISPERSIÓN SCORE vs VARIACIÓN DEL COMPONENTE ===
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

#HISTOGRAMA DE PROBABILIDADES PREDICHAS (REG. LOGÍSTICA) ===
sns.histplot(x=y_proba_log, hue=y_true_bin, bins=30, kde=True, palette="Set2")
plt.axvline(umbral_opt, color="red", linestyle="--", label="Umbral óptimo")
plt.title("Distribución de probabilidades predichas")
plt.xlabel("Probabilidad clase 'anómalo'")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#COMPARATIVA DE MÉTRICAS (BARRAS) ===
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

#GRÁFICOS RADAR PARA AMBOS MÉTODOS ===
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

#RECONSTRUCCIONES (ÍNDICE SELECCIONABLE)
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

#HEATMAP DE MATRIZ DE CONFUSIÓN (REG. LOGÍSTICA)
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Sano", "Anómalo"],
                yticklabels=["Sano", "Anómalo"])
    plt.xlabel("Predicción"); plt.ylabel("Real"); plt.title(title); plt.tight_layout(); plt.show()

plot_confusion_matrix(y_true_bin, y_pred_final, "Matriz de Confusión (Regresión Logística)")




#Descomentar para probar el randomForest
'''
# === MATRIZ DE CARACTERÍSTICAS ===
X_feat = np.column_stack([
    errs_mse,
    err_d_re,
    err_d_im,
    np.max(np.abs(recon - Xv), axis=1),
    np.std(recon - Xv, axis=1),
])

# === ESCALAR CARACTERÍSTICAS ===
scaler_feat = StandardScaler()
X_scaled = scaler_feat.fit_transform(X_feat)

# === DIVISIÓN TRAIN/TEST ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_true, test_size=0.2, random_state=42)

# === ENTRENAR REGRESIÓN LOGÍSTICA SIN REGULARIZACIÓN ===
clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
clf.fit(X_train, y_train)

# === PREDICCIONES (PROBABILIDADES) ===
y_proba_log = clf.predict_proba(X_scaled)[:, 1]  # Probabilidades de clase 1 (anómalo)

# === BÚSQUEDA DE UMBRAL ÓPTIMO ===
umbral_opt = 0
mejor_f1 = 0
for thr in np.linspace(0, 1, 1000):
    y_pred_thr = (y_proba_log > thr).astype(int)
    f1 = f1_score(y_true, y_pred_thr)
    if f1 > mejor_f1:
        mejor_f1 = f1
        umbral_opt = thr

# === PREDICCIONES FINALES ===
y_pred_final = (y_proba_log > umbral_opt).astype(int)

# === MÉTRICAS ===
print(f"\nUmbral óptimo encontrado: {umbral_opt:.10f}")
print(f"F1-score óptimo: {mejor_f1:.4f}")
print(f"Accuracy : {accuracy_score(y_true, y_pred_final):.4f}")
print(f"Precision: {precision_score(y_true, y_pred_final):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred_final):.4f}")
print(f"F1-score : {f1_score(y_true, y_pred_final):.4f}")

# === ARCHIVOS CLASIFICADOS COMO BUENOS ===
print("\n--- Archivos que la Regresión Logística considera BUENOS ---")
for name, pred in zip(nombres, y_pred_final):
    if pred == 0:
        print(f"- {name}")
'''