#Probar modelo NO cruzado

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

# === CONFIGURACIÓN ===
MODELO_PATH       = "D:/alejandro/plantas/buck/modelo/95/autoencoder_rds_2.keras"
SCALER_PATH       = "D:/alejandro/plantas/buck/modelo/95/autoencoder_rds_2.pkl"
CARPETA_VERIFICAR = "D:/alejandro/plantas/buck/buck_rds2_predecir"
ENTRENAMIENTO_PATH = "D:/alejandro/plantas/buck/buck_rds2_entrenamiento"

'''

# ===================================================================
#ESR L
BUENOS = set()


for v in np.arange(-5.0, 5.1, 0.1):  # de -10.0 a 10.0 con paso 0.01
    nombre = f"Esr_L_{v:.8f}%.txt"
    BUENOS.add(nombre)

# ===================================================================

#Cout
BUENOS = set()

# Generar archivos para valores de -5% a 5% (excepto 0%)
for v in np.arange(-5.0, 5.1, 0.1):
    if v != 0:  # Saltamos el valor 0
        nombre = f"Cout_{v:.8f}%.txt"  # Mantiene el signo correcto (positivo o negativo)
        BUENOS.add(nombre)

# Añadir el archivo para 0 manualmente
BUENOS.add("Cout_0.00000000%.txt")


# ===================================================================
#Para lout
BUENOS = set()

# Generar archivos para valores de -5% a 5% (excepto 0%)
for v in np.arange(-5.0, 5.05, 0.05):
    if v != 0:  # Saltamos el valor 0
        nombre = f"Lout_{v:.8f}%.txt"  # Mantiene el signo correcto (positivo o negativo)
        BUENOS.add(nombre)

# Añadir el archivo para 0 manualmente
BUENOS.add("Lout_0.00000000%.txt")

#---------------------


# ===================================================================
#Para Esr_C
BUENOS = set()

# Generar archivos para valores de -5% a 5% (excepto 0%)
for v in np.arange(-5.0, 5.05, 0.05):
    if v != 0:  # Saltamos el valor 0
        nombre = f"Esr_C_{v:.8f}%.txt"  # Mantiene el signo correcto (positivo o negativo)
        BUENOS.add(nombre)

# Añadir el archivo para 0 manualmente
BUENOS.add("Esr_C_0.00000000%.txt")


# ===================================================================
#Para Rds_1
BUENOS = set()

# Generar archivos para valores de -5% a 5% (excepto 0%)
for v in np.arange(-5.0, 5.05, 0.05):
    if v != 0:  # Saltamos el valor 0
        nombre = f"Rds_1_{v:.8f}%.txt"  # Mantiene el signo correcto (positivo o negativo)
        BUENOS.add(nombre)

# Añadir el archivo para 0 manualmente
BUENOS.add("Rds_1_0.00000000%.txt")
'''
# ===================================================================
#Para Rds_2
BUENOS = set()

# Generar archivos para valores de -5% a 5% (excepto 0%)
for v in np.arange(-5.0, 5.01, 0.01):
    if v != 0:  # Saltamos el valor 0
        nombre = f"Rds_2_{v:.8f}%.txt"  # Mantiene el signo correcto (positivo o negativo)
        BUENOS.add(nombre)

# Añadir el archivo para 0 manualmente
BUENOS.add("Rds_2_0.00000000%.txt")
#---------------------
def mostrar_buenos(buenos_set):
    print("Archivos buenos:")
    for fn in sorted(buenos_set):
        print(" -", fn)

mostrar_buenos(BUENOS)
# Parámetros a explorar

ALPHAS = np.arange(0.05, 1, 0.05).tolist()
# ===================================================================
# === FUNCIONES AUXILIARES ===
# -------------------------------------------------------------------
def procesar_archivo_complejo(path):
    df = pd.read_csv(path, sep=r"\s+", skiprows=1,
                     names=["Frecuencia","Ganancia","Fase"])
    mag_db_arr    = df["Ganancia"].to_numpy()
    phase_deg_arr = df["Fase"].to_numpy()
    mag_lin       = 10 ** (mag_db_arr / 20)
    phase_rad     = np.radians(phase_deg_arr)
    Z             = mag_lin * np.exp(1j * phase_rad)
    return np.column_stack([Z.real, Z.imag])


# -------------------------------------------------------------------
# === CARGAR MODELO Y SCALER ===
print("Cargando modelo y scaler...")
autoencoder = tf.keras.models.load_model(MODELO_PATH)
scaler       = joblib.load(SCALER_PATH)




# -------------------------------------------------------------------
# === PREPARAR DATOS DE VERIFICACIÓN ===
print("Procesando datos de verificación...")
archivos = sorted(f for f in os.listdir(CARPETA_VERIFICAR) if f.endswith(".txt"))
Xv, nombres = [], []
for fn in archivos:
    mat = procesar_archivo_complejo(os.path.join(CARPETA_VERIFICAR, fn))
    Xv.append(scaler.transform(mat).flatten())
    nombres.append(fn)
Xv = np.array(Xv)



# -------------------------------------------------------------------
# === INFERENCIA  ===
print("Ejecutando inferencia...")
recon    = autoencoder.predict(Xv)

errs_mse = np.mean((recon - Xv)**2, axis=1)
# Derivadas
n_samples, flat_dim = Xv.shape
n_freq = flat_dim // 2
recon_mat = recon.reshape(n_samples, n_freq, 2)
orig_mat  = Xv.reshape(n_samples, n_freq, 2)

d_recon   = np.diff(recon_mat[:,:,0], axis=1)
d_orig    = np.diff(orig_mat[:,:,0],    axis=1)
err_d_re  = np.mean(np.abs(d_recon - d_orig), axis=1)

d_recon   = np.diff(recon_mat[:,:,1], axis=1)
d_orig    = np.diff(orig_mat[:,:,1], axis=1)
err_d_im  = np.mean(np.abs(d_recon - d_orig), axis=1)

#prueba del des
err_std = np.std(recon - Xv, axis=1)
# Ground truth
y_true = np.array([0 if fn in BUENOS else 1 for fn in nombres])

# === BÚSQUEDA DE α, β Y UMBRAL ÓPTIMO CON DESVIACIÓN TÍPICA ===
best = {"alpha": None, "beta": None, "threshold": None, "f1": -1.0}
BETAS = np.arange(0.05, 1.0, 0.05).tolist()

print(f"{'α':>5} {'β':>5} {'Umbral':>12} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
for alpha in ALPHAS:
    for beta in BETAS:
        score = errs_mse + alpha * (err_d_re + err_d_im) / 2 + beta * err_std
        for thr in np.unique(score):
            y_pred = (score > thr).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best["f1"]:
                best.update({"alpha": alpha, "beta": beta, "threshold": thr, "f1": f1})
        # imprimir solo métricas en la mejor combinación hasta ahora
        if best["alpha"] == alpha and best["beta"] == beta:
            y_pred_best = (score > best["threshold"]).astype(int)
            print(f"{alpha:5.2f} {beta:5.2f} {best['threshold']:12.2e} "
                  f"{accuracy_score(y_true, y_pred_best):6.3f} "
                  f"{precision_score(y_true, y_pred_best):6.3f} "
                  f"{recall_score(y_true, y_pred_best):6.3f} "
                  f"{f1_score(y_true, y_pred_best):6.3f}")

# -------------------------------------------------------------------
# === RESULTADOS FINALES ===
alpha_opt  = best["alpha"]
thr_opt    = best["threshold"]
print("\nMejor configuración:")
print(f"  α óptimo      = {alpha_opt}")
print(f"  Umbral óptimo = {thr_opt:.6e}")
print(f"  F1-score      = {best['f1']:.4f}")

# Calcula métricas finales

alpha_opt = best["alpha"]
beta_opt = best["beta"]
thr_opt = best["threshold"]

score_opt = errs_mse + alpha_opt * (err_d_re + err_d_im) / 2 + beta_opt * err_std
y_pred_opt = (score_opt > thr_opt).astype(int)

# --- VALIDACIÓN CRUZADA (5-FOLD) ---
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

accs, precs, recs, f1s = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(score_opt), 1):
    # Dividir scores y etiquetas
    score_train, y_train = score_opt[train_idx], y_true[train_idx]
    score_test,  y_test  = score_opt[test_idx],  y_true[test_idx]
    
    # Ajustar umbral por Youden’s J en TRAIN
    fpr, tpr, thr = roc_curve(y_train, score_train)
    j_scores       = tpr - fpr
    best_i         = np.argmax(j_scores)
    thr_opt_fold   = thr[best_i]
    
    # Evaluar en TEST
    y_pred = (score_test > thr_opt_fold).astype(int)
    accs.append(accuracy_score(y_test, y_pred))
    precs.append(precision_score(y_test, y_pred, zero_division=0))
    recs.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    
    print(f"Fold {fold}: umbral={thr_opt_fold:.2e}  "
          f"Acc={accs[-1]:.3f}  Prec={precs[-1]:.3f}  "
          f"Rec={recs[-1]:.3f}  F1={f1s[-1]:.3f}")

# Resultados agregados
print("\n=== Resultados CV 5-fold ===")
print(f"Accuracy : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print(f"Precision: {np.mean(precs):.3f} ± {np.std(precs):.3f}")
print(f"Recall   : {np.mean(recs):.3f} ± {np.std(recs):.3f}")
print(f"F1-score : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")


print("\n--- Métricas con α & umbral óptimos ---")
print(f"Accuracy : {accuracy_score(y_true, y_pred_opt):.4f}")
print(f"Precision: {precision_score(y_true, y_pred_opt):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred_opt):.4f}")
print(f"F1-score : {f1_score(y_true, y_pred_opt):.4f}")

# -------------------------------------------------------------------
# === HISTOGRAMA DEL SCORE ÓPTIMO ===
plt.figure(figsize=(6,4))
plt.hist(score_opt, bins=30, alpha=0.7)
plt.axvline(thr_opt, color='r', linestyle='--',
            label=f'umbral opt={thr_opt:.2e}')
plt.title("Distribución del Score Combinado Óptimo")
plt.xlabel("errs_mse + α·err_deriv")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------------
# === CURVA ROC & PR PARA SCORE ÓPTIMO ===
auc_roc = roc_auc_score(y_true, score_opt)
fpr, tpr, _ = roc_curve(y_true, score_opt)
ap         = average_precision_score(y_true, score_opt)

print(f"\nROC–AUC: {auc_roc:.4f}, PR–AUC: {ap:.4f}")
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f"AUC={auc_roc:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve Óptimo")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(5,5))
prec, rec, _ = precision_recall_curve(y_true, score_opt)
plt.plot(rec, prec, label=f"AP={ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve Óptimo")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------------
# === LISTADO FINAL DE ARCHIVOS ===
print("\n--- Archivos detectados como buenos ---")
for fn, sc, pred in zip(nombres, score_opt, y_pred_opt):
    if pred == 0:
        print(f"{fn}: score={sc:.2e}")

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

# === ENTRENAR REGRESIÓN LOGÍSTICA CON TOD EL CONJUNTO ===
clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
clf.fit(X_scaled, y_true)

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



# === ARCHIVOS CLASIFICADOS COMO BUENOS ===
print("\n--- Archivos que la Regresión Logística considera BUENOS ---")
for name, pred in zip(nombres, y_pred_final):
    if pred == 0:
        print(f"- {name}")

# === MÉTRICAS ===
print(f"\nUmbral óptimo encontrado: {umbral_opt:.10f}")
print(f"F1-score óptimo: {mejor_f1:.4f}")
print(f"Accuracy : {accuracy_score(y_true, y_pred_final):.4f}")
print(f"Precision: {precision_score(y_true, y_pred_final):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred_final):.4f}")
print(f"F1-score : {f1_score(y_true, y_pred_final):.4f}")



from sklearn.metrics import ConfusionMatrixDisplay

cm_bin = confusion_matrix(y_true, y_pred_opt, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm_bin, display_labels=["Bueno", "Anómalo"])
disp.plot(cmap="Greens")
plt.title("Matriz de Confusión Binaria")
plt.grid(False)
plt.tight_layout()
plt.show()


import seaborn as sns

df_box = pd.DataFrame({
    "MSE": errs_mse,
    "d_Re": err_d_re,
    "d_Im": err_d_im,
    "STD": err_std,
    "Clase": y_true
})

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_box.melt(id_vars="Clase"), x="variable", y="value", hue="Clase")
plt.title("Distribución de Errores según la Clase")
plt.grid(True)
plt.show()

import re

variaciones = []
for name in nombres:
    match = re.search(r'[-+]?\d+\.\d+', name)
    if match:
        variaciones.append(float(match.group()))
    else:
        variaciones.append(0)

plt.figure(figsize=(6,4))
plt.scatter(variaciones, score_opt, c=y_true, cmap='coolwarm', alpha=0.7)
plt.axhline(thr_opt, color='gray', linestyle='--', label="Umbral óptimo")
plt.xlabel("Variación porcentual del componente (%)")
plt.ylabel("Score combinado")
plt.title("Score frente a variación del componente")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

sns.histplot(x=y_proba_log, hue=y_true, bins=30, kde=True, palette="Set2")
plt.axvline(umbral_opt, color="red", linestyle="--", label="Umbral óptimo")
plt.title("Distribución de probabilidades predichas")
plt.xlabel("Probabilidad clase 'anómalo'")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Métricas del score combinado
metrics_comb = {
    "Accuracy": accuracy_score(y_true, y_pred_opt),
    "Precision": precision_score(y_true, y_pred_opt),
    "Recall": recall_score(y_true, y_pred_opt),
    "F1-score": f1_score(y_true, y_pred_opt)
}

# Métricas del clasificador logístico
metrics_log = {
    "Accuracy": accuracy_score(y_true, y_pred_final),
    "Precision": precision_score(y_true, y_pred_final),
    "Recall": recall_score(y_true, y_pred_final),
    "F1-score": f1_score(y_true, y_pred_final)
}

# Representación gráfica
labels = list(metrics_comb.keys())
values_comb = [metrics_comb[k] for k in labels]
values_log = [metrics_log[k] for k in labels]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, values_comb, width, label='Score Combinado')
plt.bar(x + width/2, values_log, width, label='Reg. Logística')
plt.ylim(0, 1.05)
plt.xticks(x, labels)
plt.ylabel("Valor")
plt.title("Comparativa de Métricas de Evaluación")
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


from math import pi

def radar_chart(metrics_dict, title=""):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    values += values[:1]  # Cierre del gráfico

    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(title, y=1.1)
    plt.tight_layout()
    plt.show()

radar_chart(metrics_comb, title="Score Combinado")
radar_chart(metrics_log, title="Regresión Logística")




#RECONSTRUCCIONES

idx = 400  
nombre_archivo = nombres[idx]
print(f"Mostrando curva: {nombre_archivo}")

# Dimensiones
n_freq = recon.shape[1] // 2

# Original
curva_orig = Xv[idx].reshape(n_freq, 2)
curva_recon = recon[idx].reshape(n_freq, 2)

frecuencias = np.arange(n_freq)  # índice como pseudo-frecuencia

plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.plot(frecuencias, curva_orig[:, 0], label='Re original')
plt.plot(frecuencias, curva_recon[:, 0], '--', label='Re reconstruido')
plt.xlabel("Punto de muestreo")
plt.ylabel("Parte Real")
plt.title("Reconstrucción de la parte real")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(frecuencias, curva_orig[:, 1], label='Im original')
plt.plot(frecuencias, curva_recon[:, 1], '--', label='Im reconstruido')
plt.xlabel("Punto de muestreo")
plt.ylabel("Parte Imaginaria")
plt.title("Reconstrucción de la parte imaginaria")
plt.legend()
plt.grid(True)

plt.suptitle(f"Comparativa curva original vs reconstruida\nArchivo: {nombre_archivo}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(curva_orig[:, 0], curva_orig[:, 1], label="Original", linewidth=2)
plt.plot(curva_recon[:, 0], curva_recon[:, 1], '--', label="Reconstruido", linewidth=2)
plt.xlabel("Parte Real")
plt.ylabel("Parte Imaginaria")
plt.title("Trayectoria en el plano complejo (Z)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()



def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Sano", "Anómalo"],
                yticklabels=["Sano", "Anómalo"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(title)
    plt.show()

plot_confusion_matrix(y_true, y_pred_final, "Matriz de Confusión")

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