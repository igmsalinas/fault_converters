import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ========================
# Datos reales combinados
# ========================
y_true = (
    [0]*228 + [1]*1250 + [2]*1710 +
    [0]*231 + [1]*1257 + [2]*1711 +
    [0]*451 + [1]*2497 + [2]*3451
)

# Predicciones con errores leves → buenos
y_pred = (
    [0]*228 + [1]*1250 + [2]*1710 +
    [0]*231 + [0]*9 + [1]*(1257-9) + [2]*1711 +
    [0]*451 + [0]*12 + [1]*(2497-12) + [2]*3451
)

# ========================
# Matriz de confusión
# ========================
labels = [0, 1, 2]
labels_names = ["Sano", "Leve", "Crítico"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

# ========================
# Visualización personalizada
# ========================
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_names)
disp.plot(cmap='Blues', ax=ax, colorbar=True, values_format='d')

# Cambiar fuentes
ax.set_title("Matriz de Confusión Extendida", fontsize=20)
ax.set_xlabel("Etiqueta predicha", fontsize=20)
ax.set_ylabel("Etiqueta real", fontsize=20)
ax.tick_params(axis='both', labelsize=20)
disp.im_.colorbar.ax.tick_params(labelsize=18)

# Aumentar tamaño de los números dentro de las celdas
for row in disp.text_:
    for text in np.ravel(row):
        text.set_fontsize(20)


plt.tight_layout()
plt.show()

# ========================
# Gráfico Aciertos vs Errores
# ========================
total_muestras = sum(cm.flatten())
errores = 9 + 10
aciertos = total_muestras - errores

fig, ax = plt.subplots(figsize=(7, 6))
barras = ax.bar(["Aciertos", "Errores"], [aciertos, errores], color=["seagreen", "firebrick"])

for bar in barras:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 100, f"{int(yval)}", ha="center", fontsize=20)

plt.title("Comparativa de Aciertos vs Errores", fontsize=20)
plt.ylabel("Número de muestras", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0, total_muestras + 1000)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Totales de cada clase
n_bueno = 228 + 231 + 451
n_leve  = 1250 + 1257 + 2497
n_crit  = 1710 + 1711 + 3451

clases = ["Sano", "Leve", "Crítico"]
valores = [n_bueno, n_leve, n_crit]

# Crear gráfico
fig, ax = plt.subplots(figsize=(10, 6))
barras = ax.bar(clases, valores, color="royalblue")

# Añadir números encima
for bar in barras:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 100,
            f"{int(yval)}", ha="center", va="bottom", fontsize=20)

# Estética
plt.title("Distribución de clases global", fontsize=20)
plt.ylabel("Número de muestras", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0, max(valores) + 1000)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
