import os
import shutil
import re

# Carpeta de origen con todos los resultados
carpeta_origen = r"D:\alejandro\plantas\buck\buck_cruzado_Cout_Rds2_paralelo"

# Carpeta de destino (entrenamiento)
carpeta_destino = r"D:\alejandro\plantas\buck\buck_cruzado_Cout_Rds_2_entrenamiento"
os.makedirs(carpeta_destino, exist_ok=True)

# Rango de interés (%)
umbral_min = -5.0
umbral_max = 5.0

# Patrón para extraer los porcentajes, VARIAR LOS NOMBRES PARA CADA TIPO DE MODELO
patron = re.compile(r"Cout_([+-]\d+\.\d+)%__Rds_2_([+-]\d+\.\d+)%\.txt")

archivos = os.listdir(carpeta_origen)
total_movidos = 0

for nombre in archivos:
    match = patron.match(nombre)
    if match:
        cout_pct = float(match.group(1))
        rout_pct = float(match.group(2))

        if umbral_min <= cout_pct <= umbral_max and umbral_min <= rout_pct <= umbral_max:
            ruta_origen = os.path.join(carpeta_origen, nombre)
            ruta_destino = os.path.join(carpeta_destino, nombre)
            try:
                shutil.move(ruta_origen, ruta_destino)
                total_movidos += 1
            except Exception as e:
                print(f"❌ Error al mover {nombre}: {e}")

print(f"Total de archivos movidos a entrenamiento: {total_movidos}")

import os

# === RUTAS ===
ENTRENAMIENTO_PATH = "D:/alejandro/plantas/buck/buck_cruzado_Cout_Rds_2_entrenamiento"
CARPETA_VERIFICAR  = "D:/alejandro/plantas/buck/buck_cruzado_Cout_Rds_2_predecir"

# === Listar archivos .txt ===
archivos_entrenamiento = set(f for f in os.listdir(ENTRENAMIENTO_PATH) if f.endswith(".txt"))
archivos_verificacion  = set(f for f in os.listdir(CARPETA_VERIFICAR) if f.endswith(".txt"))

# === Encontrar archivos duplicados ===
duplicados = archivos_entrenamiento & archivos_verificacion

# === Eliminar duplicados de verificación ===
for fn in duplicados:
    ruta = os.path.join(CARPETA_VERIFICAR, fn)
    try:
        os.remove(ruta)
        print(f"Eliminado de verificación: {fn}")
    except Exception as e:
        print(f"Error al eliminar {fn}: {e}")

print(f"\nTotal archivos eliminados: {len(duplicados)}")