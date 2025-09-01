# buck cruzado ascendente multi 8 

import numpy as np
import os
import shutil
import subprocess
import time
from multiprocessing import Pool
from itertools import product

# Componentes a variar
componente_1 = 'Cout'
componente_2 = 'Rds_2'

# Parámetros nominales
nominales = {
    'Vin': 50,
    'fsw': 100000,
    'Rout': 50,
    'Lout': 0.0002,
    'Cout': 0.0001,
    'Rds_1': 0.010,
    'Rds_2': 0.010,
    'Esr_L': 0.1,
    'Esr_C': 0.020,
    'D': 0.4
}

# RANGOS ASCENDENTES
rangos = {
    'Cout':  (0.00008, 0.00012, 0.000001),
    'Rds_2': (0.008, 0.012, 0.00005)
}

# Archivos base
plantilla_psimsch = r"D:\alejandro\plantas\buck\buck.psimsch"
archivo_param = "parameters.txt"

# Ruta base de trabajo
output_dir = r"D:\alejandro\plantas\buck\buck_cruzado_Cout_Rds2_paralelo"
os.makedirs(output_dir, exist_ok=True)

# Generar combinaciones de parámetros
rango_1 = np.arange(*rangos[componente_1])
rango_2 = np.arange(*rangos[componente_2])
combinaciones = list(product(rango_1, rango_2))


def ejecutar_simulacion(args):
    val1, val2 = args
    carpeta_temp = os.path.join(output_dir, f"temp_{componente_1}_{val1:.8f}__{componente_2}_{val2:.8f}")
    os.makedirs(carpeta_temp, exist_ok=True)

    psimsch_path = os.path.join(carpeta_temp, "buck.psimsch")
    shutil.copy(plantilla_psimsch, psimsch_path)

    # Crear parameters.txt
    params = nominales.copy()
    params[componente_1] = val1
    params[componente_2] = val2
    param_path = os.path.join(carpeta_temp, archivo_param)
    with open(param_path, "w") as f:
        for k, v in params.items():
            f.write(f"{k}={v}\n")

    comando = [
        r"C:\Altair\Altair_PSIM_2025\PsimCmd",
        "-i", psimsch_path,
        "-o", os.path.join(carpeta_temp, "buck.txt")
    ]

    try:
        subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=carpeta_temp)
    except Exception as e:
        print(f"Error en simulación {val1}, {val2}: {e}")
        return

    salida = os.path.join(carpeta_temp, "buck.txt")
    for _ in range(10):
        if os.path.exists(salida) and os.path.getsize(salida) > 0:
            break
        time.sleep(1)
    else:
        print(f"Archivo no generado para {val1}, {val2}")
        return

    var_pct_1 = (val1 - nominales[componente_1]) / nominales[componente_1] * 100
    var_pct_2 = (val2 - nominales[componente_2]) / nominales[componente_2] * 100
    nombre = f"{componente_1}_{var_pct_1:+.2f}%__{componente_2}_{var_pct_2:+.2f}%.txt"
    destino = os.path.join(output_dir, nombre)

    try:
        shutil.copy(salida, destino)
        print(f"Simulación guardada: {nombre}")
    except Exception as e:
        print(f"Error guardando resultado: {e}")

    try:
        shutil.rmtree(carpeta_temp)
    except Exception:
        pass


def main():
    print(f"Lanzando {len(combinaciones)} simulaciones con 8 procesos...")
    with Pool(processes=8) as pool:
        pool.map(ejecutar_simulacion, combinaciones)
    print("Todas las simulaciones paralelas completadas.")


if __name__ == '__main__':
    main()

