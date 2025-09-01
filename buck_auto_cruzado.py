# Simulación cruzada PSIM: variación Cout y Rout con pasos individuales

import numpy as np
import os
import time
import subprocess

# Componentes a variar
componente_1 = 'Cout'
componente_2 = 'Rout'

# Parámetros nominales
nominales = {
    'Vin': 50,
    'fsw': 100000,
    'Rout': 4,
    'Lout': 0.0002,
    'Cout': 0.0001,
    'Rds_1': 0.010,
    'Rds_2': 0.010,
    'Esr_L': 0.1,
    'Esr_C': 0.020,
    'D': 0.4,
    'Rout': 50
}

# === RANGOS PERSONALIZADOS ===
rangos = {
    'Lout':  (0.00016, 0.00024, 0.0000001),
    'Cout':  (0.00008, 0.00012, 0.0000001),
    'Rds_1': (0.008,   0.012,   0.00001),
    'Rds_2': (0.008,   0.012,   0.000001),
    'Esr_L': (0.08,    0.12,    0.0001),
    'Esr_C': (0.016,   0.024,    0.00001),
    'Rout': (40, 60, 0.1)
}

# Generar rangos
inicio_1, fin_1, paso_1 = rangos[componente_1]
inicio_2, fin_2, paso_2 = rangos[componente_2]

rango_1 = np.arange(inicio_1, fin_1 + paso_1, paso_1)
rango_2 = np.arange(inicio_2, fin_2 + paso_2, paso_2)


#Cambiar ruta de resultados por una más general para evitar tener que cambiarla todo el rato
# Rutas
param_file = r"D:\alejandro\plantas\buck\parameters.txt"
result_base_dir = r"D:\alejandro\plantas\buck\buck_cruzado_Cout_Rout"
os.makedirs(result_base_dir, exist_ok=True)

def update_param_file(params):
    with open(param_file, "w") as f:
        for k, v in params.items():
            f.write(f"{k}={v}\n")

def run_simulation(params, val1, val2):
    update_param_file(params)
    print(f"Simulando con {componente_1}={val1:.8f}, {componente_2}={val2:.8f}")

    comando = [
        r"C:\Altair\Altair_PSIM_2025\PsimCmd",
        "-i", r"D:\alejandro\plantas\buck\buck.psimsch",
        "-o", r"D:\alejandro\plantas\buck\buck.txt"
    ]

    try:
        subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        print(f"Error ejecutando simulación: {e}")
        return

    sim_result_file = r"D:\alejandro\plantas\buck\buck.txt"
    for _ in range(3):
        if os.path.exists(sim_result_file) and os.path.getsize(sim_result_file) > 0:
            break
        time.sleep(1)

    var_pct_1 = (val1 - nominales[componente_1]) / nominales[componente_1] * 100
    var_pct_2 = (val2 - nominales[componente_2]) / nominales[componente_2] * 100

    nombre = f"{componente_1}_{var_pct_1:+.2f}%__{componente_2}_{var_pct_2:+.2f}%.txt"
    file_path = os.path.join(result_base_dir, nombre)

    try:
        with open(file_path, 'w') as out, open(sim_result_file, 'r') as sim:
            out.write(sim.read())
        print(f"✔ Guardado: {file_path}")
    except Exception as e:
        print(f"Error guardando archivo: {e}")
 
def main():
    for val1 in rango_1:
        for val2 in rango_2:
            params = nominales.copy()
            params[componente_1] = val1
            params[componente_2] = val2
            run_simulation(params, val1, val2)
    print("Todas las simulaciones completadas.")

if __name__ == '__main__':
    main()
