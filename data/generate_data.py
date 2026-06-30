import argparse
import numpy as np
import os
import shutil
import time
from multiprocessing import Pool
from itertools import product
from functools import partial
from psimapipy import PSIM


def parse_value(val_str):
    val_str = val_str.strip()
    multiplier = 1.0
    if val_str.endswith("m"):
        multiplier = 1e-3
        val_str = val_str[:-1]
    elif val_str.endswith("u"):
        multiplier = 1e-6
        val_str = val_str[:-1]
    elif val_str.endswith("n"):
        multiplier = 1e-9
        val_str = val_str[:-1]
    elif val_str.endswith("k") or val_str.endswith("K"):
        multiplier = 1e3
        val_str = val_str[:-1]
    elif val_str.endswith("M") or val_str.lower().endswith("meg"):
        multiplier = 1e6
        if val_str.lower().endswith("meg"):
            val_str = val_str[:-3]
        else:
            val_str = val_str[:-1]
    return float(val_str) * multiplier


def read_nominal_params(param_file):
    params = {"Simview": 0}
    
    try:
        with open(param_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(param_file, "r", encoding="utf-16") as f:
            lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            params[key.strip()] = parse_value(val)
    return params


# PSIM Path
PSIM_PATH = os.environ.get("PSIM_PATH", r"C:\Altair\Altair_PSIM_2026")

# Global variables for worker processes
p1 = None
worker_temp_dir = None
template_path = None
converter_name = None


def init_worker(psim_path, template_file, temp_root, converter):
    global p1, worker_temp_dir, template_path, converter_name
    converter_name = converter

    # Initialize PSIM once per worker
    try:
        p1 = PSIM(psim_path)
    except Exception as e:
        print(f"Error initializing PSIM: {e}")
        p1 = None

    # Setup worker directory
    pid = os.getpid()
    worker_temp_dir = os.path.join(temp_root, f"worker_{pid}")
    os.makedirs(worker_temp_dir, exist_ok=True)

    template_path = os.path.abspath(template_file)


def get_filename(values, component_names, nominal_params):
    name_parts = []
    
    excluded = {"Vin", "fsw", "D", "Simview", "Rout"}
    all_components = [k for k in nominal_params.keys() if k not in excluded]
    
    varied_values = dict(zip(component_names, values))
    
    for name in all_components:
        val = varied_values.get(name, nominal_params[name])
        nominal = nominal_params[name]
        var_pct = (val - nominal) / nominal * 100
        name_parts.append(f"{name}_{int(round(var_pct)):+d}")
        
    return "__".join(name_parts) + ".txt"


def run_simulation(values, component_names, output_dir, nominal_params):
    t_start = time.time()
    global p1, worker_temp_dir, template_path, converter_name

    if p1 is None:
        return

    # Use worker-specific paths
    psimsch_path = os.path.join(worker_temp_dir, f"{converter_name}.psimsch")
    output_file = os.path.join(worker_temp_dir, f"{converter_name}.txt")

    try:
        shutil.copy(template_path, psimsch_path)
    except FileNotFoundError:
        print(f"\nError: Template file not found at {template_path}")
        return

    # Prepare parameters
    params = nominal_params.copy()
    for name, val in zip(component_names, values):
        params[name] = val

    # Run Simulation
    try:
        res = p1.PsimSimulate(psimsch_path, output_file, **params)

        if res.Result == 0:
            print(
                f"\nError in simulation: {res.ErrorMessage} (Took {time.time() - t_start:.4f}s)"
            )
            return

    except Exception as e:
        print(f"\nException in simulation: {e} (Took {time.time() - t_start:.4f}s)")
        return

    # Check if output exists
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        print(f"\nFile not generated (Took {time.time() - t_start:.4f}s)")
        return

    # Generate output filename
    output_filename = get_filename(values, component_names, nominal_params)
    destination = os.path.join(output_dir, output_filename)

    try:
        shutil.copy(output_file, destination)
    except Exception as e:
        print(f"\nError saving result: {e}")


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(
        description="Generate PSIM data with parameter variations."
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=20,
        help="Percentage of variation (e.g., 20 for +/- 20%)",
    )
    parser.add_argument(
        "--step", type=float, default=5, help="Step size percentage (default 5%)"
    )
    parser.add_argument("--components", nargs="+", help="List of components to vary")
    parser.add_argument("--all", action="store_true", help="Vary all components")
    parser.add_argument(
        "--converter",
        type=str,
        default="buck",
        help="Name of the converter (e.g., buck, boost)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input directory containing parameters.txt and .psimsch",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = os.path.join(".", "data", args.converter)

    if args.output is None:
        args.output = os.path.join(args.input_dir, f"{args.converter}_data")

    param_file = os.path.join(args.input_dir, "parameters.txt")
    if not os.path.exists(param_file):
        print(f"Error: Parameter file not found at {param_file}")
        return

    nominal_params = read_nominal_params(param_file)

    psimsch_template = os.path.join(args.input_dir, f"{args.converter}.psimsch")
    if not os.path.exists(psimsch_template):
        print(f"Error: PSIM template not found at {psimsch_template}")
        return

    if args.all:
        excluded = {"Vin", "fsw", "D", "Simview", "Rout"}
        components_to_vary = [k for k in nominal_params.keys() if k not in excluded]
    elif args.components:
        components_to_vary = args.components
    else:
        print("No components specified. Using default: Cout, Rds_2, Lout")
        components_to_vary = ["Cout", "Esr_C", "Esr_L", "Lout"]

    # Validate components
    for comp in components_to_vary:
        if comp not in nominal_params:
            print(f"Error: Component {comp} not found in nominal parameters.")
            return

    print(f"Varying components: {components_to_vary}")
    print(f"Variation: +/- {args.percentage}%")
    print("Step size: 5% of nominal")
    print(f"Setup completed in {time.time() - t0:.2f}s")

    t_ranges = time.time()
    # Generate ranges
    ranges = []
    for comp in components_to_vary:
        nominal = nominal_params[comp]
        start = nominal * (1 - args.percentage / 100)
        end = nominal * (1 + args.percentage / 100)
        step = nominal * (args.step / 100)
        # Use np.arange, adding a small buffer to include the end point if it falls exactly on a step
        comp_range = np.arange(start, end + step * 0.1, step)
        ranges.append(comp_range)

    # Calculate total combinations
    total_combinations = 1
    for r in ranges:
        total_combinations *= len(r)
    print(f"Total combinations to process: {total_combinations}")
    print(f"Range generation took {time.time() - t_ranges:.2f}s")

    combinations_iter = product(*ranges)

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    t_check = time.time()
    # Filter out existing simulations
    existing_files = {f for f in os.listdir(output_dir) if f.endswith('.txt')}
    print(f"Found {len(existing_files)} existing simulations in {output_dir}.")

    filtered_combinations = []
    skipped = 0
    for values in combinations_iter:
        fname = get_filename(values, components_to_vary, nominal_params)
        if fname in existing_files:
            skipped += 1
        else:
            filtered_combinations.append(values)

    print(f"Total matching combinations skipped: {skipped}")
    print(f"Total retained (to run): {len(filtered_combinations)}")
    print(f"Existing file check took {time.time() - t_check:.2f}s")

    # Create temp root for workers in input_dir instead of output_dir
    workers_temp_root = os.path.join(args.input_dir, "temp_workers")
    os.makedirs(workers_temp_root, exist_ok=True)

    print(f"Launching simulations with 8 processes...")

    worker_func = partial(
        run_simulation,
        component_names=components_to_vary,
        output_dir=output_dir,
        nominal_params=nominal_params,
    )

    batch_size = 512
    processed_count = 0

    t_sims = time.time()
    try:
        if filtered_combinations:
            with Pool(
                processes=8,
                initializer=init_worker,
                initargs=(PSIM_PATH, psimsch_template, workers_temp_root, args.converter),
            ) as pool:
                for i in range(0, len(filtered_combinations), batch_size):
                    t_batch = time.time()
                    batch = filtered_combinations[i:i + batch_size]
                    pool.map(worker_func, batch)
                    processed_count += len(batch)
                    print(
                        f"\nProcessed {processed_count}/{len(filtered_combinations)} new simulations... (Batch: {time.time() - t_batch:.2f}s)"
                    )
        else:
            print("No new simulations to run. All requested combinations already exist.")

        print(f"All parallel simulations completed in {time.time() - t_sims:.2f}s")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
    finally:
        # Cleanup worker temp dirs
        try:
            shutil.rmtree(workers_temp_root)
        except Exception:
            pass

    print(f"Total execution time: {time.time() - t0:.2f}s")
    print(f"Total simulations executed: {processed_count}")


if __name__ == "__main__":
    main()
