"""
Exploration and Analysis Visualization Utilities
================================================

Utilities to plot individual component anomalies and hyperparameter search results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ..data.loader import parse_filename
from .plotter import load_transfer_function, plot_amplitude, plot_phase
from .deck_style import (
    apply_deck_style,
    categorical_colors,
    deviation_color,
    style_axis,
    text_on,
    ACCENT,
    INK,
    INK_SOFT,
    MUTED,
    CATEGORICAL,
    MONO_CMAP,
    SEQUENTIAL_CMAP,
    PAPER_SOFT,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

apply_deck_style()


def extract_single_component_anomalies(
    data_dir: Union[str, Path],
) -> Dict[str, Dict[float, Path]]:
    """
    Scans a directory of simulation runs and extracts single-component anomalies.
    Finds files where only ONE component deviates from 0%, and maps them.

    Args:
        data_dir: Path to directory holding the `.txt` files.

    Returns:
        A dictionary mapping component names to another dict mapping the deviation % to the FilePath.
        e.g., {"Cout": {0.0: path_normal, 5.0: path_5, -5.0: path_m5, ...}}
    """
    data_dir = Path(data_dir)
    results = {}
    normal_file = None

    if not data_dir.exists():
        logger.error(f"Directory {data_dir} does not exist.")
        return results

    logger.info(f"Scanning {data_dir} for single-component variations...")
    import os

    try:
        filenames = os.listdir(data_dir)
    except Exception as e:
        logger.error(f"Error reading directory {data_dir}: {e}")
        return results

    for filename in filenames:
        if not filename.endswith(".txt") or filename == "parameters.txt":
            continue

        filepath = data_dir / filename
        try:
            metadata = parse_filename(filename)
        except Exception:
            continue

        variations = metadata.variations
        # Find which components are non-zero
        non_zero_components = [k for k, v in variations.items() if abs(v) > 0.1]

        if len(non_zero_components) == 0:
            normal_file = filepath
        elif len(non_zero_components) == 1:
            comp = non_zero_components[0]
            val = variations[comp]

            if comp not in results:
                results[comp] = {}
            results[comp][val] = filepath

    # Attach the normal run (0%) to each component as the baseline
    if normal_file:
        for comp in results:
            results[comp][0.0] = normal_file
    else:
        logger.warning(f"No zero-variation normal file found in {data_dir}.")

    return results


def plot_component_anomalies(
    component_name: str,
    variation_paths: Dict[float, Path],
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plots the amplitude and phase shifts caused by varying a single component.

    Args:
        component_name: The name of the component being varied.
        variation_paths: Mapping from variation % to the path of the `.txt` run.
        save_path: Location to save the generated image.
        show_plot: Whether to display it using `plt.show()`.
    """
    sorted_variations = sorted(variation_paths.keys())

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Use max magnitude to normalize colors properly (-20 to 20 goes from 0.0 to 1.0)
    max_val = max([abs(v) for v in sorted_variations] + [1.0])

    for val in sorted_variations:
        filepath = variation_paths[val]
        try:
            freq, amp, phase = load_transfer_function(filepath)
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            continue

        is_normal = abs(val) < 0.1

        if is_normal:
            color = INK
            linewidth = 3.0
            zorder = 10
            label = f"{component_name} Normal (0%)"
        else:
            # Muted diverging ramp (slate = negative, terracotta = positive)
            color = deviation_color(val, max_val)
            linewidth = 1.6
            zorder = 5
            label = f"{val:+} %"

        ax1.semilogx(
            freq, amp, label=label, color=color, linewidth=linewidth, zorder=zorder
        )
        ax2.semilogx(
            freq, phase, label=label, color=color, linewidth=linewidth, zorder=zorder
        )

    # Post-process Ax1 (Amplitude)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Amplitude (dB)")
    ax1.set_title(f"Amplitude Deviation · {component_name}")
    ax1.grid(True, axis="both")
    ax1.legend(loc="best", fontsize="small", ncol=2)
    style_axis(ax1)

    # Post-process Ax2 (Phase)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.set_title(f"Phase Deviation · {component_name}")
    ax2.grid(True, axis="both")
    ax2.legend(loc="best", fontsize="small", ncol=2)
    style_axis(ax2)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def parse_kerastuner_results(
    experiments_dir: Union[str, Path],
) -> Dict[str, List[Dict]]:
    """
    Parses keras-tuner `trial.json` files from the specified `experiments/` directory.

    Args:
        experiments_dir: Root `experiments/` directory

    Returns:
        Dict mapping model names -> list of trial data
        {"conv1d_ae": [{"score": 0.001, "hps": {...}}, ...], "lstm_ae": [...]}
    """
    experiments_dir = Path(experiments_dir)
    results = {}

    if not experiments_dir.exists():
        logger.error(f"{experiments_dir} does not exist.")
        return results

    # Find all trial.json files (e.g., experiments/conv1d_ae/hp_tuning/*/trial_*/trial.json)
    for trial_file in experiments_dir.rglob("hp_tuning/**/trial.json"):
        parts = trial_file.parts
        try:
            hptuning_idx = parts.index("hp_tuning")
            model_name = parts[hptuning_idx - 1]
        except ValueError:
            continue

        try:
            with open(trial_file, "r") as f:
                trial_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read JSON {trial_file}: {e}")
            continue

        score = trial_data.get("score")
        status = trial_data.get("status")
        values = trial_data.get("hyperparameters", {}).get("values", {})

        if score is None or status != "COMPLETED":
            continue

        if model_name not in results:
            results[model_name] = []

        results[model_name].append(
            {"score": score, "hps": values, "path": str(trial_file)}
        )

    return results


def plot_hyperparameter_comparison(
    tuner_results: Dict[str, List[Dict]],
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Scatter plot comparing the validation losses of all completed trials across different architectures.
    """
    if not tuner_results:
        logger.warning("No tuner results provided.")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(tuner_results.keys())

    # Curated deck palette, one hue per architecture
    colors = categorical_colors(len(models))

    for i, model in enumerate(models):
        trials = tuner_results[model]
        scores = [t["score"] for t in trials]

        # Add jitter to X-axis
        x_jittered = np.random.normal(i, 0.05, size=len(scores))

        # Plot all trials as dots
        ax.scatter(
            x_jittered, scores, color=colors[i], alpha=0.55, label=model,
            s=42, edgecolors="none",
        )

        # Mark the best score with a star
        best_score = min(scores)
        ax.scatter(
            [i],
            [best_score],
            color=ACCENT,
            marker="*",
            s=220,
            zorder=5,
            edgecolors=INK,
            linewidths=0.7,
        )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_title("Model Validation Losses across Hyperparameter Searches")
    ax.set_ylabel("Validation Loss (MSE) \u00b7 Log Scale")
    ax.set_xlabel("Autoencoder Architecture")
    ax.set_yscale("log")
    ax.grid(axis="y")
    ax.legend()
    style_axis(ax)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_model_hp_search(
    model_name: str,
    trials: List[Dict],
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Creates multiple scatter plots comparing validation loss against individual hyperparameters.
    """
    if not trials:
        logger.warning(f"No trials provided for {model_name}.")
        return plt.figure()

    # Get arbitrary hyperparameter keys from the first trial
    hp_keys = list(trials[0]["hps"].keys())
    if not hp_keys:
        return plt.figure()

    num_hps = len(hp_keys)
    cols = min(3, num_hps)
    rows = int(np.ceil(num_hps / cols)) if cols > 0 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    if num_hps == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    scores = [t["score"] for t in trials]

    for i, hp_key in enumerate(hp_keys):
        ax = axes[i]
        hp_vals = [t["hps"].get(hp_key) for t in trials]

        # Categorical or boolean variables get jittered X plots
        if len(hp_vals) > 0 and isinstance(hp_vals[0], (str, bool)):
            unique_vals = list(set(hp_vals))
            x_num = [unique_vals.index(v) for v in hp_vals]
            x_jittered = np.random.normal(x_num, 0.05, size=len(x_num))

            ax.scatter(x_jittered, scores, alpha=0.6, color=CATEGORICAL[2], edgecolors="none")
            ax.set_xticks(range(len(unique_vals)))
            ax.set_xticklabels([str(v) for v in unique_vals])
        else:
            ax.scatter(hp_vals, scores, alpha=0.6, color=CATEGORICAL[2], edgecolors="none")
            # If values span a large range monotonically and are positive, graph on log scale
            if all(v is not None and v > 0 for v in hp_vals):
                min_val = min(hp_vals)
                if min_val > 0 and (max(hp_vals) / min_val > 100):
                    ax.set_xscale("log")

        ax.set_xlabel(hp_key)
        ax.set_ylabel("Validation Loss")
        ax.set_yscale("log")
        ax.grid(True, axis="both")
        style_axis(ax)

    # Hide unused extra grid spaces
    for j in range(len(hp_keys), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{model_name} Hyperparameter Search Surface")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_metrics_comparison(
    model_metrics: Dict[str, Dict[str, Any]],
    metrics_to_plot: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plots a grouped bar chart comparing multiple performance metrics across models.
    Supports both standard metrics and the new 'all_thresholds' structure.
    """
    if not model_metrics:
        logger.warning("No metrics provided for comparison.")
        return plt.figure()

    # Define default metrics if not provided
    metrics_to_plot = metrics_to_plot or [
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "separability",
    ]

    # Extract model results into a DataFrame-ready format
    rows = []
    for model_name, metrics in model_metrics.items():
        # Handle flattened metrics vs nested metrics
        # If metrics has 'all_thresholds', use the top-level (best) ones

        # Calculate separability if missing but possible
        if "separability" not in metrics and all(
            k in metrics
            for k in ["mean_anomaly_error", "mean_normal_error", "std_normal_error"]
        ):
            if metrics["std_normal_error"] > 0:
                raw_sep = (
                    metrics["mean_anomaly_error"] - metrics["mean_normal_error"]
                ) / metrics["std_normal_error"]
                # Normalize to [0, 1] using S / (1 + S)
                metrics["separability"] = raw_sep / (1 + raw_sep)
            else:
                metrics["separability"] = 0.0

        for m_name in metrics_to_plot:
            val = metrics.get(m_name, 0.0)
            rows.append(
                {
                    "Model": model_name,
                    "Metric": m_name.replace("_", " ").title(),
                    "Value": val,
                }
            )

    df = pd.DataFrame(rows)
    n_models = df["Model"].nunique()

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.barplot(
        data=df,
        x="Metric",
        y="Value",
        hue="Model",
        palette=categorical_colors(n_models),
        edgecolor=PAPER_SOFT,
        linewidth=0.6,
        ax=ax,
    )

    ax.set_title("Performance Metrics Comparison (Best Test Threshold)")
    ax.set_ylabel("Value")
    ax.set_xlabel("Metric Type")
    ax.grid(True, axis="y")
    ax.legend(title="Model Architectures", bbox_to_anchor=(1.02, 1), loc="upper left")
    style_axis(ax)

    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="center",
                xytext=(0, 9),
                textcoords="offset points",
                fontsize=8,
                color=INK_SOFT,
            )

    plt.tight_layout()

    fig = plt.gcf()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Comparison plot saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_threshold_comparison(
    model_metrics: Dict[str, Dict[str, Any]],
    target_metric: str = "f1",
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plots a comparison of a specific metric across all threshold methods for each model.
    """
    rows = []
    found_any = False

    for model_name, metrics in model_metrics.items():
        all_thresh = metrics.get("all_thresholds", {})
        if not all_thresh:
            continue

        found_any = True
        for method, results in all_thresh.items():
            method_metrics = results.get("metrics", {})
            val = method_metrics.get(target_metric, 0.0)
            rows.append({"Model": model_name, "Method": method.title(), "Value": val})

    if not found_any:
        logger.warning("No multi-threshold data found in any model metrics.")
        return plt.figure()

    df = pd.DataFrame(rows)
    n_methods = df["Method"].nunique()

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=df, x="Model", y="Value", hue="Method",
        palette=categorical_colors(n_methods), edgecolor=PAPER_SOFT, linewidth=0.6,
        ax=ax,
    )

    title_metric = target_metric.replace("_", " ").title()
    ax.set_title(f"Threshold Method Comparison: {title_metric} Performance")
    ax.set_ylabel(f"Test Set {title_metric}")
    ax.set_xlabel("Model Architecture")
    ax.grid(True, axis="y")
    ax.legend(title="Threshold Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    style_axis(ax)

    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="center",
                xytext=(0, 7),
                textcoords="offset points",
                fontsize=8,
                color=INK_SOFT,
            )

    plt.tight_layout()

    fig = plt.gcf()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Threshold comparison plot saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_confusion_matrix_from_counts(
    counts: Dict[str, Any],
    model_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Renders a confusion matrix heatmap using pre-calculated TP, TN, FP, FN counts.
    """
    try:
        tp = int(counts.get("true_positives", 0))
        tn = int(counts.get("true_negatives", 0))
        fp = int(counts.get("false_positives", 0))
        fn = int(counts.get("false_negatives", 0))
    except (ValueError, TypeError):
        logger.error(f"Could not parse confusion matrix counts for {model_name}")
        return plt.figure()

    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(8, 6))

    denom = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(
        cm.astype("float"),
        denom,
        out=np.zeros_like(cm.astype("float")),
        where=denom != 0,
    )

    labels = np.array(
        [
            [f"{int(tn)}\n({cm_norm[0, 0]:.1%})", f"{int(fp)}\n({cm_norm[0, 1]:.1%})"],
            [f"{int(fn)}\n({cm_norm[1, 0]:.1%})", f"{int(tp)}\n({cm_norm[1, 1]:.1%})"],
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap=MONO_CMAP,
        cbar=False,
        linewidths=1.4,
        linecolor=PAPER_SOFT,
        vmin=0,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=ax,
    )

    # Recolour annotations per cell for contrast on the mono ramp
    vmax = cm.max() if cm.max() > 0 else 1
    for txt, val in zip(ax.texts, cm.flatten()):
        txt.set_color(text_on(MONO_CMAP(val / vmax)))

    ax.set_title(f"Confusion Matrix \u00b7 {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.tick_params(length=0)

    plt.tight_layout()

    fig = plt.gcf()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix for {model_name} saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_component_error_analysis(
    results_df: pd.DataFrame,
    model_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Renders an error breakdown analysis grouped by component, supporting multi-component variations.
    Expects a DataFrame with ['varied_components', 'label', 'prediction'] columns.
    """
    if results_df.empty:
        logger.warning(f"No results data provided for error analysis of {model_name}")
        return plt.figure()

    # Create a local copy to avoid modifying the input df
    df = results_df.copy()

    # Identify errors
    df["is_fn"] = (df["label"] == 1) & (df["prediction"] == 0)
    df["is_fp"] = (df["label"] == 0) & (df["prediction"] == 1)

    error_summary = []

    # Identify all unique components across all anomalies
    all_components = set()
    for comps in df["varied_components"].dropna().unique():
        if comps:
            all_components.update(comps.split(","))

    components = sorted(list(all_components))

    for comp in components:
        # A sample belongs to component 'comp' if 'comp' is in its varied_components list
        mask = df["varied_components"].fillna("").apply(lambda x: comp in x.split(","))
        comp_data = df[mask]

        anomalies = comp_data[comp_data["label"] == 1]
        fn_count = anomalies["is_fn"].sum()
        total_anomalies = len(anomalies)
        fn_rate = (fn_count / total_anomalies) * 100 if total_anomalies > 0 else 0

        error_summary.append(
            {
                "Component": comp,
                "Type": "Missed Anomaly (FN)",
                "Rate": fn_rate,
                "Count": fn_count,
            }
        )

    # Standard FP analysis (on all normal data)
    normal_data = df[df["label"] == 0]
    fp_count = normal_data["is_fp"].sum()
    total_normal = len(normal_data)
    fp_rate = (fp_count / total_normal) * 100 if total_normal > 0 else 0
    error_summary.append(
        {
            "Component": "All Normal",
            "Type": "False Alarm (FP)",
            "Rate": fp_rate,
            "Count": fp_count,
        }
    )

    df_plot = pd.DataFrame(error_summary)

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(
        data=df_plot,
        x="Component",
        y="Rate",
        hue="Type",
        palette=[ACCENT, "#b79155"],
        edgecolor=PAPER_SOFT,
        linewidth=0.6,
    )

    ax.set_title(f"Error Distribution by Component Influence \u00b7 {model_name}")
    ax.set_ylabel("Error Rate (%)")
    ax.set_ylim(0, max(df_plot["Rate"].max() * 1.3, 10))
    ax.grid(True, axis="y")
    style_axis(ax)

    # Add count labels
    for p in ax.patches:
        height = p.get_height()
        if height >= 0:
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="center",
                xytext=(0, 9),
                textcoords="offset points",
                fontsize=9,
                color=INK_SOFT,
            )

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Component error analysis saved to {save_path}")

    if show_plot:
        plt.show()

    return plt.gcf()


def plot_error_by_deviation_magnitude(
    results_df: pd.DataFrame,
    model_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plots two complementary views of FN errors by signed maximum deviation:
      - Left: Distribution of false negatives across deviation bins (sums to 100%).
      - Right: Per-bin miss rate (FN / total anomalies in that bin).
    """
    anomalies = results_df[results_df["label"] == 1].copy()
    if anomalies.empty:
        return plt.figure()

    def get_max_signed_deviation(row):
        try:
            if "variations_json" in row and pd.notna(row["variations_json"]):
                vars_dict = json.loads(row["variations_json"])
                if vars_dict:
                    return max(vars_dict.values(), key=abs)
            return row["max_deviation"]
        except Exception:
            return row["max_deviation"]

    if "variations_json" in anomalies.columns:
        anomalies["signed_max_dev"] = anomalies.apply(get_max_signed_deviation, axis=1)
    else:
        anomalies["signed_max_dev"] = anomalies["max_deviation"]

    # Round to nearest integer for grouping (they are already percentages)
    anomalies["dev_bin"] = anomalies["signed_max_dev"].round().astype(int)
    anomalies["is_fn"] = (anomalies["prediction"] == 0).astype(int)

    # --- Compute per-bin stats ---
    grouped = anomalies.groupby("dev_bin").agg(
        total=("is_fn", "size"),
        fn_count=("is_fn", "sum"),
    ).reset_index()

    total_fn = grouped["fn_count"].sum()

    # Distribution: what % of ALL FN come from this bin (sums to 100%)
    grouped["fn_distribution"] = (
        (grouped["fn_count"] / total_fn * 100) if total_fn > 0 else 0
    )
    # Per-bin miss rate: what % of anomalies in this bin were missed
    grouped["fn_rate"] = grouped["fn_count"] / grouped["total"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: FN Distribution (sums to 100%) ---
    sns.barplot(
        data=grouped, x="dev_bin", y="fn_distribution", color=ACCENT, alpha=0.9,
        edgecolor=PAPER_SOFT, linewidth=0.6, ax=ax1,
    )
    ax1.set_title(
        f"Distribution of Missed Anomalies (FN) \u00b7 {model_name}",
    )
    ax1.set_xlabel("Max Deviation (%)")
    ax1.set_ylabel("Share of all FN (%)")
    ax1.set_ylim(0, max(grouped["fn_distribution"].max() * 1.3, 10))
    style_axis(ax1)

    for i, (p, row) in enumerate(zip(ax1.patches, grouped.itertuples())):
        height = p.get_height()
        ax1.annotate(
            f"{height:.1f}%\n({row.fn_count}/{row.total})",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center", va="bottom", fontsize=9, color=INK_SOFT,
        )

    ax1.text(
        0.98, 0.95, f"Total FN: {total_fn} / {grouped['total'].sum()} anomalous",
        transform=ax1.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PAPER_SOFT, edgecolor="none", alpha=0.9),
    )

    # --- Right: Per-bin miss rate ---
    sns.barplot(
        data=grouped, x="dev_bin", y="fn_rate", color="#7a3f26", alpha=0.9,
        edgecolor=PAPER_SOFT, linewidth=0.6, ax=ax2,
    )
    ax2.set_title(
        f"Per-Bin Miss Rate \u00b7 {model_name}",
    )
    ax2.set_xlabel("Max Deviation (%)")
    ax2.set_ylabel("Miss Rate within bin (%)")
    ax2.set_ylim(0, max(grouped["fn_rate"].max() * 1.3, 10))
    style_axis(ax2)

    for i, (p, row) in enumerate(zip(ax2.patches, grouped.itertuples())):
        height = p.get_height()
        ax2.annotate(
            f"{height:.1f}%\n(n={row.total})",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center", va="bottom", fontsize=9, color=INK_SOFT,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    return fig


def plot_error_by_variation_count(
    results_df: pd.DataFrame,
    model_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plots two complementary views of FN errors by number of simultaneous variations:
      - Left: Distribution of false negatives across variation counts (sums to 100%).
      - Right: Per-bin miss rate (FN / total anomalies with that variation count).
    """
    anomalies = results_df[results_df["label"] == 1].copy()
    if anomalies.empty:
        return plt.figure()

    anomalies["is_fn"] = (anomalies["prediction"] == 0).astype(int)

    # --- Compute per-group stats ---
    grouped = anomalies.groupby("num_variations").agg(
        total=("is_fn", "size"),
        fn_count=("is_fn", "sum"),
    ).reset_index()

    total_fn = grouped["fn_count"].sum()

    # Distribution: what % of ALL FN come from this group (sums to 100%)
    grouped["fn_distribution"] = (
        (grouped["fn_count"] / total_fn * 100) if total_fn > 0 else 0
    )
    # Per-group miss rate: what % of anomalies with N variations were missed
    grouped["fn_rate"] = grouped["fn_count"] / grouped["total"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: FN Distribution (sums to 100%) ---
    sns.barplot(
        data=grouped, x="num_variations", y="fn_distribution",
        color="#8c5361", alpha=0.9, edgecolor=PAPER_SOFT, linewidth=0.6, ax=ax1,
    )
    ax1.set_title(
        f"Distribution of Missed Anomalies (FN) \u00b7 {model_name}",
    )
    ax1.set_xlabel("Number of Simultaneous Component Variations")
    ax1.set_ylabel("Share of all FN (%)")
    ax1.set_ylim(0, max(grouped["fn_distribution"].max() * 1.3, 10))
    style_axis(ax1)

    for i, (p, row) in enumerate(zip(ax1.patches, grouped.itertuples())):
        height = p.get_height()
        ax1.annotate(
            f"{height:.1f}%\n({row.fn_count}/{row.total})",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center", va="bottom", fontsize=9, color=INK_SOFT,
        )

    ax1.text(
        0.98, 0.95, f"Total FN: {total_fn} / {grouped['total'].sum()} anomalous",
        transform=ax1.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PAPER_SOFT, edgecolor="none", alpha=0.9),
    )

    # --- Right: Per-group miss rate ---
    sns.barplot(
        data=grouped, x="num_variations", y="fn_rate",
        color="#5f3743", alpha=0.9, edgecolor=PAPER_SOFT, linewidth=0.6, ax=ax2,
    )
    ax2.set_title(
        f"Per-Group Miss Rate \u00b7 {model_name}",
    )
    ax2.set_xlabel("Number of Simultaneous Component Variations")
    ax2.set_ylabel("Miss Rate within group (%)")
    ax2.set_ylim(0, max(grouped["fn_rate"].max() * 1.3, 10))
    style_axis(ax2)

    for i, (p, row) in enumerate(zip(ax2.patches, grouped.itertuples())):
        height = p.get_height()
        ax2.annotate(
            f"{height:.1f}%\n(n={row.total})",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center", va="bottom", fontsize=9, color=INK_SOFT,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    return fig


def plot_prediction_score_distribution(
    results_df: pd.DataFrame,
    model_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plots the distribution of anomaly scores for TN, TP, FP, FN to diagnose prediction confidence.
    """
    if results_df.empty or "score" not in results_df.columns:
        logger.warning(
            f"No score data provided for score distribution analysis of {model_name}"
        )
        return plt.figure()

    df = results_df.copy()

    def get_quadrant(row):
        if row["label"] == 0 and row["prediction"] == 0:
            return "True Negative (TN)"
        if row["label"] == 1 and row["prediction"] == 1:
            return "True Positive (TP)"
        if row["label"] == 0 and row["prediction"] == 1:
            return "False Positive (FP)"
        if row["label"] == 1 and row["prediction"] == 0:
            return "False Negative (FN)"
        return "Unknown"

    df["Quadrant"] = df.apply(get_quadrant, axis=1)

    plt.figure(figsize=(10, 6))

    palette = {
        "True Negative (TN)": "#4a6670",   # slate teal
        "True Positive (TP)": "#3f5a52",   # pine
        "False Positive (FP)": "#b79155",  # ochre
        "False Negative (FN)": ACCENT,     # terracotta
    }

    order = [q for q in palette if (df["Quadrant"] == q).any()]

    ax = sns.boxplot(
        data=df, x="Quadrant", y="score", order=order,
        hue="Quadrant", palette=palette, legend=False, showfliers=False,
    )
    # Subsample per group so dense classes don't collapse into a solid block.
    sample_idx = (
        df.reset_index()
        .groupby("Quadrant")["index"]
        .apply(lambda s: s.sample(min(len(s), 800), random_state=0))
    )
    strip_df = df.loc[sample_idx.values]
    sns.stripplot(
        data=strip_df, x="Quadrant", y="score", order=order,
        color=INK_SOFT, size=2, alpha=0.25, jitter=0.32,
    )

    ax.set_yscale("log")
    ax.set_title(
        f"Reconstruction Loss Distribution by Prediction Type \u00b7 {model_name}",
    )
    ax.set_xlabel("Prediction Outcome")
    ax.set_ylabel("Anomaly/Reconstruction Score (Log Scale)")
    ax.grid(True, axis="y")
    style_axis(ax)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    return plt.gcf()


def plot_error_scatter(
    results_df: pd.DataFrame,
    model_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Scatter plot of max absolute deviation versus anomaly score, colored by prediction correctness.
    """
    if results_df.empty or "score" not in results_df.columns:
        return plt.figure()

    df = results_df.copy()

    # Calculate signed maximum deviation properly
    def get_max_signed_deviation(row):
        try:
            if "variations_json" in row and pd.notna(row["variations_json"]):
                vars_dict = json.loads(row["variations_json"])
                if vars_dict:
                    return max(vars_dict.values(), key=abs)
            return row["max_deviation"]
        except Exception:
            return row["max_deviation"]

    if "variations_json" in df.columns:
        df["signed_max_dev"] = df.apply(get_max_signed_deviation, axis=1)
    else:
        df["signed_max_dev"] = df["max_deviation"]

    def get_status(row):
        if row["label"] == row["prediction"]:
            return "Correct (TP/TN)"
        elif row["label"] == 1 and row["prediction"] == 0:
            return "Missed Anomaly (FN)"
        else:
            return "False Alarm (FP)"

    df["Status"] = df.apply(get_status, axis=1)

    plt.figure(figsize=(10, 6))

    palette = {
        "Correct (TP/TN)": "#9a938a",       # warm grey
        "False Alarm (FP)": "#b79155",      # ochre
        "Missed Anomaly (FN)": ACCENT,      # terracotta
    }

    ax = sns.scatterplot(
        data=df,
        x="signed_max_dev",
        y="score",
        hue="Status",
        palette=palette,
        alpha=0.7,
        linewidth=0,
        s=40,
    )

    # Draw a horizontal line at the threshold. We calculate an approximate threshold
    # by taking the max score of a TN or min score of a TP.
    tn_scores = df[(df["label"] == 0) & (df["prediction"] == 0)]["score"]
    tp_scores = df[(df["label"] == 1) & (df["prediction"] == 1)]["score"]

    if not tn_scores.empty and not tp_scores.empty:
        approx_threshold = (tn_scores.max() + tp_scores.min()) / 2
        ax.axhline(
            approx_threshold,
            color=INK,
            linestyle="--",
            alpha=0.55,
            linewidth=1.2,
            label="Approx. Threshold",
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

    ax.set_yscale("log")
    ax.set_title(f"Anomaly Score vs Deviation Magnitude \u00b7 {model_name}")
    ax.set_xlabel("Maximum Component Deviation (%)")
    ax.set_ylabel("Reconstruction Score (Log Scale)")
    ax.grid(True, axis="both")
    style_axis(ax)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    return plt.gcf()
