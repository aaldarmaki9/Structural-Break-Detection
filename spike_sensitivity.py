# Spike-and-Slab two-stage sensitivity analysis (reusable)
from itertools import product
import time
import numpy as np
import pandas as pd

try:
    SpikeAndSlabDetector
    evaluate_detection_performance
except NameError:
    from spike_slab_detector import SpikeAndSlabDetector, evaluate_detection_performance


def subset_changepoints_with_threshold(
    ratio: np.ndarray,
    threshold: float,
    del_thr: int = 5,
    edge_margin: int = 5,
):
    """
    Python version of subset_changepoints with configurable p_j threshold.
    Mirrors the block-merge logic used by the R helper.
    """
    ratio = np.asarray(ratio, dtype=float)
    first = np.where(ratio >= threshold)[0]

    if len(first) == 0:
        return []

    n_c = len(first)
    low = first - del_thr

    if n_c > 1:
        id_split = np.where((low[1:] <= first[:-1]) == False)[0]
        i_low_bl = first[np.r_[0, id_split + 1]]
        i_up_bl = first[np.r_[id_split, n_c - 1]]
    else:
        i_low_bl = np.array([first[0]])
        i_up_bl = np.array([first[0]])

    cps = []
    for lo, up in zip(i_low_bl, i_up_bl):
        block = ratio[lo:up + 1]
        rel = np.where(block == np.max(block))[0]
        chosen_rel = rel[int(np.ceil(len(rel) / 2.0) - 1)]
        cps.append(int(lo + chosen_rel))

    cps = [cp for cp in cps if cp > edge_margin and cp < (len(ratio) - edge_margin)]
    return cps


def run_spike_slab_stage1_sweep(
    df: pd.DataFrame,
    q_grid,
    del_threshold_grid,
    tolerance_percentage: float = 1.0,
    reference_choice: dict | None = None,
    tau2=None,
    tau2_spike=None,
    tau2_slab=None,
    sigma=None,
    pj_threshold: float = 0.5,
):
    """
    Stage 1 sweep over q and del_threshold.
    Returns sorted results DataFrame with is_reference_choice.
    """
    series_columns = [c for c in df.columns if c.startswith("t_")]
    series_length = len(series_columns)
    X = df[series_columns].values.astype(float)
    true_breaks = df["break_points"].tolist()

    total_runs = len(q_grid) * len(del_threshold_grid)
    rows = []
    run_idx = 0

    for q, del_threshold in product(q_grid, del_threshold_grid):
        run_idx += 1
        print(f"[{run_idx:02d}/{total_runs}] q={q}, del_threshold={del_threshold}")

        detector = SpikeAndSlabDetector(
            q=q,
            tau2=tau2,
            tau2_spike=tau2_spike,
            tau2_slab=tau2_slab,
            sigma=sigma,
            del_threshold=del_threshold,
            pj_threshold=pj_threshold,
        )

        detected_breaks = []
        t0 = time.time()
        for y in X:
            try:
                detected_breaks.append(detector.detect_breaks_fast(y))
            except Exception:
                detected_breaks.append([])
        elapsed = time.time() - t0

        metrics = evaluate_detection_performance(
            true_breaks,
            detected_breaks,
            series_length,
            tolerance_percentage=tolerance_percentage,
        )

        rows.append({
            "q": q,
            "del_threshold": del_threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "avg_localization_error": metrics["avg_localization_error"],
            "true_positives": metrics["true_positives"],
            "false_positives": metrics["false_positives"],
            "false_negatives": metrics["false_negatives"],
            "mean_detected_breaks_per_series": float(np.mean([len(b) for b in detected_breaks])),
            "runtime_s": elapsed,
        })

    out = pd.DataFrame(rows).sort_values(
        ["f1_score", "recall", "precision"], ascending=False
    ).reset_index(drop=True)

    if reference_choice is not None:
        out["is_reference_choice"] = (
            (out["q"] == reference_choice["q"])
            & (out["del_threshold"] == reference_choice["del_threshold"])
        )
    else:
        out["is_reference_choice"] = False

    return out


def run_spike_slab_stage2_pj_sweep(
    df: pd.DataFrame,
    fixed_q: float,
    fixed_del_threshold: int,
    pj_grid,
    reference_pj: float = 0.5,
    tolerance_percentage: float = 1.0,
    tau2=None,
    tau2_spike=None,
    tau2_slab=None,
    sigma=None,
    edge_margin: int = 5,
):
    """
    Stage 2 sweep over p_j threshold with fixed Stage 1 settings.
    Reuses inclusion probabilities for efficiency.
    """
    series_columns = [c for c in df.columns if c.startswith("t_")]
    series_length = len(series_columns)
    X = df[series_columns].values.astype(float)
    true_breaks = df["break_points"].tolist()

    detector = SpikeAndSlabDetector(
        q=fixed_q,
        tau2=tau2,
        tau2_spike=tau2_spike,
        tau2_slab=tau2_slab,
        sigma=sigma,
        del_threshold=fixed_del_threshold,
        pj_threshold=0.5,  # not used directly here; we threshold ratios in Python
    )

    print("Computing marginal inclusion probabilities once for all series...")
    ratio_list = []
    for i, y in enumerate(X):
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(X)} series...")
        try:
            probs = detector.get_inclusion_probabilities(y)
            ratio = np.asarray(probs["ratio"], dtype=float)
        except Exception:
            ratio = np.zeros(len(y), dtype=float)
        ratio_list.append(ratio)

    rows = []
    for pj in pj_grid:
        t0 = time.time()
        detected_breaks = [
            subset_changepoints_with_threshold(
                r,
                threshold=pj,
                del_thr=fixed_del_threshold,
                edge_margin=edge_margin,
            )
            for r in ratio_list
        ]
        elapsed = time.time() - t0

        metrics = evaluate_detection_performance(
            true_breaks,
            detected_breaks,
            series_length,
            tolerance_percentage=tolerance_percentage,
        )

        rows.append({
            "p_j_threshold": pj,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "avg_localization_error": metrics["avg_localization_error"],
            "true_positives": metrics["true_positives"],
            "false_positives": metrics["false_positives"],
            "false_negatives": metrics["false_negatives"],
            "mean_detected_breaks_per_series": float(np.mean([len(b) for b in detected_breaks])),
            "runtime_s": elapsed,
        })

    out = pd.DataFrame(rows).sort_values(
        ["f1_score", "recall", "precision"], ascending=False
    ).reset_index(drop=True)
    out["is_reference_choice"] = out["p_j_threshold"].eq(reference_pj)
    return out


