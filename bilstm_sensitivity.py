# BiLSTM sensitivity sweep (reusable)
from itertools import product
import time
import numpy as np
import pandas as pd


def eval_detection_micro(true_breaks_list, detected_breaks_list, series_length, tolerance_percentage=1.0):
    """Micro metrics (overall TP/FP/FN) + avg localization error."""
    tolerance = max(1, int(series_length * (tolerance_percentage / 100.0)))
    tp_total, fp_total, fn_total = 0, 0, 0
    localization_errors = []

    for true_bp, det_bp in zip(true_breaks_list, detected_breaks_list):
        true_bp = true_bp if isinstance(true_bp, (list, np.ndarray)) else []
        det_bp = det_bp if isinstance(det_bp, (list, np.ndarray)) else []

        tp = 0
        matched_true = set()
        for d in det_bp:
            for k, t in enumerate(true_bp):
                if k not in matched_true and abs(d - t) <= tolerance:
                    matched_true.add(k)
                    tp += 1
                    localization_errors.append(abs(d - t))
                    break

        fp = len(det_bp) - tp
        fn = len(true_bp) - tp

        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_localization_error': float(np.mean(localization_errors)) if localization_errors else 0.0,
        'true_positives': tp_total,
        'false_positives': fp_total,
        'false_negatives': fn_total,
        'tolerance_used': tolerance,
    }


def run_bilstm_sensitivity_sweep(
    df: pd.DataFrame,
    detector,
    threshold_grid,
    stride_grid,
    tolerance_percentage: float = 1.0,
    reference_choice: dict | None = None,
):
    series_columns = [c for c in df.columns if c.startswith('t_')]
    series_length = len(series_columns)
    X = df[series_columns].values.astype(float)
    true_breaks = df['break_points'].tolist()

    rows = []
    total_runs = len(threshold_grid) * len(stride_grid)
    run_idx = 0

    for thr, stride in product(threshold_grid, stride_grid):
        run_idx += 1
        print(f"[{run_idx}/{total_runs}] threshold={thr}, stride={stride}")

        detected_breaks = []
        t0 = time.time()
        for y in X:
            try:
                detected_breaks.append(detector.detect_breaks(y, threshold=thr, stride=stride))
            except Exception:
                detected_breaks.append([])
        elapsed = time.time() - t0

        metrics = eval_detection_micro(
            true_breaks, detected_breaks, series_length, tolerance_percentage=tolerance_percentage
        )

        rows.append({
            'detection_threshold': thr,
            'detection_stride': stride,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'avg_localization_error': metrics['avg_localization_error'],
            'true_positives': metrics['true_positives'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives'],
            'mean_detected_breaks_per_series': float(np.mean([len(b) for b in detected_breaks])),
            'runtime_s': elapsed,
        })

    out = pd.DataFrame(rows).sort_values(
        ['f1_score', 'recall', 'precision'], ascending=False
    ).reset_index(drop=True)

    if reference_choice is not None:
        out['is_reference_choice'] = (
            (out['detection_threshold'] == reference_choice['detection_threshold']) &
            (out['detection_stride'] == reference_choice['detection_stride'])
        )
    else:
        out['is_reference_choice'] = False

    return out
