from itertools import product
import time
import rpy2.robjects as ro
import numpy as np
import pandas as pd
from wbs2_detector import evaluate_detection_performance

def run_wbs2_sensitivity_sweep(
    df: pd.DataFrame,
    th_const_min_mult_grid,
    lambda_grid,
    cusums_grid,
    tolerance_percentage: float = 1.0,
    reference_choice=None,
    universal: bool = True,
):
    """Run reusable WBS2 sensitivity sweep and return ranked results."""
    series_columns = [c for c in df.columns if c.startswith('t_')]
    series_length = len(series_columns)
    X = df[series_columns].values.astype(float)
    true_breaks = df['break_points'].tolist()

    total_runs = len(th_const_min_mult_grid) * len(lambda_grid) * len(cusums_grid)
    rows = []
    run_idx = 0

    for th_const_min_mult, lambda_param, cusums in product(th_const_min_mult_grid, lambda_grid, cusums_grid):
        run_idx += 1
        print(f"[{run_idx:02d}/{total_runs}] th_const_min_mult={th_const_min_mult}, lambda={lambda_param}, cusums={cusums}")

        detected_breaks = []
        t0 = time.time()
        for y in X:
            try:
                r_series = ro.FloatVector(y)
                result = ro.r['wbs.sdll.cpt'](
                    r_series,
                    **{
                        'universal': bool(universal),
                        'th.const.min.mult': float(th_const_min_mult),
                        'lambda': float(lambda_param),
                        'cusums': cusums,
                    }
                )
                cpt = np.array(result.rx2('cpt'), dtype=int)
                if len(cpt) > 0:
                    cpt = cpt - 1
                detected_breaks.append(cpt.tolist())
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
            'th_const_min_mult': th_const_min_mult,
            'lambda_param': lambda_param,
            'cusums': cusums,
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
            (out['th_const_min_mult'] == reference_choice['th_const_min_mult'])
            & (out['lambda_param'] == reference_choice['lambda_param'])
            & (out['cusums'] == reference_choice['cusums'])
        )
    else:
        out['is_reference_choice'] = False

    return out