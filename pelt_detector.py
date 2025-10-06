import numpy as np
import pandas as pd
from typing import List, Tuple
import ruptures as rpt
from collections import defaultdict

# Reuse evaluation from wbs2_detector to keep metrics consistent
from wbs2_detector import evaluate_detection_performance, plot_detection_examples


class PELTDetector:
    """PELT detector wrapper using ruptures for consistency with other detectors."""

    def __init__(self, model: str = "l2", min_size: int = 5, penalty: float = 10.0):
        self.model = model
        self.min_size = min_size
        self.penalty = penalty

    def detect_breaks(self, series: np.ndarray) -> List[int]:
        """Return break indices (0-indexed, like our other detectors)."""
        series = np.asarray(series, dtype=float)
        algo = rpt.Pelt(model=self.model, min_size=self.min_size).fit(series)
        # ruptures returns last index n; remove and convert to 0-indexed positions
        bkps = algo.predict(pen=self.penalty)
        bkps = [b - 1 for b in bkps if b < len(series)]
        return bkps


def run_pelt_benchmark(model: str = "l2", min_size: int = 10, penalty_500: float = 8.0, penalty_1000: float = 10.0) -> Tuple:
    """Run PELT benchmarks on the 500 and 1000 synthetic datasets.

    Returns tuple of datasets, detected lists, results dicts, and detection times like wbs2.
    """
    import time

    # Load datasets
    dataset_500 = pd.read_pickle('synthetic_breaks_100_500_min50.pkl')
    dataset_1000 = pd.read_pickle('synthetic_breaks_100_1000_min100.pkl')

    # Prepare series columns
    series_columns_500 = [f't_{i}' for i in range(500)]
    series_columns_1000 = [f't_{i}' for i in range(1000)]

    # --- 500 ---
    print("=== PELT Performance Benchmark ===\n")
    print("--- Running PELT on 500 Length Series ---")
    series_len = 500
    time_series_data = dataset_500[series_columns_500].values
    true_breaks = dataset_500['break_points'].tolist()

    detector = PELTDetector(model=model, min_size=min_size, penalty=penalty_500)

    start_time = time.time()
    detected_breaks_500 = []
    detection_times_500 = []
    for i, series in enumerate(time_series_data):
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1:,} series...")
        t0 = time.time()
        br = detector.detect_breaks(series)
        detection_times_500.append(time.time() - t0)
        detected_breaks_500.append(br)

    print(f"Detection completed for 500 series in {time.time() - start_time:.2f} seconds")
    if detection_times_500:
        print(f"Average time per 500 series (processed): {np.mean(detection_times_500)*1000:.2f} ms")
    print()

    results_500 = evaluate_detection_performance(
        true_breaks[:len(detected_breaks_500)],
        detected_breaks_500,
        series_len,
        tolerance_percentage=1.0,
    )

    print("=== PELT Performance Results (500 Series, 1% Tolerance) ===")
    print(f"Tolerance used: {results_500['tolerance_used']}")
    print(f"Overall Precision: {results_500['precision']:.3f}")
    print(f"Overall Recall: {results_500['recall']:.3f}")
    print(f"Overall F1-Score: {results_500['f1_score']:.3f}\n")
    print(f"Average Precision: {results_500['avg_precision']:.3f}")
    print(f"Average Recall: {results_500['avg_recall']:.3f}")
    print(f"Average F1-Score: {results_500['avg_f1']:.3f}\n")
    print(f"Average Localization Error: {results_500['avg_localization_error']:.1f} time points\n")
    print("Confusion Matrix:")
    print(f"True Positives: {results_500['true_positives']}")
    print(f"False Positives: {results_500['false_positives']}")
    print(f"False Negatives: {results_500['false_negatives']}")

    # --- 1000 ---
    print("\n--- Running PELT on 1000 Length Series ---")
    series_len = 1000
    time_series_data = dataset_1000[series_columns_1000].values
    true_breaks = dataset_1000['break_points'].tolist()

    detector = PELTDetector(model=model, min_size=min_size, penalty=penalty_1000)

    start_time = time.time()
    detected_breaks_1000 = []
    detection_times_1000 = []
    for i, series in enumerate(time_series_data):
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1:,} series...")
        t0 = time.time()
        br = detector.detect_breaks(series)
        detection_times_1000.append(time.time() - t0)
        detected_breaks_1000.append(br)

    print(f"Detection completed for 1000 series in {time.time() - start_time:.2f} seconds")
    if detection_times_1000:
        print(f"Average time per 1000 series (processed): {np.mean(detection_times_1000)*1000:.2f} ms")
    print()

    results_1000 = evaluate_detection_performance(
        true_breaks[:len(detected_breaks_1000)],
        detected_breaks_1000,
        series_len,
        tolerance_percentage=1.0,
    )

    print("=== PELT Performance Results (1000 Series, 1% Tolerance) ===")
    print(f"Tolerance used: {results_1000['tolerance_used']}")
    print(f"Overall Precision: {results_1000['precision']:.3f}")
    print(f"Overall Recall: {results_1000['recall']:.3f}")
    print(f"Overall F1-Score: {results_1000['f1_score']:.3f}\n")
    print(f"Average Precision: {results_1000['avg_precision']:.3f}")
    print(f"Average Recall: {results_1000['avg_recall']:.3f}")
    print(f"Average F1-Score: {results_1000['avg_f1']:.3f}\n")
    print(f"Average Localization Error: {results_1000['avg_localization_error']:.1f} time points\n")
    print("Confusion Matrix:")
    print(f"True Positives: {results_1000['true_positives']}")
    print(f"False Positives: {results_1000['false_positives']}")
    print(f"False Negatives: {results_1000['false_negatives']}")

    # Optional: plot examples
    if detected_breaks_500:
        plot_detection_examples(dataset_500.iloc[:len(detected_breaks_500)], detected_breaks_500, n_examples=min(6, len(detected_breaks_500)))

    if detected_breaks_1000:
        plot_detection_examples(dataset_1000.iloc[:len(detected_breaks_1000)], detected_breaks_1000, n_examples=min(6, len(detected_breaks_1000)))

    return dataset_500, detected_breaks_500, results_500, \
           dataset_1000, detected_breaks_1000, results_1000


def run_pelt_benchmark_real(penalty: float = 10.0, model: str = "l2", min_size: int = 10) -> Tuple:
    """Run PELT on the validated real stock dataset and report metrics.

    Uses the same 1% tolerance evaluation. Returns (dataset, detected_breaks, results).
    """
    import time

    dataset = pd.read_pickle('validated_stock_breaks.pkl')
    series_cols = [c for c in dataset.columns if c.startswith('t_')]
    series_len = len(series_cols)
    if series_len == 0:
        print("Dataset appears empty or without t_ columns.")
        return dataset, [], {}

    print("=== PELT on Validated Real Stock Dataset ===")
    print(f"Total series: {len(dataset)}  |  Length: {series_len}")

    time_series_data = dataset[series_cols].values
    true_breaks = dataset['break_points'].tolist()

    detector = PELTDetector(model=model, min_size=min_size, penalty=penalty)

    start_time = time.time()
    detected_breaks = []
    detection_times = []
    for i, series in enumerate(time_series_data):
        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1:,} series...")
        t0 = time.time()
        br = detector.detect_breaks(series)
        detection_times.append(time.time() - t0)
        detected_breaks.append(br)

    print(f"Detection completed for {len(dataset)} series in {time.time() - start_time:.2f} seconds")
    if detection_times:
        print(f"Average time per series: {np.mean(detection_times)*1000:.2f} ms")
    print()

    results = evaluate_detection_performance(true_breaks, detected_breaks, series_len, tolerance_percentage=1.0)

    print("=== PELT Performance Results (Real, 1% Tolerance) ===")
    print(f"Tolerance used: {results['tolerance_used']}")
    print(f"Overall Precision: {results['precision']:.3f}")
    print(f"Overall Recall: {results['recall']:.3f}")
    print(f"Overall F1-Score: {results['f1_score']:.3f}\n")
    print(f"Average Precision: {results['avg_precision']:.3f}")
    print(f"Average Recall: {results['avg_recall']:.3f}")
    print(f"Average F1-Score: {results['avg_f1']:.3f}\n")
    print(f"Average Localization Error: {results['avg_localization_error']:.1f} time points\n")
    print("Confusion Matrix:")
    print(f"True Positives: {results['true_positives']}")
    print(f"False Positives: {results['false_positives']}")
    print(f"False Negatives: {results['false_negatives']}")

    # Optional: plot examples
    if detected_breaks:
        plot_detection_examples(dataset.iloc[:len(detected_breaks)], detected_breaks, n_examples=min(6, len(detected_breaks)))

    return dataset, detected_breaks, results


