import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Dict, Optional
import time
from collections import defaultdict
import warnings
import numpy as np
warnings.filterwarnings('ignore')

class StructuralBreakGenerator:
    """
    Generator for synthetic time series with various types of structural breaks.

    Break Types Supported:
    1. Mean shift - Abrupt change in the mean level
    2. Variance shift - Change in volatility/variance
    3. Autocorrelation shift - Change in AR(1) parameter
    4. Trend break - Change in trend slope
    5. Combined breaks - Multiple parameter changes simultaneously
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.break_types = [
            'mean_shift', 'variance_shift', 'autocorr_shift',
            'trend_break', 'combined_break'
        ]

    def generate_ar1_segment(self, length: int, phi: float, sigma: float,
                           mu: float = 0.0) -> np.ndarray:
        """Generate AR(1) process: y_t = mu + phi * y_{t-1} + epsilon_t"""
        y = np.zeros(length)
        if length > 0:
            y[0] = np.random.normal(mu, sigma)

            for t in range(1, length):
                y[t] = mu + phi * y[t-1] + np.random.normal(0, sigma)

        return y

    def generate_arima101_segment(self, length: int, phi: float, theta: float,
                                sigma: float, mu: float = 0.0) -> np.ndarray:
        """Generate ARIMA(1,0,1) process"""
        y = np.zeros(length)
        if length > 0:
            epsilon = np.random.normal(0, sigma, length)
            y[0] = mu + epsilon[0]

            for t in range(1, length):
                y[t] = mu + phi * y[t-1] + epsilon[t] + theta * epsilon[t-1]

        return y

    def _select_break_points(self, total_length: int, n_breaks: int,
                             min_break_distance: int = 20,
                             enforce_min_distance: bool = True) -> List[int]:
        """
        Helper to select n_breaks, optionally enforcing a minimum distance between breaks.
        If enforce_min_distance is False, breaks are sampled without spacing constraints
        (aside from avoiding the first and last index).
        """
        if n_breaks == 0:
            return []

        # Unconstrained placement: sample unique break points anywhere inside the series
        if not enforce_min_distance or min_break_distance is None or min_break_distance <= 1:
            max_possible_breaks = max(0, total_length - 2)  # exclude endpoints
            if max_possible_breaks == 0:
                return []
            if n_breaks > max_possible_breaks:
                print(f"Warning: Requested {n_breaks} breaks but only {max_possible_breaks} possible without spacing. Adjusting.")
                n_breaks = max_possible_breaks
            if n_breaks == 0:
                return []
            points = np.random.choice(range(1, total_length - 1), size=n_breaks, replace=False)
            return sorted(points.tolist())

        # Enforced minimum distance path (original behavior)
        required_length = n_breaks * min_break_distance + min_break_distance # n segments need n-1 distances + space for first/last segments
        if total_length < required_length:
             print(f"Warning: Total length {total_length} too short for {n_breaks} breaks with min distance {min_break_distance}. Generating fewer breaks.")
             # Recalculate max possible breaks
             n_breaks = max(0, (total_length - min_break_distance) // min_break_distance) # Ensure at least one segment of min_break_distance
             if n_breaks == 0: return []
             print(f"Adjusted number of breaks to generate: {n_breaks}")


        # Possible points for breaks, respecting min distance from ends
        # Ensure first segment is at least min_break_distance and last segment is at least min_break_distance
        possible_points = list(range(min_break_distance, total_length - min_break_distance))

        if len(possible_points) < n_breaks:
            print(f"Warning: Not enough possible points ({len(possible_points)}) for {n_breaks} breaks with min distance {min_break_distance}. Generating fewer breaks.")
            n_breaks = len(possible_points)
            if n_breaks == 0: return []
            print(f"Adjusted number of breaks to generate: {n_breaks}")


        selected_points = []
        attempts = 0
        max_attempts = n_breaks * 500 # Increase attempts for robustness

        while len(selected_points) < n_breaks and attempts < max_attempts:
            attempts += 1
            # Choose a candidate point
            candidate = np.random.choice(possible_points)

            # Check if candidate is far enough from selected points
            is_valid = all(abs(candidate - sp) >= min_break_distance for sp in selected_points)

            if is_valid:
                selected_points.append(candidate)
                # Optional: remove points too close to the new candidate from possible_points
                # This makes selection faster for fewer points but potentially less random
                # possible_points = [p for p in possible_points if abs(p - candidate) >= min_break_distance]


        if len(selected_points) < n_breaks:
             print(f"Warning: Only generated {len(selected_points)} breaks out of requested {n_breaks} due to distance constraint after {max_attempts} attempts.")

        return sorted(selected_points)


    def generate_mean_shift_series(self, total_length: int = 500,
                                 n_breaks: int = 1,
                                 min_break_distance: int = 20,
                                 enforce_min_distance: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """Generate series with mean shifts"""
        break_points = self._select_break_points(total_length, n_breaks, min_break_distance, enforce_min_distance)

        series = np.zeros(total_length)
        breaks_info = []

        # Generate segments
        start_idx = 0
        base_mean = np.random.normal(0, 0.5)
        # Keep autocorrelation and variance fixed to isolate mean changes
        base_phi = np.random.uniform(-0.3, 0.7)
        base_sigma = np.random.uniform(0.5, 1.5)

        for i, bp in enumerate(break_points + [total_length]):
            segment_length = bp - start_idx
            if segment_length <= 0: continue # Skip empty segments


            # Mean shift magnitude
            if i == 0:
                segment_mean = base_mean
                initial_ar_value = np.random.normal(segment_mean, 1.0) # Initial state for AR(1)
            else:
                shift_magnitude = np.random.uniform(1.0, 3.0) * np.random.choice([-1, 1])
                # Shift relative to the mean of the previous segment, not the last data point
                # This is more consistent with the model definition
                segment_mean = base_mean + shift_magnitude


                breaks_info.append({
                    'break_point': break_points[i-1],
                    'break_type': 'mean_shift',
                    'pre_mean': base_mean,
                    'post_mean': segment_mean,
                    'shift_magnitude': shift_magnitude
                })
                # Initial value for AR(1) should be the end of the previous segment
                # plus the mean shift. However, AR(1) depends on the previous *value*,
                # not just the mean. A simple approach is to make the first point
                # of the new segment equal to the last point of the previous + shift.
                initial_ar_value = series[start_idx-1] + shift_magnitude


            # Generate AR(1) segment with new mean
            noise_segment = self.generate_ar1_segment(segment_length, base_phi, base_sigma, 0)

            # Shift the noise to have the desired mean
            segment = noise_segment - np.mean(noise_segment) + segment_mean # Center noise then add mean

            # Ensure smoother transition at the break point by forcing the first value
            if start_idx > 0 and segment_length > 0:
                 segment[0] = initial_ar_value


            series[start_idx:bp] = segment
            base_mean = segment_mean # Update base mean for next segment
            start_idx = bp


        return series, breaks_info

    def generate_variance_shift_series(self, total_length: int = 500,
                                     n_breaks: int = 1,
                                     min_break_distance: int = 20,
                                     enforce_min_distance: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """Generate series with variance/volatility shifts"""
        break_points = self._select_break_points(total_length, n_breaks, min_break_distance, enforce_min_distance)

        series = np.zeros(total_length)
        breaks_info = []

        start_idx = 0
        base_sigma = np.random.uniform(0.8, 1.2)
        current_mean = np.random.normal(0, 0.3) # Base mean for segments
        # Fix phi across segments to keep autocorrelation constant
        base_phi = np.random.uniform(-0.2, 0.6)

        for i, bp in enumerate(break_points + [total_length]):
            segment_length = bp - start_idx
            if segment_length <= 0: continue # Skip empty segments

            if i == 0:
                segment_sigma = base_sigma
                initial_ar_value = np.random.normal(current_mean, segment_sigma) # Initial state for AR(1)
            else:
                # Variance shift
                # Multiplicative variance shift (2x to 5x change)
                variance_multiplier = np.random.uniform(2.0, 5.0)
                if np.random.random() > 0.5:  # 50% chance of volatility increase
                    segment_sigma = base_sigma * variance_multiplier
                else:  # 50% chance of volatility decrease
                    segment_sigma = base_sigma / variance_multiplier

                breaks_info.append({
                    'break_point': break_points[i-1],
                    'break_type': 'variance_shift',
                    'pre_variance': base_sigma**2,
                    'post_variance': segment_sigma**2,
                    'variance_ratio': (segment_sigma/base_sigma)**2
                })
                base_sigma = segment_sigma # Update base sigma for next segment
                initial_ar_value = series[start_idx-1] # Initial value for AR(1) is end of previous segment


            # Generate segment
            # Use current_mean, don't shift mean in variance break
            segment = self.generate_ar1_segment(segment_length, base_phi, segment_sigma, current_mean)

            # Adjust initial value to continue from previous segment's end
            if start_idx > 0 and segment_length > 0:
                 segment[0] = initial_ar_value

            series[start_idx:bp] = segment
            start_idx = bp

        return series, breaks_info

    def generate_autocorr_shift_series(self, total_length: int = 500,
                                     n_breaks: int = 1,
                                     min_break_distance: int = 20,
                                     enforce_min_distance: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """Generate series with autocorrelation parameter shifts"""
        break_points = self._select_break_points(total_length, n_breaks, min_break_distance, enforce_min_distance)

        series = np.zeros(total_length)
        breaks_info = []

        start_idx = 0
        base_phi = np.random.uniform(0.1, 0.4)
        current_mean = np.random.normal(0, 0.2) # Base mean for segments
        current_sigma = np.random.uniform(0.8, 1.2) # Base sigma for segments


        for i, bp in enumerate(break_points + [total_length]):
            segment_length = bp - start_idx
            if segment_length <= 0: continue # Skip empty segments

            if i == 0:
                segment_phi = base_phi
                initial_ar_value = np.random.normal(current_mean, current_sigma) # Initial state for AR(1)
            else:
                # Autocorrelation shift
                # Significant change in autocorrelation (at least 0.3 difference)
                phi_change = np.random.uniform(0.3, 0.6) * np.random.choice([-1, 1])
                segment_phi = np.clip(base_phi + phi_change, -0.8, 0.9)

                breaks_info.append({
                    'break_point': break_points[i-1],
                    'break_type': 'autocorr_shift',
                    'pre_autocorr': base_phi,
                    'post_autocorr': segment_phi,
                    'autocorr_change': phi_change
                })
                base_phi = segment_phi  # Update for next segment
                initial_ar_value = series[start_idx-1] # Initial value for AR(1) is end of previous segment


            # Generate segment
            # Use current_mean and current_sigma, don't shift them in autocorr break
            segment = self.generate_ar1_segment(segment_length, segment_phi, current_sigma, current_mean)

            # Adjust initial value to continue from previous segment's end
            if start_idx > 0 and segment_length > 0:
                 segment[0] = initial_ar_value

            series[start_idx:bp] = segment
            start_idx = bp

        return series, breaks_info

    def generate_trend_break_series(self, total_length: int = 500,
                                  n_breaks: int = 1,
                                  min_break_distance: int = 20,
                                  enforce_min_distance: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """Generate series with trend breaks"""
        break_points = self._select_break_points(total_length, n_breaks, min_break_distance, enforce_min_distance)

        series = np.zeros(total_length)
        breaks_info = []

        start_idx = 0
        base_trend = np.random.uniform(-0.01, 0.01)
        cumulative_level = 0 # Start at 0
        # Fix AR noise parameters to keep variance/autocorrelation constant
        base_phi = np.random.uniform(0.1, 0.5)
        base_sigma = np.random.uniform(0.5, 1.0)

        for i, bp in enumerate(break_points + [total_length]):
            segment_length = bp - start_idx
            if segment_length <= 0: continue # Skip empty segments


            # Trend shift
            if i == 0:
                segment_trend = base_trend
                # Start the first segment at cumulative_level (which is 0)
                segment_start_level = cumulative_level
            else:
                trend_change = np.random.uniform(0.02, 0.05) * np.random.choice([-1, 1])
                segment_trend = base_trend + trend_change

                breaks_info.append({
                    'break_point': break_points[i-1],
                    'break_type': 'trend_break',
                    'pre_trend': base_trend,
                    'post_trend': segment_trend,
                    'trend_change': trend_change
                })
                base_trend = segment_trend
                # The starting level for the new segment is the end level of the previous segment
                segment_start_level = series[start_idx-1]


            # Generate segment with trend
            t_vals = np.arange(segment_length)
            trend_component = segment_start_level + segment_trend * t_vals

            # Add AR(1) noise around trend
            noise = self.generate_ar1_segment(segment_length, base_phi, base_sigma, 0) # Noise around 0 mean

            segment = trend_component + noise
            series[start_idx:bp] = segment

            cumulative_level = segment[-1]  # Update cumulative level for next segment
            start_idx = bp

        return series, breaks_info

    def generate_combined_break_series(self, total_length: int = 500,
                                     n_breaks: int = 1,
                                     min_break_distance: int = 20,
                                     enforce_min_distance: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """Generate series with combined parameter breaks"""
        break_points = self._select_break_points(total_length, n_breaks, min_break_distance, enforce_min_distance)

        series = np.zeros(total_length)
        breaks_info = []

        start_idx = 0
        base_params = {
            'mu': np.random.normal(0, 0.3),
            'phi': np.random.uniform(0.1, 0.4),
            'sigma': np.random.uniform(0.8, 1.2)
        }

        for i, bp in enumerate(break_points + [total_length]):
            segment_length = bp - start_idx
            if segment_length <= 0: continue # Skip empty segments


            if i == 0:
                segment_params = base_params.copy()
                initial_ar_value = np.random.normal(segment_params['mu'], segment_params['sigma']) # Start AR(1) with its own mean
            else:
                # Randomly change 2-3 parameters simultaneously
                changes = {}
                params_to_change = np.random.choice(
                    ['mu', 'phi', 'sigma'],
                    size=np.random.choice([2, 3]),
                    replace=False
                )

                segment_params = base_params.copy()

                if 'mu' in params_to_change:
                    mu_change = np.random.uniform(1.0, 2.5) * np.random.choice([-1, 1])
                    segment_params['mu'] += mu_change
                    changes['mean_change'] = mu_change

                if 'phi' in params_to_change:
                    phi_change = np.random.uniform(0.3, 0.5) * np.random.choice([-1, 1])
                    segment_params['phi'] = np.clip(base_params['phi'] + phi_change, -0.8, 0.9)
                    changes['autocorr_change'] = phi_change

                if 'sigma' in params_to_change:
                    sigma_multiplier = np.random.uniform(2.0, 4.0)
                    if np.random.random() > 0.5:
                        segment_params['sigma'] *= sigma_multiplier
                    else:
                        segment_params['sigma'] /= sigma_multiplier
                    changes['variance_ratio'] = (segment_params['sigma']/base_params['sigma'])**2

                breaks_info.append({
                    'break_point': break_points[i-1],
                    'break_type': 'combined_break',
                    'changed_parameters': list(params_to_change),
                    'pre_params': base_params.copy(),
                    'post_params': segment_params.copy(),
                    'changes': changes
                })

                base_params = segment_params.copy()
                # Initial value for the new segment should continue from the end of the previous one
                initial_ar_value = series[start_idx-1]


            # Generate segment
            # Need to generate AR(1) noise around the new mean, starting from the previous segment's end
            noise_segment = self.generate_ar1_segment(segment_length, segment_params['phi'], segment_params['sigma'], 0) # Noise around 0

            # Shift the noise to have the desired mean
            segment = noise_segment - np.mean(noise_segment) + segment_params['mu']

            # Ensure smoother transition at the break point by forcing the first value
            if start_idx > 0 and segment_length > 0:
                 segment[0] = initial_ar_value


            series[start_idx:bp] = segment
            start_idx = bp

        return series, breaks_info


    def generate_single_series(self, series_length: int = 500,
                             max_breaks: int = 3,
                             min_break_distance: int = 20,
                             enforce_min_distance: bool = True) -> Dict:
        """Generate a single time series with random break type and count"""
        if enforce_min_distance:
            # Ensure series is long enough for breaks + minimum distance + padding at ends
            # n breaks create n+1 segments. Minimum length is n_breaks * min_dist + 2*min_dist (for first/last segments)
            min_required_length = max_breaks * min_break_distance + 2 * min_break_distance
            if series_length < min_required_length:
                # If series is too short for max_breaks, calculate the max possible breaks
                max_possible_breaks = max(0, (series_length - 2 * min_break_distance) // min_break_distance)
                print(f"Warning: Series length {series_length} is too short for max breaks {max_breaks} with min distance {min_break_distance}. Max possible breaks: {max_possible_breaks}")
                n_breaks = np.random.randint(0, max_possible_breaks + 1)
            else:
                n_breaks = np.random.randint(0, max_breaks + 1)
        else:
            # Unconstrained: limited only by available interior points
            max_possible_breaks = max(0, min(max_breaks, series_length - 2))
            n_breaks = np.random.randint(0, max_possible_breaks + 1)


        if n_breaks == 0:
            # No breaks - pure AR(1) or ARIMA(1,0,1)
            if np.random.random() > 0.5:
                phi = np.random.uniform(-0.3, 0.7)
                sigma = np.random.uniform(0.5, 1.5)
                mu = np.random.normal(0, 0.5)
                series = self.generate_ar1_segment(series_length, phi, sigma, mu)
                model_type = 'AR(1)'
            else:
                phi = np.random.uniform(-0.3, 0.7)
                theta = np.random.uniform(-0.5, 0.5)
                sigma = np.random.uniform(0.5, 1.5)
                mu = np.random.normal(0, 0.5)
                series = self.generate_arima101_segment(series_length, phi, theta, sigma, mu)
                model_type = 'ARIMA(1,0,1)'

            return {
                'series': series,
                'n_breaks': 0,
                'break_points': [],
                'breaks_info': [],
                'model_type': model_type
            }

        # Choose random break type
        break_type = np.random.choice(self.break_types)

        if break_type == 'mean_shift':
            series, breaks_info = self.generate_mean_shift_series(series_length, n_breaks, min_break_distance, enforce_min_distance)
        elif break_type == 'variance_shift':
            series, breaks_info = self.generate_variance_shift_series(series_length, n_breaks, min_break_distance, enforce_min_distance)
        elif break_type == 'autocorr_shift':
            series, breaks_info = self.generate_autocorr_shift_series(series_length, n_breaks, min_break_distance, enforce_min_distance)
        elif break_type == 'trend_break':
            series, breaks_info = self.generate_trend_break_series(series_length, n_breaks, min_break_distance, enforce_min_distance)
        else:  # combined_break
            series, breaks_info = self.generate_combined_break_series(series_length, n_breaks, min_break_distance, enforce_min_distance)

        # Ensure break points match the info returned
        break_points = sorted([info['break_point'] for info in breaks_info])


        return {
            'series': series,
            'n_breaks': len(break_points),
            'break_points': break_points,
            'breaks_info': breaks_info,
            'primary_break_type': break_type,
            'model_type': f'AR(1)_with_{break_type}'
        }

    def generate_dataset(self, n_series: int = 100000, series_length: int = 500,
                        max_breaks: int = 3, min_break_distance: int = 20,
                        enforce_min_distance: bool = True,
                        min_break_distance_pct_choices: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Generate complete dataset for thesis.

        Args:
            n_series: number of series to generate.
            series_length: length of each series.
            max_breaks: maximum breaks per series.
            min_break_distance: fixed minimum distance between breaks (used when pct choices are not provided).
            enforce_min_distance: if False, breaks can cluster with no spacing constraint.
            min_break_distance_pct_choices: optional list of percentages (e.g., [0.05, 0.1])
                to sample per-series minimum distances as a fraction of series_length.
        """
        print(f"Generating {n_series:,} synthetic time series...")
        print(f"Series Length: {series_length}, Max Breaks: {max_breaks}, Min Break Distance: {min_break_distance}, Enforce Min Distance: {enforce_min_distance}")


        dataset = []

        for i in range(n_series):
            if (i + 1) % 10000 == 0:
                print(f"Generated {i + 1:,} series...")

            # Sample per-series min distance if percentage choices are provided
            if min_break_distance_pct_choices:
                pct_choice = np.random.choice(min_break_distance_pct_choices)
                per_series_min_dist = max(1, int(series_length * pct_choice))
            else:
                per_series_min_dist = min_break_distance

            series_data = self.generate_single_series(series_length, max_breaks, per_series_min_dist, enforce_min_distance)

            # Create row for dataset
            row = {
                'series_id': i,
                'n_breaks': series_data['n_breaks'],
                'break_points': series_data['break_points'],
                'primary_break_type': series_data.get('primary_break_type', 'none'),
                'model_type': series_data['model_type']
            }

            # Add time series values
            for t in range(series_length):
                row[f't_{t}'] = series_data['series'][t]

            # Add detailed break information
            row['breaks_info'] = series_data['breaks_info']

            dataset.append(row)

        df = pd.DataFrame(dataset)

        print("\nDataset Summary:")
        print(f"Total series: {len(df):,}")
        print(f"Break count distribution:")
        print(df['n_breaks'].value_counts().sort_index())
        print(f"\nBreak type distribution:")
        print(df['primary_break_type'].value_counts())

        return df

    def plot_examples(self, dataset: pd.DataFrame, n_examples: int = 5):
        """Plot examples of each break type"""
        break_types = ['none', 'mean_shift', 'variance_shift', 'autocorr_shift',
                      'trend_break', 'combined_break']

        # Filter dataset to ensure we have examples of each type, handling cases where type counts might be low
        example_series_ids = {}
        for break_type in break_types:
            if break_type == 'none':
                subset = dataset[dataset['n_breaks'] == 0]
            else:
                subset = dataset[dataset['primary_break_type'] == break_type]

            if not subset.empty:
                # Get first series ID of this type
                example_series_ids[break_type] = subset.iloc[0].name # Use .name for index


        # Determine number of plots needed (up to 6)
        num_plots = min(len(example_series_ids), 6)
        if num_plots == 0:
            print("No example series found with breaks to plot.")
            return # Exit if no examples

        # Determine grid size
        n_rows = min(2, (num_plots + 2) // 3)
        n_cols = min(3, num_plots)


        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        # Handle case where axes might not be a 2D array (e.g., 1 row or 1 column)
        if not isinstance(axes, np.ndarray):
             axes = np.array([axes]) # Make it an array so flatten works
        axes = axes.flatten()


        for i, (break_type, series_idx) in enumerate(example_series_ids.items()):
            if i >= len(axes): break

            example = dataset.loc[series_idx] # Use .loc for index lookup

            # Extract time series
            series_cols = [col for col in example.index if col.startswith('t_')]
            series = example[series_cols].values

            axes[i].plot(series, 'b-', alpha=0.7)

            # Mark break points
            if example['n_breaks'] > 0:
                for bp in example['break_points']:
                    axes[i].axvline(x=bp, color='red', linestyle='--', alpha=0.8)

            axes[i].set_title(f"{break_type.replace('_', ' ').title()}\n"
                            f"({example['n_breaks']} breaks)")
            axes[i].grid(True, alpha=0.3)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])


        plt.tight_layout()
        plt.suptitle('Examples of Synthetic Time Series with Different Break Types',
                    y=1.02, fontsize=14)
        plt.show()

# Example usage and demonstration
if __name__ == "__main__":
    # Create generator
    generator = StructuralBreakGenerator(seed=42)

    # Generate small example dataset for demonstration
    print("Generating example dataset...")
    # Added min_break_distance to the example generation
    small_dataset = generator.generate_dataset(n_series=1000, series_length=500, max_breaks=3, min_break_distance=50)

    # Show examples
    generator.plot_examples(small_dataset)

    # Save example dataset
    print("\nSaving example dataset...")
    small_dataset.to_pickle('synthetic_breaks_example_1k_min_dist.pkl') # Changed filename

    # Show detailed example
    print("\nExample series with detailed break information:")
    # Filter for series with breaks before trying to access the first one
    series_with_breaks = small_dataset[small_dataset['n_breaks'] > 0]
    if not series_with_breaks.empty:
        example_with_breaks = series_with_breaks.iloc[0]
        print(f"Series ID: {example_with_breaks['series_id']}")
        print(f"Number of breaks: {example_with_breaks['n_breaks']}")
        print(f"Break points: {example_with_breaks['break_points']}")
        print(f"Primary break type: {example_with_breaks['primary_break_type']}")
        print(f"Detailed break info:")
        for i, break_info in enumerate(example_with_breaks['breaks_info']):
            print(f"  Break {i+1}: {break_info}")
    else:
        print("No series with breaks found in the example dataset.")