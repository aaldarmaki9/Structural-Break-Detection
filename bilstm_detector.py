import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from typing import List, Tuple, Dict
import time
from synthetic_data_generator import StructuralBreakGenerator
from collections import defaultdict

def calculate_localization_errors(true_breaks: List[int], detected_breaks: List[int], tolerance: int) -> List[int]:
    """
    Calculates localization errors for detected breaks that match a true break
    within the specified tolerance.

    Args:
        true_breaks: List of true break point indices.
        detected_breaks: List of detected break point indices.
        tolerance: The maximum allowed distance for a match.

    Returns:
        A list of absolute differences between matched detected and true breaks.
    """
    localization_errors = []
    matched_true_indices = set() # To ensure each true break is matched only once

    for det_bp in detected_breaks:
        for k, true_bp_k in enumerate(true_breaks):
            if abs(det_bp - true_bp_k) <= tolerance:
                if k not in matched_true_indices:
                    matched_true_indices.add(k)
                    localization_errors.append(abs(det_bp - true_bp_k))
                    break # Move to the next detected break

    return localization_errors


def evaluate_bilstm_by_break_type(dataset: pd.DataFrame, bilstm_results: List[List[int]]) -> Dict:
    """
    Evaluate BiLSTM break detection performance by true break type using 1% tolerance.

    Args:
        dataset: DataFrame containing 'primary_break_type' and 'break_points'.
        bilstm_results: List of lists, where each inner list contains detected break points
                        for the corresponding series in the dataset.

    Returns:
        Dictionary containing performance metrics (P, R, F1, LocErr, counts) by break type.
    """
    if 'primary_break_type' not in dataset.columns or 'break_points' not in dataset.columns:
        print("Dataset must contain 'primary_break_type' and 'break_points' columns for type evaluation.")
        return {}

    break_types = sorted(dataset['primary_break_type'].unique().tolist())
    results_by_type = {}

    # Determine series length from dataset columns (assuming consistent length)
    series_cols = [col for col in dataset.columns if col.startswith('t_')]
    series_length = len(series_cols)
    if series_length == 0:
        print("Dataset does not contain 't_' columns to determine series length.")
        return {}

    # Calculate tolerance based on series length and 1%
    tolerance = max(1, int(series_length * (1.0 / 100.0)))
    print(f"Using tolerance of {tolerance} for evaluation by break type ({1.0}% of length {series_length})")


    for break_type in break_types:
        subset = dataset[dataset['primary_break_type'] == break_type]

        if len(subset) == 0:
            continue

        subset_indices = subset.index.tolist()
        type_true_breaks = [dataset.loc[i, 'break_points'] for i in subset_indices]
        type_detected_breaks = [bilstm_results[i] for i in subset_indices]

        total_tp, total_fp, total_fn = 0, 0, 0
        localization_errors = []

        for true_bp, detected_bp in zip(type_true_breaks, type_detected_breaks):
            # Calculate TP, FP, FN for the current series
            matched_true_indices = set()
            matched_detected_indices = set()
            current_localization_errors = []

            for k, det_bp in enumerate(detected_bp):
                for l, true_bp_val in enumerate(true_bp):
                    if abs(det_bp - true_bp_val) <= tolerance:
                        if l not in matched_true_indices:
                            matched_true_indices.add(l)
                            matched_detected_indices.add(k)
                            current_localization_errors.append(abs(det_bp - true_bp_val))
                            break

            tp = len(matched_detected_indices)
            fp = len(detected_bp) - tp
            fn = len(true_bp) - len(matched_true_indices)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            localization_errors.extend(current_localization_errors)


        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        avg_localization_error = float(np.mean(localization_errors)) if localization_errors else 0.0

        results_by_type[break_type] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_localization_error': avg_localization_error,
            'n_series': int(len(subset)),
            'true_positives': int(total_tp),
            'false_positives': int(total_fp),
            'false_negatives': int(total_fn)
        }

    return results_by_type


def evaluate_bilstm_by_break_count(dataset: pd.DataFrame, bilstm_results: List[List[int]]) -> Dict:
    """
    Evaluate BiLSTM break detection performance by true break count using 1% tolerance.

    Args:
        dataset: DataFrame containing 'n_breaks' and 'break_points'.
        bilstm_results: List of lists, where each inner list contains detected break points
                        for the corresponding series in the dataset.

    Returns:
        Dictionary containing performance metrics (P, R, F1, counts) by break count.
    """
    if 'n_breaks' not in dataset.columns or 'break_points' not in dataset.columns:
        print("Dataset must contain 'n_breaks' and 'break_points' columns for count evaluation.")
        return {}

    break_count_results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0, 'loc_errors': []})

    # Determine series length from dataset columns (assuming consistent length)
    series_cols = [col for col in dataset.columns if col.startswith('t_')]
    series_length = len(series_cols)
    if series_length == 0:
        print("Dataset does not contain 't_' columns to determine series length.")
        return {}

    # Calculate tolerance based on series length and 1%
    tolerance = max(1, int(series_length * (1.0 / 100.0)))
    print(f"Using tolerance of {tolerance} for evaluation by break count ({1.0}% of length {series_length})")


    for idx, row in dataset.iterrows():
        n_true_breaks = int(row['n_breaks'])
        true_bp = list(row['break_points'])
        detected_bp = list(bilstm_results[idx])

        break_count_results[n_true_breaks]['count'] += 1

        if n_true_breaks == 0 and len(detected_bp) == 0:
            # Correctly identified no breaks
            pass
        elif n_true_breaks == 0:
            # False positives where there are no true breaks
            break_count_results[n_true_breaks]['fp'] += len(detected_bp)
        elif len(detected_bp) == 0:
            # False negatives where breaks exist but none detected
            break_count_results[n_true_breaks]['fn'] += n_true_breaks
        else:
            # Breaks in both true and detected - count matches
            matched_true_indices = set()
            matched_detected_indices = set()
            current_localization_errors = []

            for k, det_bp in enumerate(detected_bp):
                for l, true_bp_val in enumerate(true_bp):
                    if abs(det_bp - true_bp_val) <= tolerance:
                        if l not in matched_true_indices:
                            matched_true_indices.add(l)
                            matched_detected_indices.add(k)
                            current_localization_errors.append(abs(det_bp - true_bp_val))
                            break

            tp = len(matched_detected_indices)
            fp = len(detected_bp) - tp
            fn = n_true_breaks - len(matched_true_indices)

            break_count_results[n_true_breaks]['tp'] += tp
            break_count_results[n_true_breaks]['fp'] += fp
            break_count_results[n_true_breaks]['fn'] += fn
            # Accumulate localization errors for this break-count bucket
            break_count_results[n_true_breaks]['loc_errors'].extend(current_localization_errors)

    results = {}
    for n_breaks, stats in sorted(break_count_results.items()):
        total_tp_count = stats['tp']
        total_fp_count = stats['fp']
        total_fn_count = stats['fn']

        precision = total_tp_count / (total_tp_count + total_fp_count) if (total_tp_count + total_fp_count) > 0 else 0.0
        recall = total_tp_count / (total_tp_count + total_fn_count) if (total_tp_count + total_fn_count) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        avg_localization_error = float(np.mean(stats['loc_errors'])) if stats['loc_errors'] else 0.0

        results[n_breaks] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'count': int(stats['count']),
            'true_positives': int(total_tp_count),
            'false_positives': int(total_fp_count),
            'false_negatives': int(total_fn_count),
            'avg_localization_error': avg_localization_error,
            'localization_errors': stats['loc_errors']
        }

    return results

class BreakDetectionDataset(Dataset):
    """PyTorch dataset for break detection training"""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss function combining weighted BCE and Focal Loss
    Optimized for severe class imbalance in break detection
    """

    def __init__(self, pos_weight: float = 8.0, focal_gamma: float = 2.0):
        super(AdaptiveLoss, self).__init__()
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(-1)

        if pred.shape != target.shape:
            pred = pred.view(target.shape)

        # Weighted focal loss for class imbalance
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.focal_gamma
        pos_weight = torch.where(target == 1, self.pos_weight, 1.0)

        loss = focal_weight * bce * pos_weight
        return loss.mean()


class BiLSTMBreakDetector(nn.Module):
    """
    Bidirectional LSTM for structural break detection

    Architecture:
    - Bidirectional LSTM layers for temporal context
    - Conv1d layers for sequence-to-sequence mapping
    - Optimized for 120-timestep windows (best empirical performance)
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 1, device: str = 'cpu'):
        super(BiLSTMBreakDetector, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Conv1d layers as fully connected for sequence processing
        lstm_output_size = hidden_size * 2  # Bidirectional doubles the size
        self.conv1 = nn.Conv1d(lstm_output_size, 512, kernel_size=1)
        self.conv2 = nn.Conv1d(512, output_size, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Initialize hidden states
        h_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h_0, c_0))

        # Reshape for Conv1d: (batch, features, time)
        lstm_out = lstm_out.permute(0, 2, 1)

        # Conv1d layers
        out = self.relu(self.conv1(lstm_out))
        out = self.conv2(out)

        # Back to (batch, time, features)
        out = out.permute(0, 2, 1)

        # Apply sigmoid for binary classification
        out = torch.sigmoid(out)

        return out


class CleanBiLSTMDetector:
    """
    Clean interface for BiLSTM break detection with proper domain matching
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = None
        self.sequence_length = 120  # Optimized based on sequence length experiment
        self.trained = False

    def prepare_training_data(self, dataset: pd.DataFrame, config: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with proper domain matching

        Key insight: Training data MUST match test data distribution exactly:
        - Same series length
        - Same break spacing rules
        - Same preprocessing
        """

        print(f"🎯 DOMAIN-MATCHED TRAINING DATA PREPARATION")
        print(f"=" * 50)
        print(f"Series length: {config['series_length']}")
        print(f"Target samples: {config['target_samples']:,}")

        series_cols = [f't_{i}' for i in range(config['series_length'])]
        focused_ratio = 0.3
        focused_samples = int(config['target_samples'] * focused_ratio)
        natural_samples = config['target_samples'] - focused_samples

        X_all, Y_all = [], []

        # Part 1: Randomized focused sampling (30%)
        print(f"📍 Focused sampling: {focused_samples:,} samples...")
        X_focused, Y_focused = [], []
        samples_created = 0

        for iteration in range(8):
            if samples_created >= focused_samples:
                break

            for _, row in dataset.iterrows():
                if samples_created >= focused_samples:
                    break

                series = np.array([row[col] for col in series_cols])
                break_points = row['break_points'] if row['break_points'] else []

                if break_points:
                    for bp in break_points:
                        positions = [10, 15, 20, 25, 30, 35, 40, 45, 50]
                        for target_position in positions:
                            if samples_created >= focused_samples:
                                break

                            offset = np.random.randint(-2, 3)
                            actual_position = max(5, min(55, target_position + offset))

                            start = max(0, bp - actual_position)
                            end = min(len(series), start + self.sequence_length)
                            start = max(0, end - self.sequence_length)

                            if end - start == self.sequence_length:
                                seq = series[start:end]
                                seq = (seq - np.mean(seq)) / (np.std(seq) + 1e-8)

                                labels = np.zeros(self.sequence_length)
                                break_pos = bp - start
                                if 0 <= break_pos < self.sequence_length:
                                    labels[max(0, break_pos-1):min(self.sequence_length, break_pos+2)] = 1.0

                                X_focused.append(seq.reshape(-1, 1))
                                Y_focused.append(labels)
                                samples_created += 1

                # No-break samples
                if len(break_points) == 0:
                    for start in range(0, len(series)-self.sequence_length, 50):
                        if samples_created >= focused_samples:
                            break

                        seq = series[start:start+self.sequence_length]
                        seq = (seq - np.mean(seq)) / (np.std(seq) + 1e-8)
                        labels = np.zeros(self.sequence_length)

                        X_focused.append(seq.reshape(-1, 1))
                        Y_focused.append(labels)
                        samples_created += 1

        X_focused, Y_focused = np.array(X_focused), np.array(Y_focused)

        # Part 2: Natural sliding windows (70%)
        print(f"🌊 Natural sampling: {natural_samples:,} samples...")
        X_natural, Y_natural = [], []
        samples_created = 0
        stride = 15

        for iteration in range(12):
            if samples_created >= natural_samples:
                break

            for _, row in dataset.iterrows():
                if samples_created >= natural_samples:
                    break

                series = np.array([row[col] for col in series_cols])
                break_points = set(row['break_points'] if row['break_points'] else [])

                for start in range(0, len(series) - self.sequence_length + 1, stride):
                    if samples_created >= natural_samples:
                        break

                    end = start + self.sequence_length
                    seq = series[start:end]
                    seq = (seq - np.mean(seq)) / (np.std(seq) + 1e-8)

                    labels = np.zeros(self.sequence_length)
                    for bp in break_points:
                        if start <= bp < end:
                            rel_pos = bp - start
                            labels[max(0, rel_pos-1):min(self.sequence_length, rel_pos+2)] = 1.0

                    X_natural.append(seq.reshape(-1, 1))
                    Y_natural.append(labels)
                    samples_created += 1

        X_natural, Y_natural = np.concatenate([X_focused, X_natural], axis=0), np.concatenate([Y_focused, Y_natural], axis=0)

        indices = np.random.RandomState(42).permutation(len(X_natural))
        X_natural = X_natural[indices]
        Y_natural = Y_natural[indices]


        # Combine and shuffle
        X_all = X_natural # Use X_natural as it's already combined focused and natural
        Y_all = Y_natural # Use Y_natural as it's already combined focused and natural


        pos_ratio = Y_all.mean()
        break_sequences = (Y_all.sum(axis=1) > 0).sum()

        print(f"✅ Final dataset: {len(X_all):,} samples")
        print(f"   Positive ratio: {pos_ratio:.4f} ({pos_ratio*100:.2f}%)")
        print(f"   Break sequences: {break_sequences:,}/{len(X_all):,} ({break_sequences/len(X_all)*100:.1f}%)")


        return X_all, Y_all


    def train(self, dataset: pd.DataFrame, config: dict) -> Dict:
        """Train the BiLSTM detector with domain matching"""

        print(f"🚀 TRAINING DOMAIN-MATCHED BiLSTM")
        print(f"=" * 40)

        # Prepare data
        X, Y = self.prepare_training_data(dataset, config)

        # Split data
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.15, random_state=config.get('seed', 42)
        )

        print(f"Training: {len(X_train):,} samples")
        print(f"Validation: {len(X_val):,} samples")

        # Create datasets and loaders
        train_dataset = BreakDetectionDataset(X_train, Y_train)
        val_dataset = BreakDetectionDataset(X_val, Y_val)

        train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=96, shuffle=False)

        # Create model
        self.model = BiLSTMBreakDetector(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            device=self.device
        ).to(self.device)

        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.8, patience=3
        )
        criterion = AdaptiveLoss(pos_weight=8.0, focal_gamma=2.0)

        # Training loop
        print(f"\nTraining for up to 25 epochs...")
        best_f1 = 0.0
        patience_counter = 0
        patience = 7

        start_time = time.time()

        for epoch in range(25):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for sequences, labels in train_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(sequences)
                loss = criterion(predictions, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences, labels = sequences.to(self.device), labels.to(self.device)
                    predictions = self.model(sequences)

                    preds_np = predictions.squeeze(-1).cpu().numpy().flatten()
                    labels_np = labels.cpu().numpy().flatten()

                    all_preds.extend(preds_np)
                    all_labels.extend(labels_np)

            # Find best threshold
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels).astype(int)

            epoch_best_f1 = 0.0
            best_threshold = 0.5

            # Evaluate on a range of thresholds
            threshold_range = np.linspace(0.1, 0.9, 9) # Check more thresholds
            for threshold in threshold_range:
                 binary_preds = (all_preds > threshold).astype(int)
                 # Ensure there are both positive labels and positive predictions before calculating metrics
                 if np.sum(all_labels) > 0 and np.sum(binary_preds) > 0:
                    _, _, f1, _ = precision_recall_fscore_support(
                        all_labels, binary_preds, average='binary', zero_division=0
                    )
                    if f1 > epoch_best_f1:
                        epoch_best_f1 = f1
                        best_threshold = threshold
                 elif np.sum(all_labels) == 0 and np.sum(binary_preds) == 0:
                     # Perfect prediction if no breaks and none detected
                     if epoch_best_f1 == 0: # Only update if no breaks were expected/detected
                         epoch_best_f1 = 1.0
                         best_threshold = threshold


            # Early stopping
            if epoch_best_f1 > best_f1:
                best_f1 = epoch_best_f1
                patience_counter = 0
                torch.save(self.model.state_dict(), 'clean_bilstm_model.pth')
            else:
                patience_counter += 1

            scheduler.step(epoch_best_f1)

            avg_train_loss = train_loss / len(train_loader)
            print(f'Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Val_F1={epoch_best_f1:.4f}, Best={best_f1:.4f}')

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        training_time = time.time() - start_time
        self.trained = True

        print(f"\n✅ Training completed!")
        print(f"Best validation F1: {best_f1:.4f}")
        print(f"Training time: {training_time/60:.1f} minutes")

        return {
            'best_f1': best_f1,
            'training_time': training_time,
            'best_threshold': best_threshold
        }

    def detect_breaks(self, time_series: np.ndarray, threshold: float = 0.6, stride: int = 15) -> List[int]:
        """
        Detect structural breaks in a time series using a sliding window.

        Args:
            time_series: Input time series (numpy array).
            threshold: Detection threshold (0.0 to 1.0).
            stride: The step size for the sliding window.

        Returns:
            List of detected break points (integers).
        """

        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before detection")

        self.model.eval()
        potential_breaks = []
        min_distance = 25 # Minimum distance between detected breaks

        with torch.no_grad():
            # Iterate through the time series using a sliding window
            for start in range(0, len(time_series) - self.sequence_length + 1, stride):
                end = start + self.sequence_length
                seq = time_series[start:end]

                # Normalize the window
                seq = (seq - np.mean(seq)) / (np.std(seq) + 1e-8)

                # Predict
                X_tensor = torch.FloatTensor(seq.reshape(1, -1, 1)).to(self.device)
                output = self.model(X_tensor)
                predictions = output.squeeze().cpu().numpy()

                if predictions.ndim == 0:
                    predictions = np.array([predictions]) # Handle single prediction case

                # Identify potential break points within the window
                # A potential break is the index *relative to the start of the full series*
                # where the prediction exceeds the threshold.
                window_potential_breaks = [
                    start + i for i, pred in enumerate(predictions) if pred > threshold
                ]
                potential_breaks.extend(window_potential_breaks)

        # Post-processing: Consolidate potential breaks using a non-maximum suppression-like approach
        # Sort potential breaks by their position
        potential_breaks = sorted(list(set(potential_breaks))) # Remove duplicates first

        detected_breaks = []
        if not potential_breaks:
            return detected_breaks

        # Simple greedy non-maximum suppression
        current_break = potential_breaks[0]
        detected_breaks.append(current_break)

        for bp in potential_breaks[1:]:
            if bp - current_break > min_distance:
                detected_breaks.append(bp)
                current_break = bp

        # Ensure detected breaks are within the original series bounds (0 to len(time_series)-1)
        detected_breaks = [bp for bp in detected_breaks if 0 <= bp < len(time_series)]


        return sorted(detected_breaks)


    def load_model(self, model_path: str):
        """Load a pre-trained model"""
        # Ensure the model architecture matches the saved state_dict
        self.model = BiLSTMBreakDetector(
            input_size=1, hidden_size=64, num_layers=2, output_size=1, device=self.device
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.trained = True
        print(f"✅ Loaded model from {model_path}")

if __name__ == "__main__":
    
    # Generate test data 
    generator = StructuralBreakGenerator(seed=43)
    dataset = generator.generate_dataset(n_series=200, series_length=500, max_breaks=3, min_break_distance=50)
    
    # Train detector
    detector = CleanBiLSTMDetector()
    train_config = {
        'series_length': 500,
        'target_samples': 7000,
        'seed': 43
    }
    
    results = detector.train(dataset, train_config)
    print(f"\nTraining results: {results}")
    
    # Test detection
    test_series = np.array([dataset.iloc[0][f't_{i}'] for i in range(500)])
    breaks = detector.detect_breaks(test_series, threshold=0.6)
    print(f"Detected breaks: {breaks}")
    print(f"True breaks: {dataset.iloc[0]['break_points']}")