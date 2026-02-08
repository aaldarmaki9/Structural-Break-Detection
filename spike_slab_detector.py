import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import pickle



# Load the R Spike-and-Slab functions
ro.r('''

solocp_single<-function(y,sigma,q=0.1,tau2=NULL,tau2.spike=NULL,tau2.slab=NULL){

  sigma2 <- sigma^2
  n<-length(y)
  grid <- seq(1,n)
  n.grid <- length(grid)-1
  n1<-rep(1,n.grid+1)

  if (is.null(tau2)){ tau2=2/sqrt(n)}
  if (is.null(tau2.spike)){tau2.spike=1/n}
  if (is.null(tau2.slab)){tau2.slab=n}

  weight<-matrix(1,nrow=n.grid+1,ncol=n.grid+1)

  #Fill the weight matrix
  lower<-tau2/(tau2+sigma2) #this is going down
  lower.local.means<-y[n.grid+1] #this is going down
  parsumLOW <- lower
  for (j1 in (n-1):1){
    new.lower<-tau2*((n-j1+1)-parsumLOW)^2/(tau2*((n-j1+1)-(parsumLOW))+sigma2)
    lower<-c(new.lower,lower)
    parsumLOW <- parsumLOW + new.lower
    # if (abs(new.lower-1)<.000001){
    #   lower <- c(rep(1,n-length(lower)),lower)
    #   break}
  }
  revlower <- revcumsum(lower)
  revy <- revcumsum(y)
  lower.local.means<-y[n.grid+1] #this is going down
  parsum <- lower.local.means*lower[n]
  for (j1 in (n-1):1){
    new.lower.mean <- (revy[j1]-parsum)/((n-j1+1)-revlower[j1+1])
    lower.local.means<-c(new.lower.mean,lower.local.means)
    parsum <- parsum + lower[j1]*new.lower.mean
  }

  sum.lower<-revcumsum(lower)
  mean.disc<-lower.local.means*lower
  sum.mean.disc<-revcumsum(mean.disc)
  marg.mean <- c(sum.mean.disc[2:length(sum.mean.disc)],0)


  marg.weight <- c(sum.lower[2:length(sum.lower)],0)
  #Define M and GAMMA and Y matrices
  sum.inv<-seq(n,1)

  GAM<-matrix(NA,nrow=n,ncol=n)
  M<-matrix(NA,nrow=n,ncol=n)
  Y<-matrix(NA,nrow=n,ncol=n)

  sum.y<-rev(cumsum(rev(y)))

  #param
  GAM[,1] <- 1
  Y[,1] <- sum.y[1]-marg.mean
  M[,1]<-1/((sum.inv[1] - marg.weight)*GAM[,1]+sigma2*tau2^(-1))
  GAM[2:n,2]<-1-(sum.inv[2] - marg.weight[2:n])*M[2:n,1]
  Y[2:n,2]<-sum.y[2]-marg.mean[2:n]-(sum.inv[2] - marg.weight[2:n])*M[2:n,1]*GAM[2:n,1]*Y[2:n,1]
  #weight
  w.spike<-sqrt(tau2.spike^(-1)/(lower[1]+tau2.spike^(-1)))*exp(1/(2*sigma2)*(sum(y)-sum(lower.local.means[2:(n.grid+1)]*lower[2:(n.grid+1)]))^2/(lower[1]+tau2.spike^(-1)))#maybe wrong but it does not matter
  w.slab<-sqrt(tau2.slab^(-1)/(lower[1]+tau2.slab^(-1)))*exp(1/(2*sigma2)*(sum(y)-sum(lower.local.means[2:(n.grid+1)]*lower[2:(n.grid+1)]))^2/(lower[1]+tau2.slab^(-1)))#maybe wrong but it does not matter
  parsumMGAM2 <- M[1:n,1]*GAM[1:n,1]^2
  parsumMGAMY <- M[1:n,1]*GAM[1:n,1]*Y[1:n,1]
  for (i in 3:(n-1)){
    #param
    M[(i-1):n,(i-1)]<-1/((sum.inv[(i-1)] - marg.weight[(i-1):n])*GAM[(i-1):n,(i-1)]+sigma2*tau2^(-1))
    parsumMGAM2 <- parsumMGAM2 +  M[1:n,(i-1)]*GAM[1:n,(i-1)]^2
    GAM[i:n,i]<-1-(sum.inv[i] - marg.weight[i:n])*parsumMGAM2[i:n]
    parsumMGAMY <- parsumMGAMY + M[1:n,(i-1)]*GAM[1:n,(i-1)]*Y[1:n,(i-1)]
    Y[i:n,i]<-sum.y[i]-marg.mean[i:n]-(sum.inv[i] - marg.weight[i:n])*parsumMGAMY[i:n]
    #weight
  }

  #param
  M[(n-1):n,(n-1)]<-1/((sum.inv[(n-1)] - marg.weight[(n-1):n])*GAM[(n-1):n,(n-1)]+sigma2*tau2^(-1))
  GAM[n,n]<-1-(sum.inv[n] - marg.weight[n])*sum(M[n,1:(n-1)]*GAM[n,1:(n-1)]^2)
  Y[n,n]<-sum.y[i]-marg.mean[n]-(sum.inv[n] - marg.weight[n])*sum(M[n,1:(n-1)]*GAM[n,1:(n-1)]*Y[n,1:(n-1)])

  idGAM <- matrix(c(seq(2,n),seq(1,n-1)),ncol=2)
  w.spike<-c(w.spike,sqrt(tau2.spike^(-1)/((sum.inv[2:n] - marg.weight[2:n])*GAM[idGAM]+tau2.spike^(-1)))*exp(1/(2*sigma2)*diag(Y)[-1]^2/((sum.inv[2:n] - marg.weight[2:n])*GAM[idGAM]+tau2.spike^(-1))))
  w.slab<-c(w.slab,sqrt(tau2.slab^(-1)/((sum.inv[2:n] - marg.weight[2:n])*GAM[idGAM]+tau2.slab^(-1)))*exp(1/(2*sigma2)*diag(Y)[-1]^2/((sum.inv[2:n] - marg.weight[2:n])*GAM[idGAM]+tau2.slab^(-1))))


  ratio<-q*w.slab/(q*w.slab+(1-q)*w.spike)
  id.inf <- which(w.slab==Inf)
  ratio[id.inf] <- 1


  return(list(w.slab=w.slab,
              w.spike=w.spike,
              ratio=ratio))


}




revcumsum <- function(x){
  return(rev(cumsum(rev(x))))
}
subset_changepoints <- function(ratio,del=5){

  first <- which(ratio >=.5)
  if (length(first)>0){
    n.c <- length(first)

    low <- first - del
    #id.low <- which(low[2:(n.c)]<= first[1:(n.c-1)])
    #id.up <- which(up[1:(n.c-1)]>= first[2:(n.c)]) you do not need both sides: it is symmetric!
    id.split <-which((low[2:(n.c)]<= first[1:(n.c-1)])==FALSE)

    i.low.bl <-first[c(1,id.split+1)]
    i.up.bl <- first[c(id.split,n.c)]


    #sweep through the block to pick a change point.
    change.points <- c()
    for (i in 1: length(i.low.bl)){
      f <- which(ratio[i.low.bl[i]:i.up.bl[i]]==max(ratio[i.low.bl[i]:i.up.bl[i]]))
      change.points <- c(change.points, i.low.bl[i]+ f[ceiling(length(f)/2)] -1)
    }
    #remove change.points right at the beginning and at the end
    change.points<-change.points[change.points>5 & change.points<(length(ratio)-5)]
  } else {
    change.points<-NULL
    print("No change points detected")
  }
  return(change.points)
}
''')

class SpikeAndSlabDetector:
    """Wrapper for R Spike-and-Slab changepoint detection"""
    
    def __init__(self, q=0.1, tau2=None, tau2_spike=None, tau2_slab=None, 
                 sigma=None, del_threshold=5):
        self.q = q
        self.tau2 = tau2
        self.tau2_spike = tau2_spike
        self.tau2_slab = tau2_slab
        self.sigma = sigma
        self.del_threshold = del_threshold
    
    def detect_breaks_fast(self, series):
        """
        Detect breaks using Spike-and-Slab method
        Returns list of break indices (Python 0-indexed)
        """
        with localconverter(ro.default_converter + numpy2ri.converter):
            # Convert to R vector
            r_series = ro.FloatVector(series)
            
            # Estimate sigma if not provided
            if self.sigma is None:
                sigma = np.median(np.abs(np.diff(series))) / 0.6745
            else:
                sigma = self.sigma
            
            # Build kwargs for R function (convert NumPy scalars to Python types)
            r_kwargs = {
                'y': r_series,
                'sigma': float(sigma),
                'q': float(self.q)
            }
            
            if self.tau2 is not None:
                r_kwargs['tau2'] = float(self.tau2)
            if self.tau2_spike is not None:
                r_kwargs['tau2.spike'] = float(self.tau2_spike)
            if self.tau2_slab is not None:
                r_kwargs['tau2.slab'] = float(self.tau2_slab)
            
            # Get marginal inclusion probabilities
            result = ro.r['solocp_single'](**r_kwargs)
            # Access named elements by finding the index
            names_list = list(result.names())
            ratio_idx = names_list.index('ratio')
            ratio = np.array(result[ratio_idx])
            
            # Extract changepoints using subset_changepoints
            cpt_result = ro.r['subset_changepoints'](
                ro.FloatVector(ratio),
                **{'del': self.del_threshold}
            )
            
            # Check if result is NULL (no changepoints detected)
            # Use isinstance to check for NULL properly
            if isinstance(cpt_result, type(ro.NULL)) or len(cpt_result) == 0:
                return []
            
            # Convert to numpy array
            cpt = np.array(cpt_result, dtype=int)
            if len(cpt) > 0:
                cpt = cpt - 1  # Convert R 1-indexing to Python 0-indexing
            
            return cpt.tolist()
    
    def get_inclusion_probabilities(self, series):
        """
        Get the full marginal inclusion probability vector
        Useful for visualization
        """
        with localconverter(ro.default_converter + numpy2ri.converter):
            r_series = ro.FloatVector(series)
            
            if self.sigma is None:
                sigma = np.median(np.abs(np.diff(series))) / 0.6745
            else:
                sigma = self.sigma
            
            r_kwargs = {
                'y': r_series,
                'sigma': float(sigma),
                'q': float(self.q)
            }
            
            if self.tau2 is not None:
                r_kwargs['tau2'] = float(self.tau2)
            if self.tau2_spike is not None:
                r_kwargs['tau2.spike'] = float(self.tau2_spike)
            if self.tau2_slab is not None:
                r_kwargs['tau2.slab'] = float(self.tau2_slab)
            
            result = ro.r['solocp_single'](**r_kwargs)
            
            # Access named elements by finding their indices
            names_list = list(result.names())
            ratio_idx = names_list.index('ratio')
            w_spike_idx = names_list.index('w.spike')
            w_slab_idx = names_list.index('w.slab')
            
            return {
                'ratio': np.array(result[ratio_idx]),
                'w_spike': np.array(result[w_spike_idx]),
                'w_slab': np.array(result[w_slab_idx])
            }

def evaluate_detection_performance(true_breaks_list, detected_breaks_list, 
                                  series_length, tolerance_percentage=1.0):
    """
    Evaluate changepoint detection performance with correct recall calculation
    """
    tolerance = max(1, int(series_length * (tolerance_percentage / 100.0)))
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    localization_errors = []
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for i in range(len(detected_breaks_list)):
        true_bp = true_breaks_list[i]
        detected_bp = detected_breaks_list[i]
        
        # Ensure they're lists/arrays
        if not isinstance(true_bp, (list, np.ndarray)):
            true_bp = []
        if not isinstance(detected_bp, (list, np.ndarray)):
            detected_bp = []
            
        tp = 0
        matched_true_indices = set()
        current_loc_errors = []
        
        # Match detected to true breaks
        for det_bp in detected_bp:
            for k, true_bp_k in enumerate(true_bp):
                if k not in matched_true_indices:
                    if abs(det_bp - true_bp_k) <= tolerance:
                        matched_true_indices.add(k)
                        tp += 1
                        current_loc_errors.append(abs(det_bp - true_bp_k))
                        break
        
        fp = len(detected_bp) - tp
        fn = len(true_bp) - tp
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        localization_errors.extend(current_loc_errors)
        
        # Per-series metrics - ONLY calculate if there are actual breaks or detections
        if len(detected_bp) > 0:  # Only if we detected something
            precision = tp / len(detected_bp)
            precisions.append(precision)
        
        if len(true_bp) > 0:  # Only if there are true breaks to find
            recall = tp / len(true_bp)
            recalls.append(recall)
            
            # F1 only makes sense when there are true breaks
            if len(detected_bp) > 0:
                prec = tp / len(detected_bp)
                rec = tp / len(true_bp)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                f1_scores.append(f1)
    
    # Overall metrics (micro-average)
    overall_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    overall_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    # Macro-averages (average of per-series metrics)
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    
    return {
        'tolerance_used': tolerance,
        'precision': overall_precision,  # Micro-average
        'recall': overall_recall,  # Micro-average
        'f1_score': overall_f1,  # Micro-average
        'avg_precision': avg_precision,  # Macro-average
        'avg_recall': avg_recall,  # Macro-average
        'avg_f1': avg_f1,  # Macro-average
        'avg_localization_error': np.mean(localization_errors) if localization_errors else 0.0,
        'true_positives': tp_total,
        'false_positives': fp_total,
        'false_negatives': fn_total,
        'num_series_with_detections': len(precisions),
        'num_series_with_true_breaks': len(recalls)
    }

def plot_spike_slab_examples(dataset, detected_breaks, inclusion_probs, n_examples=6):
    """
    Plot examples with both time series and posterior inclusion probabilities
    
    Parameters:
    -----------
    dataset : DataFrame
        Dataset containing time series and metadata
    detected_breaks : list
        List of detected break point lists for each series
    inclusion_probs : list
        List of inclusion probability arrays for each series
    n_examples : int
        Number of examples to plot
    """
    import matplotlib.pyplot as plt
    
    # Select diverse examples (different break counts)
    break_counts = dataset['n_breaks'].values[:len(detected_breaks)]
    unique_counts = sorted(set(break_counts))
    
    examples_to_plot = []
    for count in unique_counts:
        indices = np.where(break_counts == count)[0]
        if len(indices) > 0:
            examples_to_plot.append(indices[0])
        if len(examples_to_plot) >= n_examples:
            break
    
    # Pad with additional examples if needed
    if len(examples_to_plot) < n_examples:
        remaining = [i for i in range(len(detected_breaks)) if i not in examples_to_plot]
        examples_to_plot.extend(remaining[:n_examples - len(examples_to_plot)])
    
    examples_to_plot = examples_to_plot[:n_examples]
    
    # Create plots
    n_cols = 2
    n_rows = len(examples_to_plot)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    series_length = len([col for col in dataset.columns if col.startswith('t_')])
    series_columns = [f't_{i}' for i in range(series_length)]
    
    for idx, series_idx in enumerate(examples_to_plot):
        row = dataset.iloc[series_idx]
        series = row[series_columns].values
        true_breaks = row['break_points']
        detected = detected_breaks[series_idx]
        probs = inclusion_probs[series_idx]
        
        # Left plot: Time series with changepoints
        ax1 = axes[idx, 0]
        ax1.plot(series, 'k-', alpha=0.7, linewidth=1)
        
        # Plot true breaks
        for bp in true_breaks:
            ax1.axvline(bp, color='green', linestyle='--', linewidth=2, alpha=0.7, label='True' if bp == true_breaks[0] else '')
        
        # Plot detected breaks
        for bp in detected:
            ax1.axvline(bp, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Detected' if bp == detected[0] else '')
        
        ax1.set_title(f"Series {series_idx}: {row['primary_break_type'].replace('_', ' ').title()}\n"
                     f"True breaks: {len(true_breaks)}, Detected: {len(detected)}")
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        if idx == 0:
            ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Posterior inclusion probabilities
        ax2 = axes[idx, 1]
        ax2.plot(probs, 'b-', linewidth=1.5)
        ax2.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (0.5)')
        
        # Mark true breaks
        for bp in true_breaks:
            ax2.axvline(bp, color='green', linestyle='--', linewidth=2, alpha=0.4)
        
        # Mark detected breaks
        for bp in detected:
            ax2.axvline(bp, color='red', linestyle=':', linewidth=2, alpha=0.7)
        
        ax2.set_title(f"Marginal Inclusion Probabilities")
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Probability')
        ax2.set_ylim([-0.05, 1.05])
        if idx == 0:
            ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_spike_slab_benchmark():
    """Run comprehensive Spike-and-Slab benchmark on loaded synthetic series"""
    print("=== Spike-and-Slab Performance Benchmark ===")
    print()

    # Load synthetic data
    dataset_500 = pd.read_pickle('synthetic_breaks_100_500_min50.pkl')
    dataset_1000 = pd.read_pickle('synthetic_breaks_100_1000_min100.pkl')

    print(f"Dataset (500 length) loaded:")
    print(f"- Total series: {len(dataset_500):,}")
    print(f"- Break distribution: {dataset_500['n_breaks'].value_counts().sort_index().to_dict()}")
    print()

    print(f"Dataset (1000 length) loaded:")
    print(f"- Total series: {len(dataset_1000):,}")
    print(f"- Break distribution: {dataset_1000['n_breaks'].value_counts().sort_index().to_dict()}")
    print()

    # --- Benchmark on 500 length series ---
    print("--- Running Benchmark on 500 Length Series (Spike-and-Slab) ---")
    
    # Initialize Spike-and-Slab detector
    detector = SpikeAndSlabDetector(q=0.1, del_threshold=5)

    # Extract time series data
    series_columns_500 = [f't_{i}' for i in range(500)]
    time_series_data_500 = dataset_500[series_columns_500].values
    true_breaks_500 = dataset_500['break_points'].tolist()
    series_length_500 = len(series_columns_500)

    print("Running Spike-and-Slab detection on 500 length series...")
    start_time_500 = time.time()

    detected_breaks_500 = []
    detection_times_500 = []

    try:
        for i, series in enumerate(time_series_data_500):
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1:,} series...")

            series_start_time = time.time()
            breaks = detector.detect_breaks_fast(series)
            series_end_time = time.time()

            detected_breaks_500.append(breaks)
            detection_times_500.append(series_end_time - series_start_time)

    except KeyboardInterrupt:
        print("\nBenchmark on 500 series interrupted by user.")
        pass

    total_time_500 = time.time() - start_time_500

    print(f"Detection completed for 500 series in {total_time_500:.2f} seconds")
    if detection_times_500:
        print(f"Average time per 500 series (processed): {np.mean(detection_times_500)*1000:.2f} ms")
    print()

    # Evaluate performance for 500 series with 1% tolerance
    print("Evaluating performance for 500 series with 1% tolerance...")
    results_500 = evaluate_detection_performance(
        true_breaks_500[:len(detected_breaks_500)], 
        detected_breaks_500, 
        series_length_500, 
        tolerance_percentage=1.0
    )

    print("=== Spike-and-Slab Performance Results (500 Series, 1% Tolerance) ===")
    print(f"Tolerance used: {results_500['tolerance_used']}")
    print(f"Overall Precision: {results_500['precision']:.3f}")
    print(f"Overall Recall: {results_500['recall']:.3f}")
    print(f"Overall F1-Score: {results_500['f1_score']:.3f}")
    print()
    print(f"Average Precision: {results_500['avg_precision']:.3f}")
    print(f"Average Recall: {results_500['avg_recall']:.3f}")
    print(f"Average F1-Score: {results_500['avg_f1']:.3f}")
    print()
    print(f"Average Localization Error: {results_500['avg_localization_error']:.1f} time points")
    print()
    print(f"Confusion Matrix:")
    print(f"True Positives: {results_500['true_positives']}")
    print(f"False Positives: {results_500['false_positives']}")
    print(f"False Negatives: {results_500['false_negatives']}")

    # Performance by number of breaks for 500 series
    print("\n=== Performance by Break Count (500 Series, 1% Tolerance) ===")
    break_count_results_500 = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0, 'localization_errors': []})

    for i in range(len(detected_breaks_500)):
        true_bp = true_breaks_500[i]
        detected_bp = detected_breaks_500[i]
        n_true_breaks = len(true_bp)

        break_count_results_500[n_true_breaks]['count'] += 1

        tp = 0
        matched_true_indices = set()
        matched_detected_indices = set()
        current_localization_errors = []

        current_tolerance = max(1, int(series_length_500 * (1.0 / 100.0)))

        for j, det_bp in enumerate(detected_bp):
            for k, true_bp_k in enumerate(true_bp):
                if abs(det_bp - true_bp_k) <= current_tolerance:
                    if k not in matched_true_indices:
                        matched_true_indices.add(k)
                        matched_detected_indices.add(j)
                        tp += 1
                        current_localization_errors.append(abs(det_bp - true_bp_k))
                        break

        fp = len(detected_bp) - tp
        fn = n_true_breaks - tp

        break_count_results_500[n_true_breaks]['tp'] += tp
        break_count_results_500[n_true_breaks]['fp'] += fp
        break_count_results_500[n_true_breaks]['fn'] += fn
        break_count_results_500[n_true_breaks]['localization_errors'].extend(current_localization_errors)

    for n_breaks in sorted(break_count_results_500.keys()):
        stats = break_count_results_500[n_breaks]
        precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_loc_error = np.mean(stats['localization_errors']) if stats['localization_errors'] else 0

        print(f"{n_breaks} breaks ({stats['count']} series): P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, LocErr={avg_loc_error:.1f}")

    # Performance by break type for 500 series
    print("\n=== Performance by Break Type (500 Series, 1% Tolerance) ===")
    results_by_type_500 = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0, 'localization_errors': []})

    for i in range(len(detected_breaks_500)):
        true_bp = true_breaks_500[i]
        detected_bp = detected_breaks_500[i]
        primary_break_type = dataset_500.iloc[i]['primary_break_type']

        results_by_type_500[primary_break_type]['count'] += 1

        tp = 0
        matched_true_indices = set()
        current_localization_errors = []

        current_tolerance = max(1, int(series_length_500 * (1.0 / 100.0)))

        for j, det_bp in enumerate(detected_bp):
            for k, true_bp_k in enumerate(true_bp):
                if abs(det_bp - true_bp_k) <= current_tolerance:
                    if k not in matched_true_indices:
                        matched_true_indices.add(k)
                        tp += 1
                        current_localization_errors.append(abs(det_bp - true_bp_k))
                        break

        fp = len(detected_bp) - tp
        fn = len(true_bp) - tp

        results_by_type_500[primary_break_type]['tp'] += tp
        results_by_type_500[primary_break_type]['fp'] += fp
        results_by_type_500[primary_break_type]['fn'] += fn
        results_by_type_500[primary_break_type]['localization_errors'].extend(current_localization_errors)

    for break_type in sorted(results_by_type_500.keys()):
        stats = results_by_type_500[break_type]
        total_detected = stats['tp'] + stats['fp']
        total_true = stats['tp'] + stats['fn']

        precision = stats['tp'] / total_detected if total_detected > 0 else 0
        recall = stats['tp'] / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_loc_error = np.mean(stats['localization_errors']) if stats['localization_errors'] else 0
        print(f"{break_type.replace('_', ' ').title()} ({stats['count']} series): P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, LocErr={avg_loc_error:.1f}")
    # Before plotting, get inclusion probabilities for all series
    print("\nComputing inclusion probabilities for visualization...")
    inclusion_probs_500 = []
    for series in time_series_data_500[:len(detected_breaks_500)]:
        probs = detector.get_inclusion_probabilities(series)
        inclusion_probs_500.append(probs['ratio'])
    
    # Plot examples for 500 series with probabilities
    if detected_breaks_500:
        plot_spike_slab_examples(
            dataset_500.iloc[:len(detected_breaks_500)], 
            detected_breaks_500, 
            inclusion_probs_500,
            n_examples=min(6, len(detected_breaks_500))
        )

    # --- Benchmark on 1000 length series ---
    print("\n\n--- Running Benchmark on 1000 Length Series (Spike-and-Slab) ---")
    
    # Extract time series data
    series_columns_1000 = [f't_{i}' for i in range(1000)]
    time_series_data_1000 = dataset_1000[series_columns_1000].values
    true_breaks_1000 = dataset_1000['break_points'].tolist()
    series_length_1000 = len(series_columns_1000)

    print("Running Spike-and-Slab detection on 1000 length series...")
    start_time_1000 = time.time()

    detected_breaks_1000 = []
    detection_times_1000 = []

    try:
        for i, series in enumerate(time_series_data_1000):
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1:,} series...")

            series_start_time = time.time()
            breaks = detector.detect_breaks_fast(series)
            series_end_time = time.time()

            detected_breaks_1000.append(breaks)
            detection_times_1000.append(series_end_time - series_start_time)

    except KeyboardInterrupt:
        print("\nBenchmark on 1000 series interrupted by user.")
        pass

    total_time_1000 = time.time() - start_time_1000

    print(f"Detection completed for 1000 series in {total_time_1000:.2f} seconds")
    if detection_times_1000:
        print(f"Average time per 1000 series (processed): {np.mean(detection_times_1000)*1000:.2f} ms")
    print()

    # Evaluate performance for 1000 series with 1% tolerance
    print("Evaluating performance for 1000 series with 1% tolerance...")
    results_1000 = evaluate_detection_performance(
        true_breaks_1000[:len(detected_breaks_1000)], 
        detected_breaks_1000, 
        series_length_1000, 
        tolerance_percentage=1.0
    )

    print("=== Spike-and-Slab Performance Results (1000 Series, 1% Tolerance) ===")
    print(f"Tolerance used: {results_1000['tolerance_used']}")
    print(f"Overall Precision: {results_1000['precision']:.3f}")
    print(f"Overall Recall: {results_1000['recall']:.3f}")
    print(f"Overall F1-Score: {results_1000['f1_score']:.3f}")
    print()
    print(f"Average Precision: {results_1000['avg_precision']:.3f}")
    print(f"Average Recall: {results_1000['avg_recall']:.3f}")
    print(f"Average F1-Score: {results_1000['avg_f1']:.3f}")
    print()
    print(f"Average Localization Error: {results_1000['avg_localization_error']:.1f} time points")
    print()
    print(f"Confusion Matrix:")
    print(f"True Positives: {results_1000['true_positives']}")
    print(f"False Positives: {results_1000['false_positives']}")
    print(f"False Negatives: {results_1000['false_negatives']}")

    # Performance by number of breaks for 1000 series
    print("\n=== Performance by Break Count (1000 Series, 1% Tolerance) ===")
    break_count_results_1000 = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0, 'localization_errors': []})

    for i in range(len(detected_breaks_1000)):
        true_bp = true_breaks_1000[i]
        detected_bp = detected_breaks_1000[i]
        n_true_breaks = len(true_bp)

        break_count_results_1000[n_true_breaks]['count'] += 1

        tp = 0
        matched_true_indices = set()
        matched_detected_indices = set()
        current_localization_errors = []

        current_tolerance = max(1, int(series_length_1000 * (1.0 / 100.0)))

        for j, det_bp in enumerate(detected_bp):
            for k, true_bp_k in enumerate(true_bp):
                if abs(det_bp - true_bp_k) <= current_tolerance:
                    if k not in matched_true_indices:
                        matched_true_indices.add(k)
                        matched_detected_indices.add(j)
                        tp += 1
                        current_localization_errors.append(abs(det_bp - true_bp_k))
                        break

        fp = len(detected_bp) - tp
        fn = n_true_breaks - tp

        break_count_results_1000[n_true_breaks]['tp'] += tp
        break_count_results_1000[n_true_breaks]['fp'] += fp
        break_count_results_1000[n_true_breaks]['fn'] += fn
        break_count_results_1000[n_true_breaks]['localization_errors'].extend(current_localization_errors)

    for n_breaks in sorted(break_count_results_1000.keys()):
        stats = break_count_results_1000[n_breaks]
        precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_loc_error = np.mean(stats['localization_errors']) if stats['localization_errors'] else 0

        print(f"{n_breaks} breaks ({stats['count']} series): P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, LocErr={avg_loc_error:.1f}")

    # Performance by break type for 1000 series
    print("\n=== Performance by Break Type (1000 Series, 1% Tolerance) ===")
    results_by_type_1000 = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0, 'localization_errors': []})

    for i in range(len(detected_breaks_1000)):
        true_bp = true_breaks_1000[i]
        detected_bp = detected_breaks_1000[i]
        primary_break_type = dataset_1000.iloc[i]['primary_break_type']

        results_by_type_1000[primary_break_type]['count'] += 1

        tp = 0
        matched_true_indices = set()
        current_localization_errors = []

        current_tolerance = max(1, int(series_length_1000 * (1.0 / 100.0)))

        for j, det_bp in enumerate(detected_bp):
            for k, true_bp_k in enumerate(true_bp):
                if abs(det_bp - true_bp_k) <= current_tolerance:
                    if k not in matched_true_indices:
                        matched_true_indices.add(k)
                        tp += 1
                        current_localization_errors.append(abs(det_bp - true_bp_k))
                        break

        fp = len(detected_bp) - tp
        fn = len(true_bp) - tp

        results_by_type_1000[primary_break_type]['tp'] += tp
        results_by_type_1000[primary_break_type]['fp'] += fp
        results_by_type_1000[primary_break_type]['fn'] += fn
        results_by_type_1000[primary_break_type]['localization_errors'].extend(current_localization_errors)

    for break_type in sorted(results_by_type_1000.keys()):
        stats = results_by_type_1000[break_type]
        total_detected = stats['tp'] + stats['fp']
        total_true = stats['tp'] + stats['fn']

        precision = stats['tp'] / total_detected if total_detected > 0 else 0
        recall = stats['tp'] / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_loc_error = np.mean(stats['localization_errors']) if stats['localization_errors'] else 0
        print(f"{break_type.replace('_', ' ').title()} ({stats['count']} series): P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, LocErr={avg_loc_error:.1f}")
    # Before plotting 1000 series
    print("\nComputing inclusion probabilities for 1000 series visualization...")
    inclusion_probs_1000 = []
    for series in time_series_data_1000[:len(detected_breaks_1000)]:
        probs = detector.get_inclusion_probabilities(series)
        inclusion_probs_1000.append(probs['ratio'])
    
    # Plot examples for 1000 series
    if detected_breaks_1000:
        plot_spike_slab_examples(
            dataset_1000.iloc[:len(detected_breaks_1000)], 
            detected_breaks_1000, 
            inclusion_probs_1000,
            n_examples=min(6, len(detected_breaks_1000))
        )
    
    return dataset_500, detected_breaks_500, results_500, results_by_type_500, detection_times_500, \
           dataset_1000, detected_breaks_1000, results_1000, results_by_type_1000, detection_times_1000


# Run the benchmark only when script is executed directly
if __name__ == "__main__":
    ss_results = run_spike_slab_benchmark()
