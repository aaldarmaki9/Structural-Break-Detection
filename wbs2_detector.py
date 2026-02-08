import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List


# Your R code is already loaded (from your document)
ro.r('''
wbs.sdll.cpt <- function(x, sigma = stats::mad(diff(x)/sqrt(2)), universal = TRUE, M = NULL, th.const = NULL, th.const.min.mult = 0.3, lambda = 0.9, cusums = "systematic") {
  
  n <- length(x)
  if (n <= 1) {
    no.of.cpt <- 0
    cpt <- integer(0)
  }
  else {
    if (sigma == 0) stop("Noise level estimated at zero; therefore no change-points to estimate.")
    if (universal) {
      u <- universal.M.th.v3(n, lambda)
      th.const <- u$th.const
      M <- u$M
    }
    else if (is.null(M) || is.null(th.const)) stop("If universal is FALSE, then M and th.const must be specified.")
    th.const.min <- th.const * th.const.min.mult
    th <- th.const * sqrt(2 * log(n)) * sigma
    th.min <- th.const.min * sqrt(2 * log(n)) * sigma
    
    if (cusums == "random") cusum.sampling <- random.cusums else if (cusums == "systematic") cusum.sampling <- systematic.cusums
    
    rc <- t(wbs.K.int(x, M, cusum.sampling))
    if (max(abs(rc[,4])) < th) {
      no.of.cpt <- 0
      cpt <- integer(0)
      
    }
    else {
      indices <- which(abs(rc[,4]) > th.min)
      if (length(indices) == 1) {
        cpt <- rc[indices, 3]
        no.of.cpt <- 1
      }
      else {
        rc.sel <- rc[indices,,drop=F]
        ord <- order(abs(rc.sel[,4]), decreasing=T)
        z <- abs(rc.sel[ord,4])
        z.l <- length(z)
        dif <- -diff(log(z))
        dif.ord <- order(dif, decreasing=T)
        j <- 1
        while ((j < z.l) & (z[dif.ord[j]+1] > th)) j <- j+1
        if (j < z.l) no.of.cpt <- dif.ord[j] else no.of.cpt <- z.l
        cpt <- sort((rc.sel[ord,3])[1:no.of.cpt])			
      }
    } 
  }
  est <- mean.from.cpt(x, cpt)
  list(est=est, no.of.cpt=no.of.cpt, cpt=cpt)
}


wbs.sdll.cpt.rep <- function(x, sigma = stats::mad(diff(x)/sqrt(2)), universal = TRUE, M = NULL, th.const = NULL, th.const.min.mult = 0.3, lambda = 0.9, repeats = 9) {
  
  res <- vector("list", repeats)
  
  cpt.combined <- integer(0)
  
  nos.of.cpts <- rep(0, repeats)
  
  for (i in 1:repeats) {
    
    res[[i]] <- wbs.sdll.cpt(x, sigma, universal, M, th.const, th.const.min.mult, lambda, "random")
    cpt.combined <- c(cpt.combined, res[[i]]$cpt)
    nos.of.cpts[i] <- res[[i]]$no.of.cpt				
    
  }
  
  med.no.of.cpt <- median(nos.of.cpts)
  
  med.index <- which.min(abs(nos.of.cpts - med.no.of.cpt))
  
  med.run <- res[[med.index]]
  
  list(med.run = med.run, cpt.combined = sort(cpt.combined))
  
}



wbs.K.int <- function(x, M, cusum.sampling) {
  
  n <- length(x)
  if (n == 1) return(matrix(NA, 4, 0))
  else {
    cpt <- t(cusum.sampling(x, M)$max.val)
    return(cbind(cpt, wbs.K.int(x[1:cpt[3]], M, cusum.sampling), wbs.K.int(x[(cpt[3]+1):n], M, cusum.sampling) + c(rep(cpt[3], 3), 0)            ))
  }
  
}



random.cusums <- function(x, M) {
  
  y <- c(0, cumsum(x))
  
  n <- length(x)
  
  M <- min(M, (n-1)*n/2)
  
  res <- matrix(0, M, 4)
  
  if (n==2) ind <- matrix(c(1, 2), 2, 1)
  else if (M == (n-1)*n/2) {
    ind <- matrix(0, 2, M)
    ind[1,] <- rep(1:(n-1), (n-1):1)
    ind[2,] <- 2:(M+1) - rep(cumsum(c(0, (n-2):1)), (n-1):1)
  }
  else {
    ind <- ind2 <- matrix(floor(runif(2*M) * (n-1)), nrow=2)
    ind2[1,] <- apply(ind, 2, min)
    ind2[2,] <- apply(ind, 2, max)
    ind <- ind2 + c(1, 2)
  }
  
  res[,1:2] <- t(ind)
  res[,3:4] <- t(apply(ind, 2, max.cusum, y))
  
  max.ind <- which.max(abs(res[,4]))
  
  max.val <- res[max.ind,,drop=F]
  
  list(res=res, max.val=max.val, M.eff=M)
  
}


systematic.cusums <- function(x, M) {
  
  y <- c(0, cumsum(x))
  
  n <- length(x)
  
  M <- min(M, (n-1)*n/2)
  
  ind <- grid.intervals(n, M)
  
  M <- dim(ind)[2]
  
  res <- matrix(0, M, 4)
  
  res[,1:2] <- t(ind)
  res[,3:4] <- t(apply(ind, 2, max.cusum, y))
  
  max.ind <- which.max(abs(res[,4]))
  
  max.val <- res[max.ind,,drop=F]
  
  list(res=res, max.val=max.val, M.eff=M)
  
}


max.cusum <- function(ind, y) {
  
  z <- y[(ind[1]+1):(ind[2]+1)] - y[ind[1]]
  m <- ind[2]-ind[1]+1
  ip <- sqrt(((m-1):1) / m / (1:(m-1))) * z[1:(m-1)] - sqrt((1:(m-1)) / m / ((m-1):1)) * (z[m] - z[1:(m-1)])
  ip.max <- which.max(abs(ip))
  
  c(ip.max + ind[1] - 1, ip[ip.max])
  
}

mean.from.cpt <- function(x, cpt) {
  
  
  
  n <- length(x)
  
  len.cpt <- length(cpt)
  
  if (len.cpt) cpt <- sort(cpt)
  
  beg <- endd <- rep(0, len.cpt+1)
  
  beg[1] <- 1
  
  endd[len.cpt+1] <- n
  
  if (len.cpt) {
    
    beg[2:(len.cpt+1)] <- cpt+1
    
    endd[1:len.cpt] <- cpt
    
  }
  
  means <- rep(0, len.cpt+1)
  
  for (i in 1:(len.cpt+1)) means[i] <- mean(x[beg[i]:endd[i]])
  
  rep(means, endd-beg+1)
  
}



universal.M.th.v3 <- function(n, lambda = 0.9) {
  
  
  
  mat.90 <- matrix(0, 24, 3)
  
  mat.90[,1] <- c(10, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)
  
  mat.90[,2] <- c(1.420, 1.310, 1.280, 1.270, 1.250, 1.220, 1.205, 1.205, 1.200, 1.200, 1.200, 1.185, 1.185, 1.170, 1.170, 1.160, 1.150, 1.150, 1.150, 1.150, 1.145, 1.145, 1.135, 1.135)
  
  mat.90[,3] <- rep(100, 24)
  
  
  
  mat.95 <- matrix(0, 24, 3)
  
  mat.95[,1] <- c(10, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)
  
  mat.95[,2] <- c(1.550, 1.370, 1.340, 1.320, 1.300, 1.290, 1.265, 1.265, 1.247, 1.247, 1.247, 1.225, 1.225, 1.220, 1.210, 1.190, 1.190, 1.190, 1.190, 1.190, 1.190, 1.180, 1.170, 1.170)
  
  mat.95[,3] <- rep(100, 24)
  
  
  
  if (lambda == 0.9) A <- mat.90 else A <- mat.95
  
  
  
  d <- dim(A)
  
  if (n < A[1,1]) {
    
    th <- A[1,2]
    
    M <- A[1,3]
    
  }
  
  else if (n > A[d[1],1]) {
    
    th <- A[d[1],2]
    
    M <- A[d[1],3]
    
  }
  
  else {
    
    ind <- order(abs(n - A[,1]))[1:2]
    
    s <- min(ind)
    
    e <- max(ind)
    
    th <- A[s,2] * (A[e,1] - n)/(A[e,1] - A[s,1]) + A[e,2] * (n - A[s,1])/(A[e,1] - A[s,1])
    
    M <- A[s,3] * (A[e,1] - n)/(A[e,1] - A[s,1]) + A[e,3] * (n - A[s,1])/(A[e,1] - A[s,1])
    
  }
  
  
  
  list(th.const=th, M=M)
  
}


all.intervals.flat <- function(n) {
  
  if (n == 2) ind <- matrix(1:2, 2, 1) else {
    M <- (n-1)*n/2	
    ind <- matrix(0, 2, M)
    ind[1,] <- rep(1:(n-1), (n-1):1)
    ind[2,] <- 2:(M+1) - rep(cumsum(c(0, (n-2):1)), (n-1):1)
  }
  ind
  
}

grid.intervals <- function(n, M) {
  
  if (n==2) ind <- matrix(c(1, 2), 2, 1)
  
  else if (M >= (n-1)*n/2) ind <- all.intervals.flat(n)
  
  else {
    k <- 1
    while (k*(k-1)/2 < M) k <- k+1
    ind2 <- all.intervals.flat(k)
    ind2.mx <- max(ind2)
    ind <- round((ind2 - 1) * ((n-1) / (ind2.mx-1)) + 1)
  }	
  
  ind	
}
''')

"""
SDLL path plotting utilities
"""

def plot_sdll_path(series, universal: bool = True, lambda_param: float = 0.9, cusums: str = "systematic"):
    """Plot the SDLL selection path for a single time series.

    This visualizes:
      - z: sorted absolute CUSUM magnitudes considered by WBS2
      - dif: SDLL criterion (-diff(log(z))) used to pick number of changes
      - horizontal lines at threshold (th) and th.min
      - vertical line at selected k*

    Args:
        series: 1D array-like of numeric time series values.
        universal: Whether to use universal (n, lambda)-based thresholding.
        lambda_param: Lambda parameter for universal thresholding (0.9 or 0.95 typical).
        cusums: "systematic" or "random" sampling strategy.
    """

    # Define an R helper that exposes the SDLL internals for plotting
    ro.r('''
    wbs.sdll.path <- function(x, sigma = stats::mad(diff(x)/sqrt(2)), universal = TRUE, M = NULL,
                              th.const = NULL, th.const.min.mult = 0.3, lambda = 0.9, cusums = "systematic") {
      n <- length(x)
      if (n <= 1) return(list(z=numeric(0), dif=numeric(0), k=0, th=NA_real_, th.min=NA_real_, cpt.ord=numeric(0), cpt.sel=integer(0)))
      if (sigma == 0) stop("Noise level estimated at zero; therefore no change-points to estimate.")
      if (universal) {
        u <- universal.M.th.v3(n, lambda)
        th.const <- u$th.const
        M <- u$M
      } else if (is.null(M) || is.null(th.const)) stop("If universal is FALSE, then M and th.const must be specified.")
      th.const.min <- th.const * th.const.min.mult
      th <- th.const * sqrt(2 * log(n)) * sigma
      th.min <- th.const.min * sqrt(2 * log(n)) * sigma
      if (cusums == "random") cusum.sampling <- random.cusums else if (cusums == "systematic") cusum.sampling <- systematic.cusums
      rc <- t(wbs.K.int(x, M, cusum.sampling))
      if (is.null(dim(rc)) || ncol(rc) == 0) return(list(z=numeric(0), dif=numeric(0), k=0, th=th, th.min=th.min, cpt.ord=numeric(0), cpt.sel=integer(0)))
      if (max(abs(rc[,4])) < th.min) return(list(z=numeric(0), dif=numeric(0), k=0, th=th, th.min=th.min, cpt.ord=numeric(0), cpt.sel=integer(0)))
      indices <- which(abs(rc[,4]) > th.min)
      rc.sel <- rc[indices,,drop=FALSE]
      ord <- order(abs(rc.sel[,4]), decreasing=TRUE)
      z <- abs(rc.sel[ord,4])
      z.l <- length(z)
      cpt.ord <- rc.sel[ord,3]
      if (z.l == 0) return(list(z=z, dif=numeric(0), k=0, th=th, th.min=th.min, cpt.ord=cpt.ord, cpt.sel=integer(0)))
      dif <- -diff(log(z))
      if (length(dif) == 0) {
        k <- ifelse(z[1] > th, 1, 0)
      } else {
        dif.ord <- order(dif, decreasing=TRUE)
        j <- 1
        while ((j < z.l) & (z[dif.ord[j]+1] > th)) j <- j + 1
        if (j < z.l) k <- dif.ord[j] else k <- z.l
      }
      if (k > 0) {
        cpt.sel <- sort(cpt.ord[1:k])
      } else {
        cpt.sel <- integer(0)
      }
      list(z=z, dif=dif, k=as.integer(k), th=th, th.min=th.min, cpt.ord=cpt.ord, cpt.sel=cpt.sel)
    }
    ''')

    # Call the R helper to obtain the SDLL path objects
    r_series = ro.FloatVector(np.asarray(series, dtype=float))
    r_kwargs = {'universal': universal, 'cusums': cusums}
    r_kwargs['lambda'] = lambda_param  # 'lambda' is reserved in Python; pass via dict
    path = ro.r['wbs.sdll.path'](r_series, **r_kwargs)

    # Extract results
    z = np.array(path.rx2('z'), dtype=float)
    dif = np.array(path.rx2('dif'), dtype=float)
    k = int(np.array(path.rx2('k'))[0]) if len(np.array(path.rx2('k'))) > 0 else 0
    th = float(np.array(path.rx2('th'))[0]) if len(np.array(path.rx2('th'))) > 0 else np.nan
    th_min = float(np.array(path.rx2('th.min'))[0]) if len(np.array(path.rx2('th.min'))) > 0 else np.nan
    cpt_ord = np.array(path.rx2('cpt.ord'), dtype=int) if 'cpt.ord' in path.names else np.array([], dtype=int)
    cpt_sel = np.array(path.rx2('cpt.sel'), dtype=int) if 'cpt.sel' in path.names else np.array([], dtype=int)
    if cpt_ord.size > 0:
        cpt_ord = cpt_ord - 1
    if cpt_sel.size > 0:
        cpt_sel = cpt_sel - 1

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes[0], axes[1], axes[2]

    # Left: z path with thresholds
    if z.size > 0:
        ax1.plot(np.arange(1, len(z) + 1), z, marker='o', lw=1.2, color='tab:blue')
        ax1.axhline(th, color='orange', ls='--', lw=1.0, label='th')
        ax1.axhline(th_min, color='gray', ls=':', lw=1.0, label='th.min')
        if k > 0:
            ax1.axvline(k, color='red', ls='--', lw=1.0, label='k*')
        ax1.set_title('SDLL z path (sorted |CUSUM|)')
        ax1.set_xlabel('Order index')
        ax1.set_ylabel('|CUSUM|')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.set_title('No candidates above th.min')
        ax1.axis('off')

    # Right: dif path
    if dif.size > 0:
        ax2.plot(np.arange(1, len(dif) + 1), dif, marker='o', lw=1.2, color='tab:green')
        if k > 0 and k <= len(dif):
            ax2.axvline(k, color='red', ls='--', lw=1.0, label='k*')
        ax2.axhline(0.0, color='black', lw=0.8)
        ax2.set_title('SDLL criterion: -diff(log(z))')
        ax2.set_xlabel('k')
        ax2.set_ylabel('Δ')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.set_title('Single candidate only; dif is empty')
        ax2.axis('off')

    # Rightmost: series with selected breakpoints
    ax3.plot(np.asarray(series, dtype=float), color='black', lw=1.0)
    if cpt_sel.size > 0:
        for bp in cpt_sel:
            ax3.axvline(bp, color='red', ls='--', lw=1.0)
    ax3.set_title(f'Selected breakpoints (k*={k}) on series')
    ax3.set_xlabel('t')
    ax3.set_ylabel('value')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_detection_examples(dataset: pd.DataFrame, detected_breaks: List[List[int]],
                          n_examples: int = 6):
    """Plot examples of WBS2 detection results"""

    # Dynamically determine series columns based on dataset columns
    series_columns = [col for col in dataset.columns if col.startswith('t_')]
    series_length = len(series_columns)


    # Select examples: 2 with no breaks, 2 with 1-2 breaks, 2 with 2+ breaks
    examples = []

    # No breaks
    no_break_series = dataset[dataset['n_breaks'] == 0]
    if len(no_break_series) >= 2:
        examples.extend(no_break_series.head(2).index.tolist())

    # 1-2 breaks
    few_breaks_series = dataset[dataset['n_breaks'].isin([1, 2])]
    if len(few_breaks_series) >= 2:
        examples.extend(few_breaks_series.head(2).index.tolist())

    # 3+ breaks
    many_breaks_series = dataset[dataset['n_breaks'] >= 3]
    if len(many_breaks_series) >= 2:
        examples.extend(many_breaks_series.head(2).index.tolist())

    examples = examples[:n_examples]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, idx in enumerate(examples):
        if i >= len(axes):
            break

        row = dataset.iloc[idx]
        series = row[series_columns].values
        true_breaks = row['break_points']
        detected = detected_breaks[idx]

        axes[i].plot(series, 'b-', alpha=0.7, label='Time Series')

        # Mark true breaks
        for bp in true_breaks:
            axes[i].axvline(x=bp, color='red', linestyle='-', alpha=0.8,
                          linewidth=2, label='True Break' if bp == true_breaks[0] else "")

        # Mark detected breaks
        for bp in detected:
            axes[i].axvline(x=bp, color='green', linestyle='--', alpha=0.8,
                          linewidth=2, label='Detected' if bp == detected[0] else "")

        axes[i].set_title(f"Series {idx}: {len(true_breaks)} true, {len(detected)} detected")
        axes[i].set_xlabel("Time Point")
        axes[i].set_ylabel("Value")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.tight_layout()
    plt.suptitle('WBS2 Detection Examples', y=1.02, fontsize=14)
    plt.show()

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

class RWBS2Detector:
    """Wrapper for R WBS2 to match Python detector interface"""
    
    def __init__(self, universal=True, lambda_param=0.9, cusums="systematic"):
        self.universal = universal
        self.lambda_param = lambda_param
        self.cusums = cusums
    
    def detect_breaks_fast(self, series):
        """
        Detect breaks using R WBS2 implementation
        Returns list of break indices (Python 0-indexed)
        """
        # Convert to R vector
        r_series = ro.FloatVector(series)
        
        # Build kwargs for R function
        r_kwargs = {
            'universal': self.universal,
            'lambda': self.lambda_param,  # R expects 'lambda'
            'cusums': self.cusums
        }
        
        # Call R function
        result = ro.r['wbs.sdll.cpt'](r_series, **r_kwargs)
        
        # Extract changepoints (convert from R 1-indexed to Python 0-indexed)
        cpt = np.array(result.rx2('cpt'), dtype=int)
        
        if len(cpt) > 0:
            cpt = cpt - 1  # Convert R 1-indexing to Python 0-indexing
        
        return cpt.tolist()

def run_r_wbs2_benchmark():
    """Run comprehensive R WBS2 benchmark on loaded synthetic series"""
    print("=== R WBS2 Performance Benchmark ===")
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
    print("--- Running Benchmark on 500 Length Series (R Implementation) ---")
    
    # Initialize R WBS2 detector
    detector = RWBS2Detector(universal=True, lambda_param=0.9, cusums="systematic")

    # Extract time series data
    series_columns_500 = [f't_{i}' for i in range(500)]
    time_series_data_500 = dataset_500[series_columns_500].values
    true_breaks_500 = dataset_500['break_points'].tolist()
    series_length_500 = len(series_columns_500)

    print("Running R WBS2 detection on 500 length series...")
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

    print("=== R WBS2 Performance Results (500 Series, 1% Tolerance) ===")
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

    # Plot examples for 500 series
    if detected_breaks_500:
        plot_detection_examples(dataset_500.iloc[:len(detected_breaks_500)], detected_breaks_500, n_examples=min(6, len(detected_breaks_500)))

    # --- Benchmark on 1000 length series ---
    print("\n\n--- Running Benchmark on 1000 Length Series (R Implementation) ---")
    
    # Extract time series data
    series_columns_1000 = [f't_{i}' for i in range(1000)]
    time_series_data_1000 = dataset_1000[series_columns_1000].values
    true_breaks_1000 = dataset_1000['break_points'].tolist()
    series_length_1000 = len(series_columns_1000)

    print("Running R WBS2 detection on 1000 length series...")
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

    print("=== R WBS2 Performance Results (1000 Series, 1% Tolerance) ===")
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

    # Plot examples for 1000 series
    if detected_breaks_1000:
        plot_detection_examples(dataset_1000.iloc[:len(detected_breaks_1000)], detected_breaks_1000, n_examples=min(6, len(detected_breaks_1000)))

    return dataset_500, detected_breaks_500, results_500, results_by_type_500, detection_times_500, \
           dataset_1000, detected_breaks_1000, results_1000, results_by_type_1000, detection_times_1000


# Run the benchmark
if __name__ == "__main__":
    results = run_r_wbs2_benchmark()