import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Any


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

def plot_sdll_path(
    series,
    universal: bool = True,
    lambda_param: float = 0.9,
    cusums: str = "systematic",
    th_const_min_mult: float = 0.3,
):
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
        th_const_min_mult: Multiplier for minimum threshold (th.min = th * th_const_min_mult).
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
    r_kwargs = {
        'universal': universal,
        'cusums': cusums,
        'th.const.min.mult': th_const_min_mult,
    }
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
    
    def __init__(self, universal=True, lambda_param=0.9, cusums="systematic", th_const_min_mult=0.3):
        self.universal = universal
        self.lambda_param = lambda_param
        self.cusums = cusums
        self.th_const_min_mult = th_const_min_mult
    
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
            'th.const.min.mult': self.th_const_min_mult,
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

def _group_performance(true_breaks_list, detected_breaks_list, group_labels, series_length, tolerance_percentage=1.0):
    """Aggregate TP/FP/FN and localization errors for each group label."""
    tolerance = max(1, int(series_length * (tolerance_percentage / 100.0)))
    grouped = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0, 'localization_errors': []})

    for true_bp, detected_bp, label in zip(true_breaks_list, detected_breaks_list, group_labels):
        grouped[label]['count'] += 1
        tp = 0
        matched_true = set()
        current_loc_errors = []

        for det_bp in detected_bp:
            for k, true_bp_k in enumerate(true_bp):
                if abs(det_bp - true_bp_k) <= tolerance and k not in matched_true:
                    matched_true.add(k)
                    tp += 1
                    current_loc_errors.append(abs(det_bp - true_bp_k))
                    break

        fp = len(detected_bp) - tp
        fn = len(true_bp) - tp
        grouped[label]['tp'] += tp
        grouped[label]['fp'] += fp
        grouped[label]['fn'] += fn
        grouped[label]['localization_errors'].extend(current_loc_errors)

    return grouped


def _print_overall_metrics(results: Dict[str, Any], label: str):
    print(f"=== R WBS2 Performance Results ({label}, 1% Tolerance) ===")
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


def _print_group_metrics(grouped_stats, label: str, formatter=lambda v: str(v)):
    print(f"\n=== Performance by {label} (1% Tolerance) ===")
    for group_key in sorted(grouped_stats.keys()):
        stats = grouped_stats[group_key]
        total_detected = stats['tp'] + stats['fp']
        total_true = stats['tp'] + stats['fn']
        precision = stats['tp'] / total_detected if total_detected else 0
        recall = stats['tp'] / total_true if total_true else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        avg_loc_error = np.mean(stats['localization_errors']) if stats['localization_errors'] else 0
        print(
            f"{formatter(group_key)} ({stats['count']} series): "
            f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, LocErr={avg_loc_error:.1f}"
        )


def run_r_wbs2_benchmark_for(
    path: str,
    series_length: int,
    label: str,
    th_const_min_mult: float = 0.3,
    lambda_param: float = 0.9,
    cusums: str = "systematic",
    tolerance_percentage: float = 1.0,
    plot_examples: bool = True,
):
    """Run R WBS2 benchmark for a single dataset path and series length."""
    df = pd.read_pickle(path)
    print(f"\n=== R WBS2 Benchmark ({label}) ===")
    print(f"Total series: {len(df):,}")
    print(f"Break distribution: {df['n_breaks'].value_counts().sort_index().to_dict()}")
    print()

    detector = RWBS2Detector(
        universal=True,
        lambda_param=lambda_param,
        cusums=cusums,
        th_const_min_mult=th_const_min_mult,
    )

    series_cols = [f"t_{i}" for i in range(series_length)]
    ts = df[series_cols].values
    true_breaks = df['break_points'].tolist()

    detected = []
    times = []
    print(f"Running R WBS2 detection on {series_length}-length series...")
    start = time.time()
    for i, series in enumerate(ts):
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1:,} series...")
        t0 = time.time()
        det = detector.detect_breaks_fast(series)
        t1 = time.time()
        detected.append(det)
        times.append(t1 - t0)
    total = time.time() - start
    print(f"Detection completed in {total:.2f} seconds")
    if times:
        print(f"Average time per series: {np.mean(times)*1000:.2f} ms")
    print()

    results = evaluate_detection_performance(
        true_breaks[:len(detected)],
        detected,
        series_length,
        tolerance_percentage=tolerance_percentage,
    )
    _print_overall_metrics(results, label)

    # No-break diagnostics (comparable across detectors)
    true_eval = true_breaks[:len(detected)]
    no_break_indices = [i for i, tb in enumerate(true_eval) if len(tb) == 0]
    n_no_break_series = len(no_break_indices)
    if n_no_break_series > 0:
        no_break_zero_detect = sum(1 for i in no_break_indices if len(detected[i]) == 0)
        no_break_any_detect = n_no_break_series - no_break_zero_detect
        no_break_spurious_counts = [len(detected[i]) for i in no_break_indices]
        no_break_tnr = no_break_zero_detect / n_no_break_series
        no_break_far = no_break_any_detect / n_no_break_series
        no_break_mean_spurious = float(np.mean(no_break_spurious_counts))
    else:
        no_break_zero_detect = 0
        no_break_any_detect = 0
        no_break_tnr = np.nan
        no_break_far = np.nan
        no_break_mean_spurious = np.nan

    print("\n=== No-Break Diagnostics ===")
    print(f"No-break series: {n_no_break_series}")
    if n_no_break_series > 0:
        print(f"TNR_0 (zero detections on no-break): {no_break_tnr:.3f}")
        print(f"FAR_0 (>=1 false alarm on no-break): {no_break_far:.3f}")
        print(f"Mean spurious breaks on no-break: {no_break_mean_spurious:.3f}")
    else:
        print("No no-break series available in evaluated set.")

    by_break_count = _group_performance(
        true_breaks[:len(detected)],
        detected,
        [len(tb) for tb in true_breaks[:len(detected)]],
        series_length,
        tolerance_percentage=tolerance_percentage,
    )
    _print_group_metrics(by_break_count, "Break Count", formatter=lambda v: f"{v} breaks")

    by_break_type = _group_performance(
        true_breaks[:len(detected)],
        detected,
        df['primary_break_type'].iloc[:len(detected)].tolist(),
        series_length,
        tolerance_percentage=tolerance_percentage,
    )
    _print_group_metrics(by_break_type, "Break Type", formatter=lambda v: v.replace('_', ' ').title())

    if plot_examples and detected:
        plot_detection_examples(df.iloc[:len(detected)], detected, n_examples=min(6, len(detected)))

    return {
        'dataset': df,
        'detected_breaks': detected,
        'overall_results': results,
        'results_by_break_count': by_break_count,
        'results_by_type': by_break_type,
        'detection_times': times,
        'no_break_metrics': {
            'n_no_break_series': n_no_break_series,
            'no_break_zero_detect': no_break_zero_detect,
            'no_break_any_detect': no_break_any_detect,
            'tnr_0': no_break_tnr,
            'far_0': no_break_far,
            'mean_spurious_breaks_no_break': no_break_mean_spurious,
        },
    }


def run_r_wbs2_benchmark():
    """Run comprehensive R WBS2 benchmark on constrained 500/1000 datasets."""
    print("=== R WBS2 Performance Benchmark ===\n")

    res_500 = run_r_wbs2_benchmark_for(
        path='synthetic_breaks_100_500_min50.pkl',
        series_length=500,
        label='500 constrained',
    )
    print("\n")
    res_1000 = run_r_wbs2_benchmark_for(
        path='synthetic_breaks_100_1000_min100.pkl',
        series_length=1000,
        label='1000 constrained',
    )

    return (
        res_500['dataset'],
        res_500['detected_breaks'],
        res_500['overall_results'],
        res_500['results_by_type'],
        res_500['detection_times'],
        res_1000['dataset'],
        res_1000['detected_breaks'],
        res_1000['overall_results'],
        res_1000['results_by_type'],
        res_1000['detection_times'],
    )


# Run the benchmark
if __name__ == "__main__":
    results = run_r_wbs2_benchmark()