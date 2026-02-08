# Structural Break Detection (Master Thesis)

Code and experiments for comparing structural break / changepoint detectors on:

- **Synthetic datasets** with labelled breakpoints (constrained and unconstrained breakpoint spacing)
- **Semi-synthetic “real” stock-return datasets** created by concatenating segments from different stocks (concatenation points are labelled as breaks)

## What’s in this repo

- `results_notebook.ipynb`: main notebook to generate datasets, run benchmarks, and produce plots.
- `synthetic_data_generator.py`: synthetic structural-break generator.
- `real_data_generator.py`: downloads stock data (via `yfinance`), computes log returns, and builds concatenated series.
- `wbs2_detector.py`: WBS2-SDLL (R) via `rpy2` + benchmarking utilities.
- `spike_slab_detector.py`: Spike-and-Slab / SoloCP-style (R) via `rpy2` + benchmarking utilities.
- `bilstm_detector.py`: BiLSTM model (PyTorch) training + inference utilities.

Data/model artifacts currently present:

- `synthetic_breaks_*.pkl`
- `real_stock_breaks.pkl`
- `clean_bilstm_model.pth`

## Setup

### 1) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install Python dependencies

At minimum you’ll need:

- `numpy`, `pandas`, `matplotlib`, `scipy`
- `torch` (BiLSTM)
- `yfinance` (real stock dataset)
- `rpy2` (to call the R implementations)

```bash
pip install numpy pandas matplotlib scipy torch yfinance rpy2
```

### 3) R requirements (for WBS2 + Spike-and-Slab)

You must have **R installed** and accessible so `rpy2` can call it.

The R implementations used by this project are embedded in the Python files (`wbs2_detector.py`, `spike_slab_detector.py`).

## Usage

### A) Synthetic datasets (constrained vs unconstrained)

Use `results_notebook.ipynb` to generate and save synthetic datasets. Example outputs:

- `synthetic_breaks_100_500_min50.pkl` (constrained)
- `synthetic_breaks_100_500_unconstrained.pkl` (unconstrained)
- `synthetic_breaks_100_1000_min100.pkl` (constrained)
- `synthetic_breaks_100_1000_unconstrained.pkl` (unconstrained)

**Constrained** means a minimum spacing between breaks is enforced during generation; **unconstrained** allows breaks to cluster.

### B) Real stock dataset (semi-synthetic)

In a notebook cell:

```python
from real_data_generator import generate_real_stock_dataset, StockDataConfig1

config = StockDataConfig1(
    n_series=100,
    series_length=1000,
    max_breaks=3,
    min_segment_length=100,
    start_date="2000-01-01",
    end_date="2017-12-31",
    seed=42,
)

real_df = generate_real_stock_dataset(config, cache_file="real_stock_breaks.pkl")
```

Notes:

- Internet is required on first run (uses `yfinance`).
- Concatenation points are labelled breaks; internal breaks inside segments may still exist.

### C) Run benchmarks

Benchmarks are run from `results_notebook.ipynb` (recommended).

- **WBS2**: see `wbs2_detector.py`
- **Spike-and-Slab**: see `spike_slab_detector.py`
- **BiLSTM**: see `bilstm_detector.py`

Keep evaluation settings consistent (tolerance %, thresholds, stride, etc.) when comparing methods.


