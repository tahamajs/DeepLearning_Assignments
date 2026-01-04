# This_year/CA4 — Sequence Modeling

This folder contains CA4 materials: image captioning and time-series forecasting experiments.

Structure:
- `Image_Captioning/` — datasets, generation scripts and notebooks
- `codes/notebooks/ATIS_and_SP500_Experiments.ipynb` — S&P500 forecasting experiments and ATIS SLU baselines
- `report/` — assignment writeup and figures

How to run:
- For S&P500 experiments, run `codes/notebooks/ATIS_and_SP500_Experiments.ipynb`. Helper scripts under `codes/` perform training and plotting.
- For image captioning, see `Image_Captioning/README.md` and `codes/` for dataset extraction and training commands.

Notes:
- Time-series experiments may require `yfinance` and `statsmodels` packages (use `pip install -r requirements.txt`).
