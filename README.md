# Regime-Aware Deep Reinforcement Learning for Portfolio Management

Bachelor's thesis — Computer Engineering, German University in Cairo (GUC)
Author: **Shorouk Rafik** · Supervisor: Dr. Walaa Gad

## Overview
A deep reinforcement learning agent that allocates a stock portfolio and adapts its behaviour to the current market regime.

- **Agent:** PPO (Stable-Baselines3) with a multi-head attention feature extractor over the stocks in the universe
- **Regime detection:** Hidden Markov Model (hmmlearn) that classifies the market into bull / bear / sideways states
- **Reward:** regime-conditioned, penalising drawdowns more heavily in bear regimes
- **Environment:** custom Gymnasium portfolio environment with technical-indicator features
- **Markets:** Dow Jones 30, S&P 100 and Nasdaq-100
- **Baselines and checks:** Markowitz mean-variance, equal weight, ablations (no attention / no regime), multi-seed runs, bootstrap confidence intervals and walk-forward evaluation

## Repository structure
```
models/      training, evaluation and analysis scripts (portfolio env, PPO trainers, MAC meta-controller, baselines)
results/     metrics (JSON/MD) and figures for every experiment; see results/INDEX.md
dashboard/   Streamlit demo: an investor simulator running the agent on 2025 out-of-sample data
images/      thesis figures (PNG/PDF)
build_fig_*.py   scripts that generate the thesis and defense figures
docs/        literature comparison
```

## Running
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements stable-baselines3 gymnasium torch
pip install -r dashboard/requirements.txt

# demo dashboard
streamlit run dashboard/app.py
```

**Not included** because of GitHub's size limits: the raw/processed price data (`data/`, downloaded with yfinance) and the trained model checkpoints (`models/**/*.zip`). Regenerate them with the data-preparation and training scripts in `models/`.

## Tech stack
Python · Stable-Baselines3 (PPO) · PyTorch · Gymnasium · hmmlearn · pandas · NumPy · scikit-learn · yfinance · ta · Matplotlib · Plotly · Streamlit
