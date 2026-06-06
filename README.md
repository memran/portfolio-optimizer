# 📊 Portfolio Optimizer — Streamlit Web App

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/memran/portfolio-optimizer.svg)](https://github.com/memran/portfolio-optimizer/commits/main)

> A web-based portfolio optimization tool that computes the optimal asset allocation by maximizing the Sharpe ratio, plots the efficient frontier, and visualizes asset covariance — all from a CSV upload.

---

## Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Which Script Do I Run?](#-which-script-do-i-run)
- [Quick Start](#-quick-start)
- [CSV File Format](#-csv-file-format)
- [Usage Walkthrough](#-usage-walkthrough)
- [Methodology](#-methodology)
- [Outputs](#-outputs)
- [Assumptions & Limitations](#-assumptions--limitations)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## ✨ Features

- 📤 **CSV upload** — works with either raw prices or pre-computed returns.
- 🧮 **Automatic return calculation** (simple monthly returns) from price data.
- 🎯 **Two optimization modes:**
  - Maximum Sharpe ratio (long-only / no short selling)
  - Maximum Sharpe ratio (short selling allowed)
- 📈 **Performance metrics** — annualized return, volatility, Sharpe ratio.
- 📉 **Efficient frontier** plot.
- 🔥 **Covariance matrix heatmap.**
- 🏷️ **Optional benchmark column** (e.g., `DSEX`) that can be excluded from the asset set.
- 📥 **Downloadable weight CSVs** for both optimization modes.

---

## 📂 Project Structure

```
portfolio-optimizer/
├── app_price.py            # Streamlit app — input CSV contains raw prices
├── main.py                 # Streamlit app — input CSV contains pre-computed monthly returns
├── data/
│   └── sample_prices.csv   # Sample price file (try the app instantly)
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

---

## 🤔 Which Script Do I Run?

| If your CSV contains…                       | Run                              |
| ------------------------------------------- | -------------------------------- |
| **Raw prices** (e.g., monthly closing)      | `streamlit run app_price.py`     |
| **Pre-computed monthly returns**            | `streamlit run main.py`          |

If unsure, use **`app_price.py`** — it derives returns automatically from prices.

---

## 🚀 Quick Start

### Option A — Using `uv` (recommended, matches `uv.lock`)

```bash
git clone https://github.com/memran/portfolio-optimizer.git
cd portfolio-optimizer
uv sync
uv run streamlit run app_price.py
```

### Option B — Using `pip` + `venv`

```bash
git clone https://github.com/memran/portfolio-optimizer.git
cd portfolio-optimizer

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app_price.py
```

The app will open in your browser at <http://localhost:8501>.

---

## 📑 CSV File Format

### Prices mode (`app_price.py`)

- First column: **date** in **`dd/mm/yyyy`** format (e.g., `01/01/2014`).
- Remaining columns: **asset closing prices** (numeric).
- Optional **benchmark** column (e.g., `DSEX`) — you'll be prompted to exclude it in the UI.

**Example** (`data/sample_prices.csv`):

```csv
Date,Eastern Bank Ltd,Islami Bank Limited,BRAC Bank,DSEX
01/01/2014,100.0,95.5,102.3,5000
01/02/2014,103.5,94.2,98.7,5100
01/03/2014,105.1,96.8,101.4,5180
...
```

> ⚠️ The date parser is hard-coded to `%d/%m/%Y`. Other formats will produce `NaT` and drop those rows.

### Returns mode (`main.py`)

- Columns: `Year`, `Month`, one column per asset, optional `DSEX` (benchmark — auto-dropped).
- Values: **monthly returns as decimals** (e.g., `0.0123` = 1.23%).

**Example:**

```csv
Year,Month,Eastern Bank Ltd,Islami Bank Limited,BRAC Bank,DSEX
2014,1,0.0123,-0.0045,0.0210,0.0150
2014,2,0.0089,0.0067,-0.0034,0.0098
...
```

---

## 🧭 Usage Walkthrough

1. **Upload** your CSV via the file uploader.
2. (Prices mode) **Select a benchmark** to exclude, or choose `None`.
3. Review the **Asset Summary** — annualized mean returns and covariance matrix.
4. Inspect the **Optimal Portfolio (No Short Selling)** card — expected return, volatility, Sharpe ratio, and weights.
5. View the **Efficient Frontier** plot — the red star marks the max-Sharpe portfolio.
6. Expand **Optimal Portfolio (Short Selling Allowed)** for the unconstrained variant.
7. Examine the **Covariance Heatmap** for asset interdependence.
8. **Download** weight CSVs for either mode.

---

## 📐 Methodology

Built on **Markowitz mean–variance optimization** (Modern Portfolio Theory).

**Sharpe ratio**

```
S = (Rₚ − R_f) / σₚ
```

where `Rₚ` is portfolio return, `R_f` is the risk-free rate (assumed `0`), and `σₚ` is portfolio volatility.

**Annualization** (monthly → annual)

```
μ_annual = μ_monthly × 12
Σ_annual = Σ_monthly × 12
```

**Optimization**

- Solver: SciPy `minimize` with **SLSQP**.
- Constraint: `Σ wᵢ = 1` (fully invested).
- Bounds:
  - Long-only: `0 ≤ wᵢ ≤ 1`
  - Short selling allowed: `−1 ≤ wᵢ ≤ 1`
- Objective: minimize `−Sharpe`.

The efficient frontier is computed by sweeping target returns and minimizing portfolio variance for each.

---

## 📦 Outputs

- Mean annual returns table.
- Annualized covariance matrix.
- Optimal portfolio weights (no short selling).
- Efficient frontier chart with max-Sharpe marker.
- Short-selling-allowed analysis (expandable section).
- Covariance heatmap.
- Download buttons for weight CSV files.

---

## ⚠️ Assumptions & Limitations

- **Risk-free rate = 0** (not user-configurable yet).
- **Simple returns** (`pct_change`) by default; log-return mode exists in code but is not exposed in the UI.
- **No transaction costs, taxes, or slippage.**
- **No position limits** other than the bounds above.
- **Annualization factor = 12** — assumes monthly data. Daily/weekly inputs will be mis-annualized.
- **Date format must be `dd/mm/yyyy`** in prices mode.
- **Historical performance ≠ future returns.** This tool is for education / research, not investment advice.

---

## ☁️ Deployment

Deploy to **Streamlit Community Cloud** in three steps:

1. Push the repo to GitHub (already done if you cloned this).
2. Go to <https://share.streamlit.io> and click **New app**.
3. Select your repo, branch `main`, and set **Main file path** to `app_price.py` (or `main.py`).

The first deploy takes ~1–2 minutes; subsequent pushes redeploy automatically.

---

## 🛟 Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `ValueError: time data ... doesn't match format` | Dates are not `dd/mm/yyyy` | Re-format the date column or adjust `format=` in `app_price.py:22` |
| Empty results / all-NaN returns | Non-numeric cells (e.g., `"1,000"`) coerced to NaN | Clean the CSV — remove thousand separators and stray text |
| Optimization fails or weights look random | Too few rows, or singular covariance matrix | Provide more history; ensure at least ~24 monthly observations |
| Benchmark column appears in the optimization | `DSEX` (or your benchmark) wasn't excluded | Select it in the **"benchmark column"** dropdown |
| `streamlit: command not found` | Virtualenv not activated, or install failed | Re-activate venv and re-run `pip install -r requirements.txt` |

---

## 🗺️ Roadmap

- [ ] Configurable **risk-free rate** in the UI.
- [ ] Configurable **annualization factor** (daily / weekly / monthly).
- [ ] **Auto-detect** date formats in `app_price.py`.
- [ ] Merge `main.py` and `app_price.py` into a single app with a mode toggle.
- [ ] Additional optimization objectives: **min-variance**, **risk parity**, **Black–Litterman**.
- [ ] Export full report (PDF / HTML).

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repo and create your branch: `git checkout -b feature/my-feature`
2. Commit your changes: `git commit -m "feat: add my feature"`
3. Push to the branch: `git push origin feature/my-feature`
4. Open a Pull Request.

---

## 📄 License

Released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Markowitz, H. (1952). *Portfolio Selection*. **The Journal of Finance**, 7(1), 77–91.
- [SciPy SLSQP documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
- [Streamlit documentation](https://docs.streamlit.io/)
