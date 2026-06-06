# 📊 Portfolio Optimizer – Streamlit Web App

A web-based portfolio optimization tool built with **Streamlit**.  
Upload a CSV file containing **historical price data** (e.g., monthly closing prices of stocks, ETFs, or indices). The app automatically computes monthly returns, annualizes them, and finds the optimal portfolio that maximizes the Sharpe ratio (with and without short selling). It also plots the efficient frontier and displays a covariance heatmap.

## ✨ Features

- Upload price data – CSV with dates and asset columns.
- Automatic return calculation (simple monthly returns).
- Optional benchmark column (e.g., market index) to exclude from assets.
- Optimal portfolios:
  - Maximum Sharpe ratio (no short selling)
  - Maximum Sharpe ratio (short selling allowed)
- Performance metrics: annualized return, volatility, Sharpe ratio.
- Efficient frontier plot.
- Covariance matrix heatmap.
- Download portfolio weights as CSV files.

## 🧰 Tech Stack

- Python 3.8+
- Streamlit – web UI
- Pandas – data manipulation
- NumPy – numerical operations
- SciPy – portfolio optimisation (SLSQP)
- Matplotlib & Seaborn – plotting

## 📂 CSV File Format

The input CSV must contain **price data** with the first column as **date**.  
Each subsequent column represents an asset's closing price. An optional benchmark column (e.g., `DSEX`) can be selected and excluded.

**Example:**

```csv
Date, Eastern Bank Ltd, Islami Bank Limited, BRAC Bank, DSEX
1/1/2014, 100.0, 95.5, 102.3, 5000
1/2/2014, 103.5, 94.2, 98.7, 5100
...
```

The date format can be any format recognized by pandas.to_datetime() (e.g., 1/1/2014, 2014-01-01).

## Installation & Usage

1. Clone the repository

```
   git clone https://github.com/memran/portfolio-optimizer.git
   cd portfolio-optimizer
```

2. Create a virtual environment (recommended)

```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```
4. Run the app

```
streamlit run app.py
The app will open in your browser at http://localhost:8501.
```

### Output

Mean annual returns table
Annualized covariance matrix
Optimal portfolio weights (no short selling)
Efficient frontier chart
Short selling allowed analysis (expandable section)
Covariance heatmap
Download buttons for weight CSV files

📄 License
MIT

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first.
