import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("📊 Portfolio Optimization from Price Data")

# Sidebar: risk-free rate input
st.sidebar.header("⚙️ Settings")
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (annual)",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.005,
    format="%.4f",
    help="Annualized risk-free rate used to compute the Sharpe ratio. Enter as a decimal (e.g., 0.04 for 4%).",
)

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def load_prices(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # Strip column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # First column is date (e.g., "1/1/2014")
    date_col = df.columns[0]
    # Convert date column to datetime (day/month/year format)
    df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
    # Set date as index
    df.set_index(date_col, inplace=True)
    
    # Convert all remaining columns (asset prices) to numeric
    # errors='coerce' turns invalid entries (e.g., "1,000") into NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Optional: drop rows where all price columns are NaN
    df.dropna(how='all', inplace=True)
    
    return df

def compute_returns(prices, method='simple'):
    """Compute monthly returns from price data."""
    if method == 'simple':
        returns = prices.pct_change().dropna()
    else:
        returns = np.log(prices / prices.shift(1)).dropna()
    return returns

def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.sum(weights * mean_returns)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_risk

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0):
    p_return, p_risk = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_risk

def optimize_portfolio(mean_returns, cov_matrix, bounds, risk_free_rate=0):
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    initial_weights = np.array(num_assets * [1 / num_assets])
    result = minimize(
        negative_sharpe,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x

def efficient_frontier(mean_returns, cov_matrix, bounds, points=50):
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), points)
    risks = []
    num_assets = len(mean_returns)
    constraints_base = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    initial_weights = np.array(num_assets * [1 / num_assets])
    
    def portfolio_risk(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    for target in target_returns:
        constraints = (
            constraints_base,
            {'type': 'eq', 'fun': lambda w, t=target: np.sum(w * mean_returns) - t}
        )
        opt = minimize(
            portfolio_risk,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        risks.append(opt.fun if opt.success else np.nan)
    return target_returns, risks

# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file with price data", type=["csv"])
if uploaded_file is not None:
    # Load price data
    prices = load_prices(uploaded_file)
    st.subheader("📅 Price Data Preview")
    st.dataframe(prices.head())
    
    # Exclude a benchmark column if present (e.g., "DSEX")
    all_assets = prices.columns.tolist()
    benchmark_col = st.selectbox("Select benchmark column (if any, to exclude from assets)", ["None"] + all_assets)
    if benchmark_col != "None":
        asset_cols = [col for col in all_assets if col != benchmark_col]
        benchmark_prices = prices[benchmark_col]
    else:
        asset_cols = all_assets
        benchmark_prices = None
    
    # Compute returns
    returns = compute_returns(prices[asset_cols], method='simple')
    st.subheader("📈 Monthly Returns (computed from prices)")
    st.dataframe(returns.head())

    # ------------------ Portfolio Resizing ------------------
    st.subheader("✂️ Portfolio Resizing")
    total_assets = len(asset_cols)

    quality_scores = (returns.mean() * 12) / (returns.std() * np.sqrt(12))
    quality_scores = quality_scores.replace([np.inf, -np.inf], np.nan).dropna()
    quality_df = quality_scores.sort_values(ascending=False).to_frame("Sharpe (individual)")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        n_keep = st.number_input(
            "Number of stocks to keep",
            min_value=2,
            max_value=total_assets,
            value=total_assets,
            step=1,
            help="Drops the lowest-quality stocks based on individual annualized Sharpe ratio (mean / std).",
        )
    with col_b:
        st.caption(f"Ranking {total_assets} candidate assets by individual Sharpe ratio.")

    kept_assets = quality_df.head(int(n_keep)).index.tolist()
    dropped_assets = quality_df.tail(total_assets - int(n_keep)).index.tolist()

    col_k, col_d = st.columns(2)
    with col_k:
        st.write(f"**✅ Kept ({len(kept_assets)})**")
        st.dataframe(quality_df.loc[kept_assets].style.format({"Sharpe (individual)": "{:.3f}"}))
    with col_d:
        st.write(f"**❌ Dropped ({len(dropped_assets)})**")
        if dropped_assets:
            st.dataframe(quality_df.loc[dropped_assets].style.format({"Sharpe (individual)": "{:.3f}"}))
        else:
            st.info("No assets dropped — keeping all.")

    asset_cols = kept_assets
    returns = returns[asset_cols]

    # Annualize monthly returns & covariance
    mean_returns = returns.mean() * 12
    cov_matrix = returns.cov() * 12
    
    st.subheader("📊 Asset Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Mean Annual Returns**")
        st.dataframe(mean_returns.to_frame("Return").style.format("{:.2%}"))
    with col2:
        st.write("**Covariance Matrix** (annualized)")
        st.dataframe(cov_matrix.style.format("{:.4f}"))
    
    # ------------------ No short selling ------------------
    st.header("🔒 Optimal Portfolio (No Short Selling)")
    bounds_no_short = tuple((0, 1) for _ in asset_cols)
    opt_weights_no_short = optimize_portfolio(mean_returns, cov_matrix, bounds_no_short, risk_free_rate)
    opt_return, opt_risk = portfolio_performance(opt_weights_no_short, mean_returns, cov_matrix)
    sharpe = (opt_return - risk_free_rate) / opt_risk if opt_risk > 0 else np.nan

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Annual Return", f"{opt_return:.2%}")
    col2.metric("Annual Volatility", f"{opt_risk:.2%}")
    col3.metric(f"Sharpe Ratio (rf={risk_free_rate:.2%})", f"{sharpe:.3f}")
    
    weights_df = pd.DataFrame({"Asset": asset_cols, "Weight": opt_weights_no_short})
    weights_df = weights_df[weights_df["Weight"] > 0.001].sort_values("Weight", ascending=False)
    st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}))
    
    # Efficient Frontier plot
    st.subheader("📉 Efficient Frontier")
    fig, ax = plt.subplots(figsize=(10, 6))
    targets, risks = efficient_frontier(mean_returns, cov_matrix, bounds_no_short)
    ax.plot(risks, targets, 'b-', label="Efficient Frontier")
    ax.scatter(opt_risk, opt_return, marker='*', s=200, c='red', label="Max Sharpe Portfolio")
    ax.set_xlabel("Annual Portfolio Risk (Std Dev)")
    ax.set_ylabel("Annual Expected Return")
    ax.set_title("Efficient Frontier (No Short Selling)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # ------------------ Backtest ------------------
    st.header("💰 Backtest — Investment Value Over Time")
    st.caption(
        "Hypothetical buy-and-hold backtest: the investment amount is allocated at the "
        "first date using the optimal weights above (no short selling) and held without rebalancing."
    )

    col_amt, col_info = st.columns([1, 2])
    with col_amt:
        investment_amount = st.number_input(
            "Investment amount",
            min_value=1.0,
            value=10000.0,
            step=1000.0,
            format="%.2f",
            help="Initial lump sum invested at the first available price date.",
        )

    bt_prices = prices[asset_cols].dropna(how='any')
    if len(bt_prices) < 2:
        st.warning("Not enough price observations for a backtest.")
    else:
        start_date = bt_prices.index[0]
        end_date = bt_prices.index[-1]
        n_periods = len(bt_prices) - 1
        years = (end_date - start_date).days / 365.25

        with col_info:
            st.caption(
                f"Backtest window: **{start_date.date()} → {end_date.date()}** "
                f"({n_periods} periods, ≈ {years:.2f} years)."
            )

        normalized = bt_prices / bt_prices.iloc[0]
        weights_series = pd.Series(opt_weights_no_short, index=asset_cols)
        per_asset_value = normalized.multiply(weights_series, axis=1) * investment_amount
        portfolio_value = per_asset_value.sum(axis=1)

        final_value = portfolio_value.iloc[-1]
        total_return = final_value / investment_amount - 1
        cagr = (final_value / investment_amount) ** (1 / years) - 1 if years > 0 else np.nan
        running_max = portfolio_value.cummax()
        drawdown = portfolio_value / running_max - 1
        max_drawdown = drawdown.min()
        profit = final_value - investment_amount

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Initial", f"{investment_amount:,.2f}")
        m2.metric("Final Value", f"{final_value:,.2f}", f"{profit:+,.2f}")
        m3.metric("Total Return", f"{total_return:.2%}")
        m4.metric("CAGR", f"{cagr:.2%}" if not np.isnan(cagr) else "n/a")
        m5.metric("Max Drawdown", f"{max_drawdown:.2%}")

        st.subheader("📈 Portfolio Value Over Time")
        fig_bt, ax_bt = plt.subplots(figsize=(10, 5))
        ax_bt.plot(portfolio_value.index, portfolio_value.values, 'b-', linewidth=2, label="Optimized Portfolio")

        if benchmark_prices is not None:
            bench_aligned = benchmark_prices.reindex(portfolio_value.index).dropna()
            if len(bench_aligned) >= 2:
                bench_value = bench_aligned / bench_aligned.iloc[0] * investment_amount
                ax_bt.plot(bench_value.index, bench_value.values, 'orange', linestyle='--',
                           linewidth=1.5, label=f"Benchmark ({benchmark_col})")

        ax_bt.axhline(investment_amount, color='gray', linestyle=':', linewidth=1, label="Initial Investment")
        ax_bt.set_xlabel("Date")
        ax_bt.set_ylabel("Portfolio Value")
        ax_bt.set_title("Buy-and-Hold Backtest")
        ax_bt.legend()
        ax_bt.grid(True, alpha=0.3)
        st.pyplot(fig_bt)

        st.subheader("🧾 Per-Asset Allocation & Final Value")
        alloc_df = pd.DataFrame({
            "Asset": asset_cols,
            "Weight": opt_weights_no_short,
            "Initial Allocation": opt_weights_no_short * investment_amount,
            "Final Value": per_asset_value.iloc[-1].values,
        })
        alloc_df["Profit / Loss"] = alloc_df["Final Value"] - alloc_df["Initial Allocation"]
        alloc_df["Return %"] = alloc_df["Final Value"] / alloc_df["Initial Allocation"] - 1
        alloc_df = alloc_df[alloc_df["Weight"] > 0.001].sort_values("Weight", ascending=False)
        st.dataframe(
            alloc_df.style.format({
                "Weight": "{:.2%}",
                "Initial Allocation": "{:,.2f}",
                "Final Value": "{:,.2f}",
                "Profit / Loss": "{:+,.2f}",
                "Return %": "{:.2%}",
            })
        )

        bt_csv = portfolio_value.to_frame("Portfolio Value").to_csv().encode('utf-8')
        st.download_button("Download Backtest Series (CSV)", bt_csv, "backtest_value.csv", "text/csv")

    # ------------------ Short selling allowed ------------------
    st.header("📊 Optimal Portfolio (Short Selling Allowed)")
    with st.expander("Show short selling analysis"):
        bounds_short = tuple((-1, 1) for _ in asset_cols)
        opt_weights_short = optimize_portfolio(mean_returns, cov_matrix, bounds_short, risk_free_rate)
        ret_short, risk_short = portfolio_performance(opt_weights_short, mean_returns, cov_matrix)
        sharpe_short = (ret_short - risk_free_rate) / risk_short if risk_short > 0 else np.nan

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Return", f"{ret_short:.2%}")
        col2.metric("Volatility", f"{risk_short:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe_short:.3f}")
        
        weights_short_df = pd.DataFrame({"Asset": asset_cols, "Weight": opt_weights_short})
        weights_short_df = weights_short_df.sort_values("Weight", ascending=False)
        st.dataframe(weights_short_df.style.format({"Weight": "{:.2%}"}))
        
        # Efficient frontier with short selling
        targets_short, risks_short = efficient_frontier(mean_returns, cov_matrix, bounds_short)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(risks_short, targets_short, 'g-', label="Efficient Frontier (short allowed)")
        ax2.scatter(risk_short, ret_short, marker='*', s=200, c='red', label="Max Sharpe")
        ax2.set_xlabel("Annual Portfolio Risk")
        ax2.set_ylabel("Annual Expected Return")
        ax2.set_title("Efficient Frontier with Short Selling")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)
    
    # ------------------ Covariance Heatmap ------------------
    st.subheader("🔥 Covariance Matrix Heatmap")
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    sns.heatmap(cov_matrix, annot=True, fmt=".4f", cmap="coolwarm", ax=ax3)
    ax3.set_title("Annualized Covariance Matrix")
    st.pyplot(fig3)
    
    # ------------------ Download weights ------------------
    st.subheader("📥 Download Results")
    csv_no_short = weights_df.to_csv(index=False).encode('utf-8')
    csv_short = weights_short_df.to_csv(index=False).encode('utf-8')
    col1, col2 = st.columns(2)
    col1.download_button("Download No‑Short Weights", csv_no_short, "weights_no_short.csv", "text/csv")
    col2.download_button("Download Short‑Selling Weights", csv_short, "weights_short.csv", "text/csv")
    
    st.caption(f"Annualized statistics are derived from monthly returns (×12). Risk‑free rate = {risk_free_rate:.2%}.")