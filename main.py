import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("📊 Portfolio Optimization from Monthly Returns")

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def load_returns(uploaded_file):
    """Load CSV, drop Year, Month, DSEX, return asset returns."""
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    # Drop Year, Month, and DSEX (benchmark)
    returns = df.drop(columns=["Year", "Month", "DSEX"], errors="ignore")
    return returns

def portfolio_performance(weights, mean_returns, cov_matrix):
    """Annualized return, risk (volatility)."""
    port_return = np.sum(weights * mean_returns)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_risk

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0):
    p_return, p_risk = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_risk

def optimize_portfolio(mean_returns, cov_matrix, bounds):
    """Find weights that maximize Sharpe ratio."""
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    initial_weights = np.array(num_assets * [1 / num_assets])
    result = minimize(
        negative_sharpe,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x

def efficient_frontier(mean_returns, cov_matrix, bounds, points=50):
    """Compute efficient frontier for a range of target returns."""
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
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    returns = load_returns(uploaded_file)
    asset_names = returns.columns.tolist()
    
    # Annualize monthly returns & covariance
    mean_returns = returns.mean() * 12
    cov_matrix = returns.cov() * 12
    
    st.subheader("📈 Asset Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Mean Annual Returns**")
        st.dataframe(mean_returns.to_frame("Return").style.format("{:.2%}"))
    with col2:
        st.write("**Covariance Matrix** (annualized)")
        st.dataframe(cov_matrix.style.format("{:.4f}"))
    
    # ------------------ No short selling ------------------
    st.header("🔒 Optimal Portfolio (No Short Selling)")
    bounds_no_short = tuple((0, 1) for _ in asset_names)
    opt_weights_no_short = optimize_portfolio(mean_returns, cov_matrix, bounds_no_short)
    opt_return, opt_risk = portfolio_performance(opt_weights_no_short, mean_returns, cov_matrix)
    sharpe = opt_return / opt_risk
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Annual Return", f"{opt_return:.2%}")
    col2.metric("Annual Volatility", f"{opt_risk:.2%}")
    col3.metric("Sharpe Ratio (rf=0)", f"{sharpe:.3f}")
    
    weights_df = pd.DataFrame({"Asset": asset_names, "Weight": opt_weights_no_short})
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
    
    # ------------------ Short selling allowed ------------------
    st.header("📊 Optimal Portfolio (Short Selling Allowed)")
    with st.expander("Show short selling analysis"):
        bounds_short = tuple((-1, 1) for _ in asset_names)
        opt_weights_short = optimize_portfolio(mean_returns, cov_matrix, bounds_short)
        ret_short, risk_short = portfolio_performance(opt_weights_short, mean_returns, cov_matrix)
        sharpe_short = ret_short / risk_short
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Return", f"{ret_short:.2%}")
        col2.metric("Volatility", f"{risk_short:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe_short:.3f}")
        
        weights_short_df = pd.DataFrame({"Asset": asset_names, "Weight": opt_weights_short})
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
    
    st.caption("Annualized statistics are derived from monthly returns (×12). Risk‑free rate assumed 0%.")