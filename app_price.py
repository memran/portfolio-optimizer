import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.optimize import minimize
from io import BytesIO
from fpdf import FPDF

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
investment_amount = st.sidebar.number_input(
    "Investment amount ($)",
    min_value=1.0,
    value=10000.0,
    step=1000.0,
    format="%.2f",
    help="Initial lump sum invested at the first available price date.",
)

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def load_prices(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file as CSV: {e}")
        return None

    if df.shape[1] < 2:
        st.error(
            "File must have at least 2 columns: a date column followed by one or more price columns."
        )
        return None

    df.columns = df.columns.str.strip()
    date_col = df.columns[0]

    before = len(df)
    df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
    bad_dates = df[date_col].isna()
    if bad_dates.any():
        df = df[~bad_dates].copy()
        dropped = bad_dates.sum()
        if dropped == before:
            st.error(
                f"All {dropped} rows have unparseable dates. "
                "Expected format: **DD/MM/YYYY** (e.g., 01/01/2014)."
            )
            return None
        st.warning(f"Dropped {dropped} row(s) with invalid date format.")

    df.set_index(date_col, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    before = len(df)
    df.dropna(how='all', inplace=True)
    bad_rows = before - len(df)
    if bad_rows:
        st.warning(f"Dropped {bad_rows} row(s) with no valid price data.")

    if df.empty:
        st.error("No valid data remaining after cleaning. All price columns should contain numbers.")
        return None

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
# PDF Report
# -------------------------------------------------------------------
def save_chart(fig, name, section="opt"):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    if "report_images" not in st.session_state:
        st.session_state.report_images = []
    st.session_state.report_images.append((section, name, buf))

def generate_pdf_report(weights_df, metrics, backtest_metrics, oos_metrics=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    def add_metrics_block(title, metrics_dict):
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for k, v in metrics_dict.items():
            pdf.cell(0, 7, f"  {k}: {v}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    def add_image_by_section(section):
        for s, n, buf in st.session_state.report_images:
            if s == section:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 10, n, new_x="LMARGIN", new_y="NEXT", align="C")
                try:
                    img_w = pdf.w - 20
                    pdf.image(buf, x=10, w=img_w)
                except Exception:
                    pdf.cell(0, 10, "[Chart could not be embedded]", new_x="LMARGIN", new_y="NEXT")

    # ── Title page ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "Portfolio Optimization Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── Section 1: Optimum Portfolio Report ──
    add_metrics_block("1. Optimum Portfolio Report", metrics)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Optimal Weights", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 10)
    col_w = pdf.w / 2 - 10
    pdf.cell(col_w, 7, "Asset", border=1, align="C")
    pdf.cell(col_w, 7, "Weight", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for _, row in weights_df.iterrows():
        pdf.cell(col_w, 7, str(row["Asset"]), border=1)
        pdf.cell(col_w, 7, f"{row['Weight']:.2%}", border=1, align="C",
                 new_x="LMARGIN", new_y="NEXT")
    add_image_by_section("opt")

    # ── Section 2: Backtest Report ──
    if backtest_metrics:
        add_metrics_block("2. Backtest Report", backtest_metrics)
        add_image_by_section("bt")

    # ── Section 3: Out-of-Sample Report ──
    if oos_metrics:
        add_metrics_block("3. Out-of-Sample Report", oos_metrics)
        add_image_by_section("oos")

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()

# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file with price data", type=["csv"])

with st.expander("📄 Expected CSV format"):
    st.markdown(
        """
        **Required format:**
        - **First column:** dates in **DD/MM/YYYY** format
        - **Remaining columns:** asset price columns with numeric values
        - **Optional:** a benchmark index column (selectable below)

        **Example:**
        ```
        Date,Eastern Bank Ltd,Islami Bank Limited,BRAC Bank,Square Pharma,DSEX
        01/01/2014,100.00,95.50,102.30,210.40,5000.00
        01/02/2014,103.50,94.20,98.70,213.10,5100.00
        ```
        """
    )

if uploaded_file is not None:
    st.session_state.report_images = []
    prices = load_prices(uploaded_file)
    if prices is None:
        st.stop()
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

    st.subheader("🔥 Covariance Matrix Heatmap")
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    sns.heatmap(cov_matrix, annot=True, fmt=".4f", cmap="coolwarm", ax=ax3)
    ax3.set_title("Annualized Covariance Matrix")
    st.pyplot(fig3)

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

    # Asset Allocation and Diversification — Donut + Treemap
    import squarify
    st.subheader("🧩 Asset Allocation and Diversification")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_donut, ax_donut = plt.subplots(figsize=(6, 6))
        plot_weights = weights_df[weights_df["Weight"] > 0.01]
        other_weight = 1 - plot_weights["Weight"].sum()
        if other_weight > 0.001:
            plot_weights = pd.concat([
                plot_weights,
                pd.DataFrame([{"Asset": "Other (<1%)", "Weight": other_weight}])
            ], ignore_index=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(plot_weights)))
        wedges, texts, autotexts = ax_donut.pie(
            plot_weights["Weight"], labels=None, autopct="%1.1f%%",
            startangle=90, pctdistance=0.78, colors=colors,
            wedgeprops=dict(width=0.4, edgecolor="w", linewidth=1),
        )
        for t in autotexts:
            t.set_fontsize(8)
        ax_donut.set_title("Portfolio Weight Allocation", fontsize=12, fontweight="bold")
        ax_donut.legend(
            wedges, plot_weights["Asset"],
            title="Assets", loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8,
        )
        save_chart(fig_donut, "Portfolio Weight Allocation")
        st.pyplot(fig_donut)

    with col_d2:
        fig_tm, ax_tm = plt.subplots(figsize=(6, 5))
        tm_labels = [f"{a}\n{w:.1%}" for a, w in zip(weights_df["Asset"], weights_df["Weight"])]
        tm_colors = plt.cm.Set2(np.linspace(0, 1, len(weights_df)))
        squarify.plot(
            sizes=weights_df["Weight"].values,
            label=tm_labels,
            color=tm_colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.5,
            ax=ax_tm,
            text_kwargs={"fontsize": 9, "fontweight": "bold"},
        )
        ax_tm.set_title("Portfolio Weight Treemap", fontsize=12, fontweight="bold")
        ax_tm.axis("off")
        save_chart(fig_tm, "Portfolio Weight Treemap")
        st.pyplot(fig_tm)

    # Risk vs Return Analysis — Scatter Plot
    st.subheader("🎯 Risk vs Return Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))

    asset_risks = returns.std() * np.sqrt(12)
    asset_returns = returns.mean() * 12
    asset_sharpe = (asset_returns - risk_free_rate) / asset_risks

    sc = ax.scatter(asset_risks, asset_returns, c=asset_sharpe, cmap="RdYlGn",
                    s=120, edgecolors="black", linewidth=0.8, zorder=5)
    for asset in asset_cols:
        ax.annotate(asset, (asset_risks[asset], asset_returns[asset]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")

    targets, risks = efficient_frontier(mean_returns, cov_matrix, bounds_no_short)
    ax.plot(risks, targets, 'b-', linewidth=1.5, alpha=0.7, label="Efficient Frontier")

    ax.scatter(opt_risk, opt_return, marker='*', s=300, c='red', edgecolors="darkred",
               linewidth=1, zorder=6, label="Max Sharpe Portfolio")

    cbar = plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    ax.set_xlabel("Annual Portfolio Risk (Std Dev)")
    ax.set_ylabel("Annual Expected Return")
    ax.set_title("Risk vs Return — Individual Assets & Efficient Frontier")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    save_chart(fig, "Risk vs Return Analysis")
    st.pyplot(fig)

    # ------------------ Backtest ------------------
    st.header("💰 Backtest — Investment Value Over Time")
    st.caption(
        "Hypothetical buy-and-hold backtest: the investment amount is allocated at the "
        "first date using the optimal weights above (no short selling) and held without rebalancing."
    )

    bt_prices = prices[asset_cols].dropna(how='all').ffill()
    if len(bt_prices) < 2:
        st.warning("Not enough price observations for a backtest.")
    else:
        start_date = bt_prices.index[0]
        end_date = bt_prices.index[-1]
        n_periods = len(bt_prices) - 1
        years = (end_date - start_date).days / 365.25

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

        st.session_state["bt_investment"] = investment_amount
        st.session_state["bt_final"] = final_value
        st.session_state["bt_return"] = total_return
        st.session_state["bt_cagr"] = cagr
        st.session_state["bt_max_dd"] = max_drawdown
        st.session_state["bt_profit"] = profit

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Initial", f"{investment_amount:,.2f}")
        m2.metric("Final Value", f"{final_value:,.2f}", f"{profit:+,.2f}")
        m3.metric("Total Return", f"{total_return:.2%}")
        m4.metric("CAGR", f"{cagr:.2%}" if not np.isnan(cagr) else "n/a")
        m5.metric("Max Drawdown", f"{max_drawdown:.2%}")

        st.subheader("📈 Performance and Value Over Time")
        fig_bt, (ax_bt, ax_dd) = plt.subplots(2, 1, figsize=(12, 7),
                                              gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

        ax_bt.fill_between(portfolio_value.index, portfolio_value.values, investment_amount,
                           where=portfolio_value.values >= investment_amount,
                           color="green", alpha=0.15, label="Gain Area")
        ax_bt.fill_between(portfolio_value.index, portfolio_value.values, investment_amount,
                           where=portfolio_value.values < investment_amount,
                           color="red", alpha=0.15, label="Loss Area")
        ax_bt.plot(portfolio_value.index, portfolio_value.values, 'b-', linewidth=2, label="Optimized Portfolio")

        if benchmark_prices is not None:
            bench_aligned = benchmark_prices.reindex(portfolio_value.index).ffill().dropna()
            if len(bench_aligned) >= 2:
                bench_value = bench_aligned / bench_aligned.iloc[0] * investment_amount
                ax_bt.plot(bench_value.index, bench_value.values, 'orange', linestyle='--',
                           linewidth=1.5, label=f"Benchmark ({benchmark_col})")

        ax_bt.axhline(investment_amount, color='gray', linestyle=':', linewidth=1, label="Initial Investment")
        ax_bt.set_ylabel("Portfolio Value ($)")
        ax_bt.set_title("Portfolio Value Over Time — Line & Area Chart")
        ax_bt.legend(loc="upper left")
        ax_bt.grid(True, alpha=0.3)
        ax_bt.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        ax_dd.fill_between(drawdown.index, drawdown.values * 100, 0, color="crimson", alpha=0.4, step="pre")
        ax_dd.plot(drawdown.index, drawdown.values * 100, color="crimson", linewidth=1, drawstyle="steps")
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.set_xlabel("Date")
        ax_dd.grid(True, alpha=0.3)
        ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        save_chart(fig_bt, "Performance and Value Over Time", section="bt")
        st.pyplot(fig_bt)

        # Yearly Performance bar chart
        st.subheader("📅 Yearly Portfolio Returns")
        year_end_vals = portfolio_value.resample('YE').last()
        if len(year_end_vals) >= 2:
            yearly_returns = year_end_vals.pct_change().dropna() * 100
        else:
            total_ret = (portfolio_value.iloc[-1] / investment_amount - 1) * 100
            yearly_returns = pd.Series([total_ret], index=[portfolio_value.index[-1].year])
        fig_yearly, ax_yearly = plt.subplots(figsize=(12, 4.5))
        yearly_colors = ["#4CAF50" if r >= 0 else "#F44336" for r in yearly_returns]
        ax_yearly.bar(yearly_returns.index.astype(str), yearly_returns.values,
                      color=yearly_colors, edgecolor="white", linewidth=0.8, width=0.6)
        ax_yearly.axhline(0, color="gray", linewidth=0.8)
        ax_yearly.set_ylabel("Yearly Return (%)")
        ax_yearly.set_xlabel("Year")
        ax_yearly.set_title("Portfolio Yearly Returns")
        ax_yearly.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        avg_yearly = yearly_returns.values.mean()
        ax_yearly.axhline(avg_yearly, color="blue", linestyle="--", linewidth=1,
                          label=f"Avg: {avg_yearly:.2f}%")
        ax_yearly.legend(loc="upper right")
        ax_yearly.grid(True, alpha=0.3)
        save_chart(fig_yearly, "Yearly Portfolio Returns", section="bt")
        st.pyplot(fig_yearly)

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

        # Profit & Loss bar chart
        st.subheader("💰 Profit and Loss Graph")
        fig_pnl, ax_pnl = plt.subplots(figsize=(10, 4.5))
        pnl_plot = alloc_df.sort_values("Profit / Loss")
        pnl_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in pnl_plot["Profit / Loss"]]
        bars = ax_pnl.bar(pnl_plot["Asset"], pnl_plot["Profit / Loss"], color=pnl_colors, edgecolor="white", linewidth=0.8)
        ax_pnl.axhline(0, color="gray", linewidth=0.8)
        ax_pnl.set_ylabel("Profit / Loss ($)")
        ax_pnl.set_xlabel("Asset")
        ax_pnl.set_title("Per-Asset Profit & Loss")
        ax_pnl.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax_pnl.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, pnl_plot["Profit / Loss"]):
            ax_pnl.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"${val:+,.0f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
        save_chart(fig_pnl, "Profit and Loss Graph", section="bt")
        st.pyplot(fig_pnl)

        bt_csv = portfolio_value.to_frame("Portfolio Value").to_csv().encode('utf-8')
        st.download_button("Download Backtest Series (CSV)", bt_csv, "backtest_value.csv", "text/csv")

    # ------------------ Out-of-Sample Test ------------------
    st.header("🔮 Out-of-Sample Test")
    st.caption(
        "Upload a separate CSV (same format) to evaluate the optimal portfolio on unseen data. "
        "The weights computed above are applied without rebalancing."
    )
    test_file = st.file_uploader("Upload test CSV for out-of-sample evaluation", type=["csv"], key="test_csv")
    if test_file is not None:
        test_prices = load_prices(test_file)
        if test_prices is None:
            st.stop()
        test_assets = [c for c in asset_cols if c in test_prices.columns]
        missing = [c for c in asset_cols if c not in test_prices.columns]
        if missing:
            st.info(
                f"Test data is missing these assets (will be dropped & weights renormalized): {missing}"
            )
        if len(test_assets) < 2:
            st.warning("Fewer than 2 assets from the optimal portfolio are available in the test data.")
        else:
            test_weights = pd.Series(opt_weights_no_short, index=asset_cols).loc[test_assets].values
            test_weights = test_weights / test_weights.sum()
            test_prices = test_prices[test_assets].dropna(how='all').ffill()
            if len(test_prices) < 2:
                st.warning("Test data has fewer than 2 valid rows — cannot run evaluation.")
            else:
                test_normalized = test_prices / test_prices.iloc[0]
                weights_series = pd.Series(test_weights, index=test_assets)
                test_per_asset = test_normalized.multiply(weights_series, axis=1) * investment_amount
                test_portfolio_value = test_per_asset.sum(axis=1)

                test_start = test_prices.index[0]
                test_end = test_prices.index[-1]
                test_years = (test_end - test_start).days / 365.25
                test_final = test_portfolio_value.iloc[-1]
                test_total_return = test_final / investment_amount - 1
                test_cagr = (test_final / investment_amount) ** (1 / test_years) - 1 if test_years > 0 else np.nan
                test_running_max = test_portfolio_value.cummax()
                test_drawdown = test_portfolio_value / test_running_max - 1
                test_max_dd = test_drawdown.min()
                test_profit = test_final - investment_amount

                test_monthly_returns = test_portfolio_value.pct_change().dropna()
                test_ann_return = test_monthly_returns.mean() * 12
                test_ann_risk = test_monthly_returns.std() * np.sqrt(12)
                test_sharpe = (test_ann_return - risk_free_rate) / test_ann_risk if test_ann_risk > 0 else np.nan

                st.session_state["oos_investment"] = investment_amount
                st.session_state["oos_final"] = test_final
                st.session_state["oos_return"] = test_total_return
                st.session_state["oos_cagr"] = test_cagr
                st.session_state["oos_max_dd"] = test_max_dd
                st.session_state["oos_ann_return"] = test_ann_return
                st.session_state["oos_ann_risk"] = test_ann_risk
                st.session_state["oos_sharpe"] = test_sharpe
                st.session_state["oos_profit"] = test_profit

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Initial", f"{investment_amount:,.2f}")
                m2.metric("Final Value", f"{test_final:,.2f}", f"{test_profit:+,.2f}")
                m3.metric("Total Return", f"{test_total_return:.2%}")
                m4.metric("CAGR", f"{test_cagr:.2%}" if not np.isnan(test_cagr) else "n/a")
                m5.metric("Max Drawdown", f"{test_max_dd:.2%}")

                m6, m7, m8 = st.columns(3)
                m6.metric("Annual Return (OOS)", f"{test_ann_return:.2%}")
                m7.metric("Annual Volatility (OOS)", f"{test_ann_risk:.2%}")
                m8.metric(f"Sharpe (rf={risk_free_rate:.2%})", f"{test_sharpe:.3f}")

                # Portfolio value chart with drawdown
                st.subheader("📈 Out-of-Sample Performance")
                fig_oos, (ax_oos, ax_oos_dd) = plt.subplots(2, 1, figsize=(12, 7),
                                                            gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

                ax_oos.fill_between(test_portfolio_value.index, test_portfolio_value.values, investment_amount,
                                    where=test_portfolio_value.values >= investment_amount,
                                    color="green", alpha=0.15, label="Gain Area")
                ax_oos.fill_between(test_portfolio_value.index, test_portfolio_value.values, investment_amount,
                                    where=test_portfolio_value.values < investment_amount,
                                    color="red", alpha=0.15, label="Loss Area")
                ax_oos.plot(test_portfolio_value.index, test_portfolio_value.values, 'g-', linewidth=2,
                            label="Out-of-Sample Portfolio")
                ax_oos.axhline(investment_amount, color='gray', linestyle=':', linewidth=1,
                               label="Initial Investment")
                ax_oos.set_ylabel("Portfolio Value ($)")
                ax_oos.set_title("Out-of-Sample Portfolio Value")
                ax_oos.legend(loc="upper left")
                ax_oos.grid(True, alpha=0.3)
                ax_oos.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

                test_dd_series = test_drawdown * 100
                ax_oos_dd.fill_between(test_dd_series.index, test_dd_series.values, 0,
                                        color="crimson", alpha=0.4, step="pre")
                ax_oos_dd.plot(test_dd_series.index, test_dd_series.values, color="crimson",
                               linewidth=1, drawstyle="steps")
                ax_oos_dd.set_ylabel("Drawdown (%)")
                ax_oos_dd.set_xlabel("Date")
                ax_oos_dd.grid(True, alpha=0.3)
                ax_oos_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
                save_chart(fig_oos, "Out-of-Sample Portfolio Value", section="oos")
                st.pyplot(fig_oos)

                # Per-asset breakdown for test period
                st.subheader("🧾 Per-Asset Allocation — Out-of-Sample")
                test_alloc = pd.DataFrame({
                    "Asset": test_assets,
                    "Weight": test_weights,
                    "Initial Allocation": test_weights * investment_amount,
                    "Final Value": test_per_asset.iloc[-1].values,
                })
                test_alloc["Profit / Loss"] = test_alloc["Final Value"] - test_alloc["Initial Allocation"]
                test_alloc["Return %"] = test_alloc["Final Value"] / test_alloc["Initial Allocation"] - 1
                test_alloc = test_alloc[test_alloc["Weight"] > 0.001].sort_values("Weight", ascending=False)
                st.dataframe(
                    test_alloc.style.format({
                        "Weight": "{:.2%}",
                        "Initial Allocation": "{:,.2f}",
                        "Final Value": "{:,.2f}",
                        "Profit / Loss": "{:+,.2f}",
                        "Return %": "{:.2%}",
                    })
                )

                # P&L chart for test period
                st.subheader("💰 Profit and Loss — Out-of-Sample")
                fig_pnl_oos, ax_pnl_oos = plt.subplots(figsize=(10, 4.5))
                pnl_plot_oos = test_alloc.sort_values("Profit / Loss")
                pnl_colors_oos = ["#4CAF50" if v >= 0 else "#F44336" for v in pnl_plot_oos["Profit / Loss"]]
                bars_oos = ax_pnl_oos.bar(pnl_plot_oos["Asset"], pnl_plot_oos["Profit / Loss"],
                                          color=pnl_colors_oos, edgecolor="white", linewidth=0.8)
                ax_pnl_oos.axhline(0, color="gray", linewidth=0.8)
                ax_pnl_oos.set_ylabel("Profit / Loss ($)")
                ax_pnl_oos.set_xlabel("Asset")
                ax_pnl_oos.set_title("Per-Asset Profit & Loss — Out-of-Sample")
                ax_pnl_oos.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
                ax_pnl_oos.tick_params(axis="x", rotation=30)
                for bar, val in zip(bars_oos, pnl_plot_oos["Profit / Loss"]):
                    ax_pnl_oos.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                                    f"${val:+,.0f}", ha="center",
                                    va="bottom" if val >= 0 else "top", fontsize=8)
                save_chart(fig_pnl_oos, "Out-of-Sample Profit and Loss", section="oos")
                st.pyplot(fig_pnl_oos)

                test_csv = test_portfolio_value.to_frame("Portfolio Value").to_csv().encode('utf-8')
                st.download_button("Download Out-of-Sample Series (CSV)", test_csv,
                                   "out_of_sample_value.csv", "text/csv")

    # ------------------ Download weights ------------------
    st.subheader("📥 Download Results")
    csv_no_short = weights_df.to_csv(index=False).encode('utf-8')

    col_pdf, col_csv = st.columns(2)
    with col_csv:
        st.download_button("Download Optimal Weights (CSV)", csv_no_short, "optimal_weights.csv", "text/csv")
    with col_pdf:
        metrics = {
            "Expected Annual Return": f"{opt_return:.2%}",
            "Annual Volatility": f"{opt_risk:.2%}",
            f"Sharpe Ratio (rf={risk_free_rate:.2%})": f"{sharpe:.3f}",
            "Risk-Free Rate": f"{risk_free_rate:.2%}",
        }
        bt_metrics = {}
        if "bt_investment" in st.session_state:
            bt_metrics["Initial Investment"] = f"${st.session_state.bt_investment:,.2f}"
            bt_metrics["Final Value"] = f"${st.session_state.bt_final:,.2f}"
            bt_metrics["Total Return"] = f"{st.session_state.bt_return:.2%}"
            bt_metrics["CAGR"] = f"{st.session_state.bt_cagr:.2%}" if not np.isnan(st.session_state.bt_cagr) else "n/a"
            bt_metrics["Max Drawdown"] = f"{st.session_state.bt_max_dd:.2%}"
        oos_metrics = {}
        if "oos_investment" in st.session_state:
            oos_metrics["Initial Investment"] = f"${st.session_state.oos_investment:,.2f}"
            oos_metrics["Final Value"] = f"${st.session_state.oos_final:,.2f}"
            oos_metrics["Total Return"] = f"{st.session_state.oos_return:.2%}"
            oos_metrics["CAGR"] = f"{st.session_state.oos_cagr:.2%}" if not np.isnan(st.session_state.oos_cagr) else "n/a"
            oos_metrics["Max Drawdown"] = f"{st.session_state.oos_max_dd:.2%}"
            oos_metrics["Annual Return"] = f"{st.session_state.oos_ann_return:.2%}"
            oos_metrics["Annual Volatility"] = f"{st.session_state.oos_ann_risk:.2%}"
            oos_metrics["Sharpe Ratio"] = f"{st.session_state.oos_sharpe:.3f}"
        pdf_bytes = generate_pdf_report(weights_df, metrics, bt_metrics, oos_metrics)
        st.download_button(
            "📄 Download Report (PDF)",
            pdf_bytes,
            "portfolio_report.pdf",
            "application/pdf",
        )
    
    st.caption(f"Annualized statistics are derived from monthly returns (×12). Risk‑free rate = {risk_free_rate:.2%}.")