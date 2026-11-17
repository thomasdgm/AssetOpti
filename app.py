import pandas as pd               # Data manipulation and DataFrame structures
import numpy as np                # Numerical arrays and vectorized math operations
import matplotlib.pyplot as plt   # Matplotlib backend (kept for compatibility if needed)
import function as fct            # Custom quantitative finance calculation and plotting module
import streamlit as st            # Streamlit web application framework
import plotly.express as px       # High-level Plotly charts (area chart, correlation heatmap)
import plotly.graph_objects as go # Low-level Plotly figures for multi-trace financial charts

# Configure the browser tab title, wide-screen layout, and expanded sidebar on initial load
st.set_page_config(
    page_title="Optimisation et Gestion du Risque",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS to style KPI metric cards (dark translucent background, subtle border, rounded corners) and page padding
st.markdown("""
    <style>
        div[data-testid="stMetric"] {
            background-color: rgba(28, 31, 38, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 16px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem;
            color: #9ca3af;
            font-weight: 500;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
            font-weight: 700;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar : market data upload & gloabl parameters

with st.sidebar:
    # Render sidebar branding header and subtitle
    st.title("AssetOpti")
    st.caption("Portfolio Allocation, CPPI & Monte Carlo Simulations")
    st.divider()

    # File uploader widget allowing the user to select multiple historical CSV files
    st.subheader("Market Data")
    uploaded_files = st.file_uploader(
        "Upload historical series (2 or more CSVs)",
        type=["csv"],
        accept_multiple_files=True,
        help="Supported formats: Investing.com CSV exports containing 'Data' and 'Open' or 'Ouv.'."
    )

    # Global financial parameters shared across all four analytical tabs
    st.subheader("Global Parameters")
    # Annual risk-free rate input in percentage, then converted to decimal
    rf_pct = st.number_input("Risk-Free Rate Rf (%)", min_value=0.0, max_value=20.0, value=3.0, step=0.25)
    risk_free_rate = rf_pct / 100

    # Starting capital in euros used for historical backtests and Monte Carlo projections
    initial_wealth = st.number_input("Initial Capital (€)", min_value=0, max_value=10_000_000, value=10_000, step=500)
    # Toggle enabling or disabling short-selling (negative weights) in Markowitz optimization
    allow_short = st.toggle("Allow Short Selling", value=False, help="Allows negative asset weights in Markowitz optimization.")

# Render main application title
st.title("Quantitative Analysis & Multi-Asset Portfolio Optimization")

# Guardrail 1: Halt execution until at least 2 CSV files are uploaded
if not uploaded_files or len(uploaded_files) < 2:
    st.error("Start by uploading at least 2 CSV files in the sidebar to run the analysis.")
    st.stop()

# Parse uploaded CSV files into cleaned periodic return Series and extract asset names
files_data, asset_names = fct.process_uploaded_data(uploaded_files)

# Guardrail 2: Halt execution if fewer than 2 files had a valid Investing.com column structure
if len(files_data)<2:
    st.error("Fewer than two valid files could be parsed. Please check the column format ('Date' et 'Open' ou 'Ouv.').")
    st.stop()

# Align all valid return series onto common monthly dates via inner join
returns_df = fct.align_multiple_assets(files_data, asset_names)

# Guardrail 3: Require at least 12 overlapping monthly observations for reliable annualization
if returns_df.shape[0] < 12:
    st.error("Insufficient overlapping dates across uploaded assets (fewer than 12 common periods).")
    st.stop()

# Compute annualized geometric returns vector (mu), annualized volatilities, covariance matrix (Sigma), and correlation matrix
expected_returns = returns_df.apply(fct.annualize_returns)
individual_vols = returns_df.apply(fct.annualize_vol)
cov_matrix = returns_df.cov() * 12.0
corr_matrix = returns_df.corr()

# Compute 50-point Markowitz Efficient Frontier DataFrame across all N assets
eff_front = fct.efficient_frontier_matrix(expected_returns, cov_matrix, risk_free_rate=risk_free_rate, n_points=50, allow_short=allow_short)

# Compute Global Minimum Variance (GMV) portfolio weights, return, and volatility
w_gmv = fct.gmv_portfolio(cov_matrix, allow_short=allow_short)
gmv_ret = fct.portfolio_return_matrix(w_gmv, expected_returns)
gmv_vol = fct.portfolio_vol_matrix(w_gmv, cov_matrix)

# Compute Maximum Sharpe Ratio (Tangency / MSR) portfolio weights, return, volatility, and Sharpe ratio
w_msr = fct.msr_portfolio(expected_returns, cov_matrix, risk_free_rate=risk_free_rate, allow_short=allow_short)
tangency_ret = fct.portfolio_return_matrix(w_msr, expected_returns)
tangency_vol = fct.portfolio_vol_matrix(w_msr, cov_matrix)
max_sr_val = fct.sharpe_ratio(tangency_ret, tangency_vol, risk_free_rate=risk_free_rate)

# Determine maximum X-axis volatility bound and compute Capital Market Line (CML) coordinates
max_plot_vol = max(float(eff_front["vol"].max()), float(individual_vols.max()))
cml_x, cml_y = fct.capital_market_line_matrix(tangency_ret, tangency_vol, max_plot_vol, risk_free_rate=risk_free_rate)

# Reconstruct 1D historical monthly return series of the 100% Tangency portfolio (R * w_msr)
tangency_returns_hist = fct.portfolio_historical_returns_matrix(returns_df, w_msr, risky_weight=1.0, rf_weight=0.0, risk_free_rate=risk_free_rate)

# Create the four top-level navigation tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Efficient Frontier & CML",
    "Target Allocation (Tangency + Rf)",
    "Portfolio Insurance (CPPI)",
    "Monte Carlo Simulations"
])

### TAB1 : Markowitz efficient frontier & capital market line

with tab1:
    # Section header displaying the count of analyzed assets
    st.subheader(f"Tangency Portfolio (Maximum Sharpe Ratio -{len(asset_names)} Assets)")

    # Top row of 4 KPI cards summarizing Tangency and GMV metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tangency Return", f"{tangency_ret * 100:.2f}%")
    k2.metric("Tangency Volatility", f"{tangency_vol * 100:.2f} %")
    k3.metric("Maximum Sharpe Ratio", f"{max_sr_val:.2f}")
    k4.metric("GMV Volatility (Min Risk)", f"{gmv_vol * 100:.2f} %")

    st.markdown("")
    # Two-column layout: wide column (2.2) for Efficient Frontier chart, narrow column (1.1) for weights & normality test
    col_graph, col_alloc = st.columns([2.2, 1.1])

    with col_graph:
        # Format multi-line HTML hover tooltips listing exact asset weights at every point on the frontier
        hover_weights = [
            "<br>".join([f"{name}: {w * 100:.1f}%" for name, w in zip(asset_names, weights_arr)])
            for weights_arr in eff_front["weights"]
        ]
        # Initialize Plotly figure for the risk-return space
        fig_ef = go.Figure()

        # Trace 1: Markowitz Efficient Frontier curve with dynamic weight breakdown on hover
        fig_ef.add_trace(go.Scatter(
            x=eff_front["vol"],
            y=eff_front["return"],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color="#38bdf8", width=3),
            text=hover_weights,
            hovertemplate=(
                "Volatility: %{x:.2%}<br>"
                "Return: %{y:.2%}<br>"
                "%{text}<extra></extra>"
            )
        ))

        # Trace 2: Capital Market Line (CML) connecting Rf to the Tangency portfolio and beyond
        fig_ef.add_trace(go.Scatter(
            x=cml_x,
            y=cml_y,
            mode="lines",
            name="Capital Market Line (CML)",
            line=dict(color="#f59e0b", width=2, dash="dash")
        ))

        # Trace 3: Scatter markers with text labels for each individual asset
        fig_ef.add_trace(go.Scatter(
            x=individual_vols.values,
            y=expected_returns.values,
            mode="markers+text",
            name="Individual Assets",
            text=asset_names,
            textposition="top center",
            marker=dict(size=10, color="#a78bfa", line=dict(width=1, color="white"))
        ))

        # Trace 4: Diamond marker highlighting the Global Minimum Variance (GMV) portfolio
        fig_ef.add_traces(go.Scatter(
            x=[gmv_vol],
            y=[gmv_ret],
            mode='markers',
            name="GMV Portfolio",
            marker=dict(size=12, color='#10b981', symbol="diamond", line=dict(width=1.5, color='white'))
        ))

        # Trace 5: Red circle marker highlighting the Tangency (Maximum Sharpe Ratio) portfolio
        fig_ef.add_trace(go.Scatter(
            x=[tangency_vol],
            y=[tangency_ret],
            mode="markers",
            name="Tangency Portfolio (MSR)",
            marker=dict(size=14, color="#ef4444", line=dict(width=1.5, color="white"))
        ))

        # Apply unified dark theme and percentage axis formatting
        fig_ef = fct.style_plotly(
                fig_ef,
                title="Markowitz Efficient Frontier & Capital Market Line",
                x_title="Annualized Volatility",
                y_title="Annualized Return",
                x_format=".1%",
                y_format=".1%"
            )

        # Use 'closest' hovermode so individual scatter points and frontier weights display cleanly
        fig_ef.update_layout(hovermode="closest")
        # Anchor X-axis at 0% volatility up to 22% past the riskiest asset
        fig_ef.update_xaxes(range=[0, max_plot_vol * 1.22])
        st.plotly_chart(fig_ef, use_container_width=True)

    with col_alloc:
        # Sub-section comparing optimal weights between Tangency (MSR) and GMV portfolios
        st.markdown("#### Optimal Weights (Tangency vs. GMV)")
        # Build DataFrame of percentage weights sorted ascending by Tangency weight for horizontal bar ordering
        df_weights = pd.DataFrame({
            "Asset": asset_names,
            "Tangency (%)": w_msr * 100,
            "GMV (%)": w_gmv * 100
            }).sort_values("Tangency (%)", ascending=True)

        # Build grouped horizontal bar chart comparing MSR and GMV allocations
        fig_weights = go.Figure()
        fig_weights.add_trace(go.Bar(
            y=df_weights["Asset"],
            x=df_weights["Tangency (%)"],
            name="Tangency (MSR)",
            orientation="h",
            marker_color="#38bdf8"
        ))
        fig_weights.add_trace(go.Bar(
            y=df_weights["Asset"],
            x=df_weights["GMV (%)"],
            name="GMV",
            orientation="h",
            marker_color="#10b981"
        ))
        fig_weights = fct.style_plotly(
            fig_weights,
            title="Portfolio Allocation (%)",
            x_title="Weight (%)",
            y_title=""
        )
        fig_weights.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=70, b=20))
        st.plotly_chart(fig_weights, use_container_width=True)

        # Run Jarque-Bera normality test on Tangency portfolio historical returns
        jb_stat, jb_val, is_normal = fct.jarque_bera(tangency_returns_hist)
        st.markdown("#### Normality Test (Jarque-Bera)")
        # Display green badge if Gaussian, or amber warning if fat-tailed (triggering Cornish-Fisher VaR)
        if is_normal:
            st.success(f"**Normal distribution** (JB = {jb_stat:.2f}, p = {jb_val:.4f})")
        else:
            st.warning(f"**Fat tails / Non-normal** (JB = {jb_stat:.2f}, p = {jb_val:.4f}) — Cornish-Fisher VaR is used.")

# Horizontal visual divider below Tab 1 top panels
st.divider()
# Bottom two-column section: Individual asset metrics table on the left, correlation heatmap on the right
col_sum, col_corr = st.columns([1.2, 1])

with col_sum:
    st.markdown("#### Individual Assets Summary")
    # Build formatted summary table with standalone metrics and optimal weights for every asset
    df_assets_summary = pd.DataFrame({
        "Asset": asset_names,
        "Annualized Return": [f"{r * 100:.2f} %" for r in expected_returns],
        "Annualized Volatility": [f"{v * 100:.2f} %" for v in individual_vols],
        "Sharpe Ratio": [f"{fct.sharpe_ratio(r, v, risk_free_rate):.2f}" for r, v in zip(expected_returns, individual_vols)],
        "Tangency Weight": [f"{w * 100:.1f} %" for w in w_msr],
        "GMV Weight": [f"{w * 100:.1f} %" for w in w_gmv]
    })
    st.dataframe(df_assets_summary, use_container_width=True, hide_index=True)

with col_corr:
    st.markdown("#### Correlation Matrix")
    # Render annotated diverging heatmap (-1 to +1) of pairwise asset correlations
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto"
    )
    fig_corr.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
        height=280
    )
    st.plotly_chart(fig_corr, use_container_width=True)

### TAB2 : Target allocation along the capital market line

with tab2:
    st.subheader("Portfolio Construction along the Capital Market Line")

    # Control row: choose whether to target an annual Return or Volatility, and enter target percentage
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        type_ind = st.radio(
            "Targeting Objective",
            ["Return", "Volatility"],
            horizontal=True
        )
    with col_ctrl2:
        # Default input value matches the unleveraged Tangency portfolio metric
        default_val = round(tangency_ret * 100, 1) if type_ind == "Return" else round(tangency_vol * 100, 1)
        target_val_pct = st.number_input(
            f"Target Annual {type_ind} (%)",
            min_value=0.1,
            max_value=200.0,
            value=float(default_val),
            step=0.5
        )

    # Convert percentage target to decimal
    target_val = target_val_pct / 100.0

    # Solve for the optimal weight allocated to the Tangency risky portfolio along the CML
    if type_ind == "Return":
        risky_weight = fct.target_return_weight(target_val, tangency_ret, risk_free_rate=risk_free_rate)
    else:
        risky_weight = fct.target_volatility_weight(target_val, tangency_vol)

    # Remaining weight is invested in (or borrowed at) the risk-free rate
    rf_weight = 1.0 - risky_weight

    # Notify the user when the target requires borrowing at Rf (leverage > 1.0x)
    if risky_weight > 1.0:
        st.info(f"This target requires a **leverage** of **{risky_weight:.2f}x** (borrowing {abs(rf_weight) * 100:.1f} % at the risk-free rate).")

    # Reconstruct historical returns and cumulative euro wealth path of the target CML portfolio
    target_portf_returns = fct.portfolio_historical_returns_matrix(returns_df, w_msr, risky_weight=risky_weight, rf_weight=rf_weight, risk_free_rate=risk_free_rate)
    target_portf_wealth = fct.compute_wealth_index(target_portf_returns)
    target_portf_wealth["Value (€)"] = target_portf_wealth["Price"] * (initial_wealth / 100.0)

    # Compute theoretical annualized volatility, return, and full risk metrics table
    target_vol = fct.portfolio_vol(tangency_vol, 0.0, risky_weight, rf_weight, 0.0)
    target_ret = fct.portfolio_return(tangency_ret, risk_free_rate, risky_weight, rf_weight)
    target_stats = fct.compute_risk_table(target_portf_returns, risk_free_rate, is_normal=is_normal)

    # Display 4 KPI cards summarizing weights and theoretical metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risky Asset Weight", f"{risky_weight * 100:.1f} %")
    c2.metric("Risk-Free Asset Weight", f"{rf_weight * 100:.1f} %")
    c3.metric("Theoretical Return", f"{target_ret * 100:.2f} %")
    c4.metric("Theoretical Volatility", f"{target_vol * 100:.2f} %")

    # Two-column layout: historical cumulative performance area chart on left, risk summary table on right
    col_g2, col_t2 = st.columns([2.2, 1])
    with col_g2:
        fig_target = px.area(target_portf_wealth, x="Date", y="Value (€)")
        fig_target.update_traces(line_color="#10b981", fillcolor="rgba(16, 185, 129, 0.15)")
        fig_target = fct.style_plotly(
            fig_target,
            title=f"Historical Performance of Target Portfolio (Initial {initial_wealth:,.0f} €)",
            x_title="Date",
            y_title="Portfolio Value (€)"
        )
        st.plotly_chart(fig_target, use_container_width=True)

    with col_t2:
        st.markdown("#### Historical Risk Metrics")
        # Format risk dictionary into a clean two-column DataFrame
        df_target_metrics = pd.DataFrame({
            "Metric": list(target_stats.keys()),
            "Value": [
                f"{v:.2f}" if "Sharpe" in k else f"{v * 100:.2f} %"
                for k, v in target_stats.items()
            ]
        })
        st.dataframe(df_target_metrics, use_container_width=True, hide_index=True)

### TAB3 : Historical CPPI portfolio insurance backtest

with tab3:
    st.subheader("Historical Backtest : CPPI vs. Buy & Hold")

    # Four interactive sliders/toggles controlling CPPI protection parameters
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        floor_pct = st.slider("Guaranteed Floor (%)", min_value=50, max_value=95, value=80, step=1)
    with s2:
        multiplier = st.slider("Multiplier (m)", min_value=1, max_value=10, value=3, step=1)
    with s3:
        rebalance_freq = st.slider("Rebalancing Frequency (months)", min_value=1, max_value=12, value=3, step=1)
    with s4:
        use_lockin = st.toggle("Enable Profit Lock-in", value=False, help="Ratchets up the guaranteed floor based on historical peak wealth.")

    # Convert percentage floor to decimal ratio
    floor = floor_pct / 100.0

    # Run either ratcheting profit lock-in CPPI or standard CPPI on the Tab 2 target portfolio returns
    if use_lockin:
        cppi_df = fct.run_cppi_lockin(target_portf_returns, initial_wealth=initial_wealth, floor=floor, multiplier=multiplier, risk_free_rate=risk_free_rate, rebalance_freq=rebalance_freq)
    else:
        cppi_df = fct.run_cppi(target_portf_returns, initial_wealth=initial_wealth, floor=floor, multiplier=multiplier, risk_free_rate=risk_free_rate, rebalance_freq=rebalance_freq)

    # Run passive 100% Buy & Hold benchmark on the same underlying return series
    buy_hold_df = fct.run_buy_and_hold(target_portf_returns, initial_wealth=initial_wealth)

    # Build multi-line Plotly chart comparing Buy & Hold, CPPI Wealth, and the Guaranteed Floor
    fig_cppi = go.Figure()
    fig_cppi.add_trace(go.Scatter(
        x=buy_hold_df.index, y=buy_hold_df["Wealth"],
        mode="lines", name="Buy & Hold",
        line=dict(color="#94a3b8", width=2)
    ))
    fig_cppi.add_trace(go.Scatter(
        x=cppi_df.index, y=cppi_df["CPPI Wealth"],
        mode="lines", name="CPPI (Lock-in)" if use_lockin else "CPPI Standard",
        line=dict(color="#38bdf8", width=2.8)
    ))
    fig_cppi.add_trace(go.Scatter(
        x=cppi_df.index, y=cppi_df["Floor"],
        mode="lines", name="Floor",
        line=dict(color="#ef4444", width=2, dash="dot")
    ))
    fig_cppi = fct.style_plotly(
        fig_cppi,
        title="Historical Backtest: CPPI Strategy vs. Buy and Hold",
        x_title="Date",
        y_title="Portfolio Value (€)"
    )
    st.plotly_chart(fig_cppi, use_container_width=True)

    # Compute side-by-side performance and tail-risk metrics for CPPI vs. Buy & Hold
    stats_cppi = fct.compute_risk_table(cppi_df["CPPI Wealth"].pct_change(), risk_free_rate=risk_free_rate, is_normal=is_normal)
    stats_bh = fct.compute_risk_table(buy_hold_df["Wealth"].pct_change(), risk_free_rate=risk_free_rate, is_normal=is_normal)

    # Display formatted side-by-side comparison table
    df_comp_cppi = pd.DataFrame({
        "Metric": list(stats_cppi.keys()),
        "CPPI Strategy": [f"{v:.2f}" if "Sharpe" in k else f"{v * 100:.2f} %" for k, v in stats_cppi.items()],
        "Buy & Hold": [f"{v:.2f}" if "Sharpe" in k else f"{v * 100:.2f} %" for k, v in stats_bh.items()],
    })
    st.dataframe(df_comp_cppi, use_container_width=True, hide_index=True)        

### TAB4 : Forward-looking Monte Carlo simulations

with tab4:
    st.subheader("Forward-Looking Monte Carlo Projections (Historical Bootstrap)")

    # Radio selector letting the user simulate either the 100% Tangency portfolio or the Tab 2 Target portfolio
    mc_source = st.radio(
        "Risky Underlying Engine",
        ["Tangency Portfolio (100% MSR)", "Target Portfolio (Tab 2 Allocation)"],
        horizontal=True
    )
    # Select the corresponding historical return series as the bootstrap sampling pool
    mc_base_returns = tangency_returns_hist if "Tangency" in mc_source else target_portf_returns

    # Three sliders controlling simulation count, time-step frequency, and projection horizon
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        n_scenarios = st.slider("Number of Scenarios", min_value=50, max_value=1000, value=200, step=50)
    with mc2:
        steps_per_year = st.slider("Steps per Year (12 = monthly)", min_value=1, max_value=52, value=12, step=1)
    with mc3:
        n_years = st.slider("Horizon (Years)", min_value=1, max_value=40, value=10, step=1)

    # Simulate vectorized forward CPPI wealth and floor paths via historical bootstrap
    mc_cppi_wealth, mc_cppi_floor = fct.run_cppi_monte_carlo(
        mc_base_returns,
        initial_wealth=initial_wealth,
        n_scenarios=n_scenarios,
        steps_per_year=steps_per_year,
        n_years=n_years,
        floor=floor,
        multiplier=multiplier,
        risk_free_rate=risk_free_rate,
        rebalance_freq=rebalance_freq
    )
    # Simulate forward Buy & Hold paths via historical bootstrap and align date index
    mc_bh_wealth = fct.run_monte_carlo_bootstrap(
        mc_base_returns,
        n_scenarios=n_scenarios,
        steps_per_year=steps_per_year,
        n_years=n_years,
        initial_wealth=initial_wealth
    )
    mc_bh_wealth.index = mc_cppi_wealth.index

    # Compute cross-sectional quantiles (median, 25%-75% IQR, and 5% stress tail) at each future date
    cppi_med = mc_cppi_wealth.median(axis=1)
    cppi_p25 = mc_cppi_wealth.quantile(0.25, axis=1)
    cppi_p75 = mc_cppi_wealth.quantile(0.75, axis=1)
    cppi_p05 = mc_cppi_wealth.quantile(0.05, axis=1)
    bh_med = mc_bh_wealth.median(axis=1)
    bh_p05 = mc_bh_wealth.quantile(0.05, axis=1)

    # Build fan chart displaying confidence bands, median paths, 5% stress quantiles, and guaranteed floor
    fig_mc = go.Figure()

    # Upper invisible boundary of the 25%-75% interquartile band
    fig_mc.add_trace(go.Scatter(
        x=mc_cppi_wealth.index, y=cppi_p75,
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    # Lower boundary filling upward to p75 to create a shaded interquartile fan
    fig_mc.add_trace(go.Scatter(
        x=mc_cppi_wealth.index, y=cppi_p25,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(56, 189, 248, 0.18)",
        name="CPPI (25% - 75% IQR)"
    ))

    # Median trajectories for CPPI and Buy & Hold
    fig_mc.add_trace(go.Scatter(
        x=mc_cppi_wealth.index, y=cppi_med,
        mode="lines", name="CPPI (Median)",
        line=dict(color="#38bdf8", width=3)
    ))
    fig_mc.add_trace(go.Scatter(
        x=mc_bh_wealth.index, y=bh_med,
        mode="lines", name="Buy & Hold (Median)",
        line=dict(color="#f59e0b", width=2.5)
    ))

    # 5% worst-case stress trajectories for CPPI and Buy & Hold
    fig_mc.add_trace(go.Scatter(
        x=mc_cppi_wealth.index, y=cppi_p05,
        mode="lines", name="CPPI (5% Quantile - Stress)",
        line=dict(color="#10b981", width=1.8, dash="dash")
    ))
    fig_mc.add_trace(go.Scatter(
        x=mc_bh_wealth.index, y=bh_p05,
        mode="lines", name="Buy & Hold (5% Quantile - Stress)",
        line=dict(color="#a855f7", width=1.8, dash="dash")
    ))

    # Deterministic guaranteed floor trajectory across the projection horizon
    fig_mc.add_trace(go.Scatter(
        x=mc_cppi_floor.index, y=mc_cppi_floor.iloc[:, 0],
        mode="lines", name="Guaranteed Floor",
        line=dict(color="#ef4444", width=2, dash="dot")
    ))

    fig_mc = fct.style_plotly(
        fig_mc,
        title=f"Monte Carlo Projections ({n_scenarios} paths over {n_years} years)",
        x_title="Projected Date",
        y_title="Projected Wealth (€)"
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # Extract terminal wealth distributions at the final horizon date and compute floor breach rate
    final_cppi = mc_cppi_wealth.iloc[-1]
    final_bh = mc_bh_wealth.iloc[-1]
    final_floor = mc_cppi_floor.iloc[-1, 0]
    breach_rate = (final_cppi < final_floor).mean() * 100

    # Display 4 terminal summary KPI cards across all simulated scenarios
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Median Terminal Wealth (CPPI)", f"{final_cppi.median():,.0f} €")
    m2.metric("Median Terminal Wealth (B&H)", f"{final_bh.median():,.0f} €")
    m3.metric("Worst 5% Terminal (CPPI vs B&H)", f"{final_cppi.quantile(0.05):,.0f} €", delta=f"{final_cppi.quantile(0.05) - final_bh.quantile(0.05):+,.0f} € vs B&H")
    m4.metric("Floor Breach Rate", f"{breach_rate:.1f} %")