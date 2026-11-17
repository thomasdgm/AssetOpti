# AssetOpti — Multi-Asset Portfolio Optimization & Risk Management

**AssetOpti** is an interactive quantitative finance web application built with **Python**, **Streamlit**, **SciPy**, and **Plotly**. It provides end-to-end portfolio construction, tail-risk diagnostics, dynamic portfolio insurance backtesting, and forward-looking Monte Carlo simulations across $N$ financial assets.

---

## Key Features

### 1. Multi-Asset Markowitz Optimization & Capital Market Line (CML)
* **Automatic Data Ingestion & Alignment:** Parses US (`Date`, `Open`) and European (`Date`, `Ouv.`) Investing.com CSV exports, normalizes formatting, and aligns $N$ return series onto common monthly periods.
* **Mean-Variance Efficient Frontier:** Solves quadratic programming problems via `scipy.optimize.minimize` (**SLSQP**) to trace the Efficient Frontier with optional short-selling constraints.
* **Optimal Portfolios:** Computes exact weights for the **Global Minimum Variance (GMV)** portfolio and the **Maximum Sharpe Ratio (MSR / Tangency)** portfolio.
* **Correlation & Summary Analytics:** Interactive correlation matrix heatmap and standalone asset risk-return metrics.

### 2. Target Allocation along the Capital Market Line
* **Return or Volatility Targeting:** Determines the exact allocation between the risky Tangency portfolio and the risk-free asset ($R_f$) to match a user-defined annual return or volatility target (including leveraged positions when $w_{\text{risky}} > 100\%$).
* **Non-Normality & Tail-Risk Diagnostics:** Runs a **Jarque-Bera** test on historical returns and automatically switches between **Gaussian Parametric VaR** and skewness/kurtosis-adjusted **Cornish-Fisher VaR (99%, 1M)** alongside **Conditional VaR (CVaR / Expected Shortfall)** and **Maximum Drawdown**.

### 3. Dynamic Portfolio Insurance (CPPI) Backtesting
* **Standard CPPI & Profit Lock-In:** Simulates Constant Proportion Portfolio Insurance with configurable protected floor percentage, risk multiplier ($m$), rebalancing frequency, and optional peak-wealth ratcheting (*Profit Lock-in*).
* **Benchmark Comparison:** Evaluates CPPI performance and downside protection directly against a passive **100% Buy & Hold** strategy.

### 4. Forward-Looking Monte Carlo Simulations
* **Historical Bootstrap Engine:** Generates customizable multi-year forward wealth trajectories across both the **Tangency Portfolio (100% MSR)** and the **Target CML Portfolio**.
* **Distributional Fan Charts & Stress Metrics:** Visualizes median paths, 25%–75% interquartile confidence bands, 5% worst-case stress quantiles, and terminal **Floor Breach Rates**.

---

## Mathematical Framework

| Metric / Model | Formulation |
| :--- | :--- |
| **Portfolio Expected Return** | $\mu_p = \mathbf{w}^\top \boldsymbol{\mu}$ |
| **Portfolio Volatility** | $\sigma_p = \sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}$ |
| **Maximum Sharpe Ratio (MSR)** | $\max_{\mathbf{w}} \frac{\mathbf{w}^\top \boldsymbol{\mu} - R_f}{\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}} \quad \text{s.t. } \mathbf{w}^\top \mathbf{1} = 1$ |
| **CPPI Risky Exposure** | $E_t = \min\big(V_t,\; m \cdot \max(0,\, V_t - F_t)\big)$ |
| **Cornish-Fisher VaR** | $z_{\text{CF}} = z_\alpha + \frac{S}{6}(z_\alpha^2 - 1) + \frac{K}{24}(z_\alpha^3 - 3z_\alpha) - \frac{S^2}{36}(2z_\alpha^3 - 5z_\alpha)$ |

---

## Project Structure

```text
Site_Quant/
├── app.py              # Main Streamlit dashboard, custom CSS, and Plotly visualizations
├── function.py         # Quantitative finance engine (optimization, risk metrics, CPPI, Monte Carlo)
└── README.md           # Project documentation
```

## Installation & Usage

```text
git clone https://github.com/thomasdgm/AssetOpti.git
cd <nom-du-depot>
```

### Install dependencies

```text 
pip install -r requirements.txt
```

### Run the application

```text
streamlit run app.py 
```
