import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize
import plotly.graph_objects as go

def process_uploaded_data(uploaded_files):
    """
    Processes a list of uploaded CSV files and extracts valid historical return series.
    Args : 
        uploaded_files (list) : List of Streamlit UploadedFile objects

    Returns : 
        list[pd.Series]: List of cleaned historical return Series for each valid file
    """
    # Initialize an empty list to store parsed return series
    dfs = []
    names = []
    # Loop through each file uploaded via the Streamlit interface
    for file in uploaded_files:
        # Extract and clean the historical returns for the current file
        df = get_hist_data(file)
        # Append the series only if the file format was valid and successfully parsed
        if df is not None:
            dfs.append(df)
            # Strip the file extension (e.g., '.csv') to keep a clean asset label
            names.append(file.name.rsplit(".", 1)[0])
    # Return the list of all valid return series
    return dfs, names

def get_hist_data(filepath):
    """
    Reads an Investing.com CSV file, cleans prices, and computes returns
    Args : 
        filepath (UploadedFile or str): Path or buffer pointing to the CSV file

    Returns : 
        pd.Series or None: Chronologically sorted percentage return Series, or None if format is invalid
    """
    # Reset the file pointer to the beginning of the stream before reading headers
    filepath.seek(0)
    # Read only the first row to inspect available column names
    colonnes = pd.read_csv(filepath, nrows=0).columns
    # Reset the file pointer again before loading the full dataset
    filepath.seek(0)

    #Check for US format ('Date' and 'Open' columns)
    if len(colonnes)>2  and colonnes[2] == "Open":
        # Load only the Date (as index) and Open price columns
        data = pd.read_csv(filepath, index_col=0, usecols=['Date','Open'])
        # Standardize the price column name
        data.columns = ['Price']
        # Clean formatting characters and convert prices to numeric floats
        data['Price'] = clean_price(data['Price'])
        # Remove any rows with missing or unparseable values
        data = data.dropna()
        # Parse index strings into datetime objects using MM/DD/YYYY format
        data.index = pd.to_datetime(data.index, format='%m/%d/%Y')
        # Sort observations chronologically from oldest to newest
        data = data.sort_index(ascending=True)
        # Compute simple periodic percentage returns, drop the first NaN, and return a 1D Series
        return data['Price'].pct_change().dropna().squeeze()
    # Check for French/European format ('Date' and 'Ouv.' columns)
    elif len(colonnes)>2 and colonnes[2] == "Ouv.":
        # Load only the Date (as index) and Ouv. (Open) price columns
        data = pd.read_csv(filepath, index_col=0, usecols=['Date','Ouv.'])
        # Standardize the price column name
        data.columns = ['Price']
        # Clean formatting characters and convert prices to numeric floats
        data['Price'] = clean_price(data['Price'])
        # Remove any rows with missing or unparseable values
        data = data.dropna()
        # Parse index strings into datetime objects using DD/MM/YYYY format
        data.index = pd.to_datetime(data.index, format='%d/%m/%Y')
        # Sort observations chronologically from oldest to newest
        data = data.sort_index(ascending=True)
        # Compute simple periodic percentage returns, drop the first NaN, and return a 1D Series
        return data['Price'].pct_change().dropna().squeeze()
    # Handle unrecognized CSV column structures
    else:
        # Display a Streamlit warning showing the rejected file name and its columns
        st.warning(f"Rejected file : {filepath.name} | Columns found : {list(colonnes)}")
        # Return None to exclude this file from downstream calculations
        return None

def clean_price(serie):
    """
    Cleans raw price strings (spaces, non-breaking spaces, US/EU separators) into numeric floats.

    Args:
        serie (pd.Series): Raw string or mixed-type series of asset prices.

    Returns:
        pd.Series: Numeric float64 series with invalid strings coerced to NaN.
    """
    # Convert values to string and remove non-breaking spaces (\xa0) and regular spaces
    s = serie.astype(str).str.replace('\xa0','',regex=False).str.replace(' ','',regex=False)
    # If both commas and dots exist, treat dots as thousands separators and commas as decimals
    if s.str.contains(',').any() and s.str.contains(r'\.', regex=True).any():
        s = s.str.replace('.','', regex=False).str.replace(',','.',regex=False)
    else:
        # Otherwise simply replace decimal commas with dots
        s = s.str.replace(',','.',regex=False)
    # Convert cleaned strings to numeric floats, turning unparseable entries into NaN
    return pd.to_numeric(s, errors='coerce')

def align_data(series_a, series_b):
    """
    Aligns two return series to equal length by trimming older observations from the longer series.

    Args:
        series_a (pd.Series): First asset return series.
        series_b (pd.Series): Second asset return series.

    Returns:
        tuple[pd.Series, pd.Series]: Two return series of identical length.
    """
    # Check if the two series have different lengths
    if series_a.shape != series_b.shape:
        # Compute the difference in number of observations
        diff = series_a.shape[0] - series_b.shape[0]
        # If series A is longer, keep only its most recent observations matching B's length
        if diff > 0 :
            n_min = series_b.shape[0]
            series_a = series_a.iloc[-n_min:]
        # If series B is longer, keep only its most recent observations matching A's length
        elif diff < 0:
            n_min = series_a.shape[0]
            series_b = series_b.iloc[-n_min:]
    # Return the equal-length pair
    return series_a, series_b

def align_multiple_assets(returns_list, asset_names):
    """
    Aligns N return Series onto common monthly periods via an inner join into a single DataFrame.

    Args:
        returns_list (list[pd.Series] or pd.Series): List of individual asset return Series.
        asset_names (list[str]): Names corresponding to each asset Series.

    Returns:
        pd.DataFrame: Aligned DataFrame of periodic returns (rows = dates, columns = assets).
    """
    # Wrap a single Series into a list to prevent iteration errors
    if isinstance(returns_list, pd.Series):
        returns_list = [returns_list]
    # Prepare a list to hold monthly-normalized series
    cleaned_series = []
    # Loop over each raw return series
    for s in returns_list:
        # Work on a copy to avoid mutating the original series
        s_copy = s.copy()
        # Normalize index to Year-Month periods so start-of-month and end-of-month dates match
        s_copy.index = pd.to_datetime(s_copy.index).to_period("M")
        # Keep only the last observation if duplicate months appear in the same file
        s_copy = s_copy[~s_copy.index.duplicated(keep="last")]
        # Store the normalized series
        cleaned_series.append(s_copy)
    # Concatenate all series column-wise on overlapping dates (inner join) and drop missing rows
    df = pd.concat(cleaned_series, axis=1, join='inner').dropna()
    # Convert PeriodIndex back to normalized end-of-month DatetimeIndex for Plotly compatibility
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp(how="end").normalize()
    # Rename DataFrame columns using the provided asset names
    df.columns = asset_names[:df.shape[1]]
    # Return the aligned multi-asset return DataFrame
    return df

def compute_wealth_index(returns):
    """
    Computes a cumulative wealth index starting from a base value of 100.

    Args:
        returns (pd.Series): Periodic asset or portfolio returns.

    Returns:
        pd.DataFrame: DataFrame containing 'Date' and cumulative 'Price' columns.
    """
    # Compound periodic returns starting at 100
    wealth = 100.0 * (1 + returns).cumprod()
    # Package dates and wealth levels into a two-column DataFrame
    return pd.DataFrame({'Date': returns.index, 'Price': wealth.values})

def prices_to_returns(prices):
    """
    Converts a series or DataFrame of prices into periodic percentage returns.

    Args:
        prices (pd.Series or pd.DataFrame): Historical asset prices.

    Returns:
        pd.Series or pd.DataFrame: Periodic percentage returns with the initial NaN removed.
    """
    # Compute period-over-period percentage change and drop the first NaN row
    return prices.pct_change().dropna()

def annualize_returns(returns, periods_per_year = 12):
    """
    Computes the compound annualized geometric return (CAGR) from periodic returns.

    Args:
        returns (pd.Series): Periodic return series.
        periods_per_year (int, optional): Number of return periods per year (12 = monthly). Defaults to 12.

    Returns:
        float: Annualized geometric return.
    """
    # Compute total compounded growth across all periods
    prod_recompunded_growth = (1 + returns).prod()
    # Count total number of return observations
    n_periods = returns.shape[0]
    # Annualize cumulative growth and cast to Python float
    return float(prod_recompunded_growth**(periods_per_year / n_periods) - 1)

def annualize_vol(returns, periods_per_year = 12):
    """
    Computes the annualized sample volatility from periodic returns.

    Args:
        returns (pd.Series): Periodic return series.
        periods_per_year (int, optional): Number of return periods per year (12 = monthly). Defaults to 12.

    Returns:
        float: Annualized volatility.
    """
    # Multiply sample standard deviation (ddof=1) by the square root of periods per year
    return float(returns.std(ddof=1) * (periods_per_year**0.5))

def correlation(series_a, series_b):
    """
    Computes the Pearson linear correlation coefficient between two return series.

    Args:
        series_a (pd.Series): First asset return series.
        series_b (pd.Series): Second asset return series.

    Returns:
        float: Pearson correlation coefficient between -1 and 1.
    """
    # Align both series to identical length
    series_a, series_b = align_data(series_a, series_b)
    # Compute Pearson correlation coefficient (discarding the p-value)
    corr, _ = stats.pearsonr(series_a, series_b)
    # Return correlation scalar
    return corr

def covariance(vol_a, vol_b, corr):
    """
    Computes covariance between two assets from their volatilities and correlation.

    Args:
        vol_a (float): Volatility of asset A.
        vol_b (float): Volatility of asset B.
        corr (float): Correlation coefficient between asset A and asset B.

    Returns:
        float: Covariance between asset A and asset B.
    """
    # Apply Cov(A, B) = sigma_A * sigma_B * rho_AB
    return vol_a * vol_b * corr

def market_portfolio_covariance(cov_market_a, cov_market_b, weight_a, weight_b):
    """
    Computes the covariance of a two-asset portfolio with the market portfolio.

    Args:
        cov_market_a (float): Covariance between asset A and the market.
        cov_market_b (float): Covariance between asset B and the market.
        weight_a (float): Portfolio weight allocated to asset A.
        weight_b (float): Portfolio weight allocated to asset B.

    Returns:
        float: Weighted covariance with the market portfolio.
    """
    # Use linearity of covariance: Cov(w_A*A + w_B*B, M) = w_A*Cov(A, M) + w_B*Cov(B, M)
    return cov_market_a * weight_a + cov_market_b * weight_b

def portfolio_return(return_a, return_b, weight_a, weight_b):
    """
    Computes the expected return of a two-asset portfolio.

    Args:
        return_a (float): Return of asset A.
        return_b (float): Return of asset B.
        weight_a (float): Weight allocated to asset A.
        weight_b (float): Weight allocated to asset B.

    Returns:
        float: Weighted portfolio return.
    """
    # Compute linear combination of the two asset returns
    return return_a * weight_a + return_b * weight_b

def portfolio_return_matrix(weights, expected_returns):
    """
    Computes annualized expected return for an N-asset portfolio via matrix product (w^T * mu).

    Args:
        weights (array-like): Vector of N portfolio weights.
        expected_returns (array-like): Vector of N annualized expected returns.

    Returns:
        float: Annualized expected portfolio return.
    """
    # Convert weights to a NumPy array
    w = np.asarray(weights)
    # Convert expected returns to a NumPy array
    mu = np.asarray(expected_returns)
    # Compute dot product w^T * mu
    return (w.T @ mu)

def portfolio_vol(vol_a, vol_b, weight_a, weight_b, cov_ab):
    """
    Computes the volatility of a two-asset portfolio using the quadratic variance formula.

    Args:
        vol_a (float): Volatility of asset A.
        vol_b (float): Volatility of asset B.
        weight_a (float): Weight allocated to asset A.
        weight_b (float): Weight allocated to asset B.
        cov_ab (float): Covariance between asset A and asset B.

    Returns:
        float: Portfolio volatility (standard deviation).
    """
    # Compute portfolio variance: w_A^2 * sigma_A^2 + w_B^2 * sigma_B^2 + 2 * w_A * w_B * Cov(A,B)
    variance = (vol_a**2) * (weight_a**2) + (vol_b**2) * (weight_b**2) + 2 * weight_a * weight_b * cov_ab
    # Return square root of variance
    return np.sqrt(variance) 

def portfolio_vol_matrix(weights, cov_matrix):
    """
    Computes annualized volatility for an N-asset portfolio: sqrt(w^T * Sigma * w).

    Args:
        weights (array-like): Vector of N portfolio weights.
        cov_matrix (array-like): N x N annualized covariance matrix.

    Returns:
        float: Annualized portfolio volatility.
    """
    # Convert weights to a NumPy array
    w = np.asarray(weights)
    # Convert covariance matrix to a NumPy array
    sigma = np.asarray(cov_matrix)
    # Compute quadratic form w^T * Sigma * w and take its square root
    return float(np.sqrt(w.T @ sigma @ w))

def portfolio_variance_matrix(weights, cov_matrix):
    """
    Computes annualized variance for an N-asset portfolio: w^T * Sigma * w.

    Args:
        weights (array-like): Vector of N portfolio weights.
        cov_matrix (array-like): N x N annualized covariance matrix.

    Returns:
        float: Annualized portfolio variance.
    """
    # Convert weights to a NumPy float array
    w = np.asarray(weights, dtype=float)
    # Convert covariance matrix to a NumPy float array
    sigma = np.asarray(cov_matrix, dtype=float)
    # Compute and return quadratic form w^T * Sigma * w
    return float(w.T @ sigma @ w)

def minimize_vol(target_return, expected_returns, cov_matrix, allow_short=False):
    """
    Finds minimum-variance portfolio weights achieving a given target return using SLSQP.

    Args:
        target_return (float): Desired annualized portfolio return.
        expected_returns (array-like): Vector of N annualized expected returns.
        cov_matrix (array-like): N x N annualized covariance matrix.
        allow_short (bool, optional): Allows negative weights if True; enforces [0, 1] if False. Defaults to False.

    Returns:
        np.ndarray: Optimal portfolio weights vector of length N.
    """
    # Count the number of assets N
    n_assets = len(expected_returns)
    # Initialize solver starting point with equal weights (1/N)
    init_guess = np.repeat(1.0 / n_assets, n_assets)
    # Set bounds: unconstrained if shorting allowed, otherwise [0, 1] per asset
    bounds = None if allow_short else ((0.0, 1.0),) * n_assets

    # Define equality constraints: full capital allocation (sum = 1) and target return achieved
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                   {"type": "eq", "fun": lambda w: portfolio_return_matrix(w, expected_returns) - target_return}
                   ]
    # Minimize portfolio variance subject to constraints using SLSQP
    result = minimize(
        portfolio_variance_matrix, 
        init_guess,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500}
    )
    # Return the optimal weight vector
    return result.x

def gmv_portfolio(cov_matrix, allow_short=False):
    """
    Computes the Global Minimum Variance (GMV) portfolio weights for N assets.

    Args:
        cov_matrix (array-like): N x N annualized covariance matrix.
        allow_short (bool, optional): Allows negative weights if True; enforces [0, 1] if False. Defaults to False.

    Returns:
        np.ndarray: GMV portfolio weight vector of length N.
    """
    # Determine number of assets N from covariance matrix rows
    n_assets = cov_matrix.shape[0]
    # Start optimization from equal weights (1/N)
    init_guess = np.repeat(1.0 / n_assets, n_assets)
    # Set bounds: unconstrained if shorting allowed, otherwise [0, 1] per asset
    bounds = None if allow_short else ((0.0, 1.0),) * n_assets
    # Enforce full investment constraint (sum of weights = 1)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Minimize portfolio volatility without any return target
    result = minimize(
        portfolio_vol_matrix,
        init_guess,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500}
    )
    # Return GMV weights
    return result.x

def msr_portfolio(expected_returns, cov_matrix, risk_free_rate=0.03, allow_short=False):
    """
    Computes the Maximum Sharpe Ratio (Tangency) portfolio weights for N assets.

    Args:
        expected_returns (array-like): Vector of N annualized expected returns.
        cov_matrix (array-like): N x N annualized covariance matrix.
        risk_free_rate (float, optional): Annual risk-free interest rate. Defaults to 0.03.
        allow_short (bool, optional): Allows negative weights if True; enforces [0, 1] if False. Defaults to False.

    Returns:
        np.ndarray: Tangency portfolio weight vector of length N.
    """
    # Count the number of assets N
    n_assets = len(expected_returns)
    # Initialize solver starting point with equal weights (1/N)
    init_guess = np.repeat(1.0 / n_assets, n_assets)
    # Set bounds: unconstrained if shorting allowed, otherwise [0, 1] per asset
    bounds = None if allow_short else ((0.0, 1.0),) * n_assets
    # Enforce full investment constraint (sum of weights = 1)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Define objective function returning the negative Sharpe ratio (to maximize Sharpe via minimization)
    def neg_sharpe(w):
        ret = portfolio_return_matrix(w, expected_returns)
        vol = portfolio_vol_matrix(w, cov_matrix)
        return -(ret - risk_free_rate) / vol

    # Run SLSQP optimization to minimize negative Sharpe ratio
    result = minimize(
        neg_sharpe,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500}
    )
    # Return tangency portfolio weights
    return result.x

def sharpe_ratio(portfolio_return, portfolio_vol, risk_free_rate = 0.03):
    """
    Computes the Sharpe Ratio (excess return per unit of volatility).

    Args:
        portfolio_return (float): Annualized portfolio return.
        portfolio_vol (float): Annualized portfolio volatility.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.03.

    Returns:
        float: Sharpe Ratio.
    """
    # Divide excess return above risk-free rate by portfolio volatility
    return (portfolio_return - risk_free_rate) / portfolio_vol

def efficient_frontier(return_a, return_b, vol_a, vol_b, cov_ab, risk_free_rate = 0.03, step=0.01):
    """
    Computes the 2-asset Markowitz efficient frontier across a uniform weight grid.

    Args:
        return_a (float): Annualized return of asset A.
        return_b (float): Annualized return of asset B.
        vol_a (float): Annualized volatility of asset A.
        vol_b (float): Annualized volatility of asset B.
        cov_ab (float): Covariance between asset A and asset B.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.03.
        step (float, optional): Weight grid increment. Defaults to 0.01.

    Returns:
        pd.DataFrame: Frontier table containing weights, return, volatility, and Sharpe ratio.
    """
    # Initialize list to collect each portfolio row
    rows = []
    # Iterate weight_a from 0 to 1 by increments of `step`
    for i in np.arange(0, 1.0+step, step):
        # Assign weight to asset A
        weight_a = i
        # Complementary weight for asset B
        weight_b = 1 - weight_a
        # Compute portfolio return, volatility, and Sharpe ratio
        portf_return = portfolio_return(return_a, return_b, weight_a, weight_b)
        portf_vol = portfolio_vol(vol_a, vol_b, weight_a, weight_b, cov_ab)
        sr = sharpe_ratio(portf_return, portf_vol, risk_free_rate=risk_free_rate)

        # Append metrics dictionary to rows list
        rows.append({
            "weight_a": float(weight_a),
            "weight_b": float(weight_b),
            "return": float(portf_return),
            "vol": float(portf_vol),
            "sharpe_ratio": float(sr)
        })

    # Convert collected rows into a DataFrame 
    return pd.DataFrame(rows)

def efficient_frontier_matrix(expected_returns, cov_matrix, risk_free_rate=0.03, n_points=50, allow_short=False):
    """
    Traces the Markowitz Efficient Frontier for N assets from GMV return up to maximum return.

    Args:
        expected_returns (pd.Series): Series of N annualized expected returns indexed by asset name.
        cov_matrix (pd.DataFrame or np.ndarray): N x N annualized covariance matrix.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.03.
        n_points (int, optional): Number of target return points along the frontier. Defaults to 50.
        allow_short (bool, optional): Allows negative weights if True; enforces [0, 1] if False. Defaults to False.

    Returns:
        pd.DataFrame: Efficient frontier DataFrame with return, vol, sharpe_ratio, weights array, and individual weight columns.
    """
    # Compute GMV weights and extract the minimum efficient return
    w_gmv = gmv_portfolio(cov_matrix, allow_short=allow_short)
    min_ret = portfolio_return_matrix(w_gmv, expected_returns)
    # Identify highest individual asset return
    max_ret = float(np.max(expected_returns))

    # Compute MSR (Tangency) weights and return so the frontier always includes the tangency point
    w_msr = msr_portfolio(expected_returns, cov_matrix, risk_free_rate=risk_free_rate, allow_short=allow_short)
    msr_ret = portfolio_return_matrix(w_msr, expected_returns)

    # Set upper return bound as the maximum of individual assets and MSR return
    max_asset_ret = float(np.max(expected_returns))
    upper_ret = max(max_asset_ret, msr_ret)
    # Guard against degenerate cases where upper_ret <= min_ret
    if upper_ret <= min_ret:
        upper_ret = min_ret = 1e-4

    # Create sorted grid of target returns explicitly including msr_ret
    target_rets = np.unique(np.sort(np.append(np.linspace(min_ret, upper_ret, n_points), msr_ret)))
    rows = []
    # Initialize list to store frontier points
    for r_target in target_rets:
        w = minimize_vol(r_target, expected_returns, cov_matrix, allow_short=allow_short)
        actual_ret = portfolio_return_matrix(w, expected_returns)
        vol = portfolio_vol_matrix(w, cov_matrix)
        sr = (actual_ret - risk_free_rate) / vol
        row = {"return": r_target, "vol": vol, "sharpe_ratio": sr, "weights": w}
        # Save each individual asset weight in its own named column
        for idx, name in enumerate(expected_returns.index):
            row[f"weight_{name}"] = float(w[idx])
        rows.append(row)
    
    # Return frontier results as a DataFrame
    return pd.DataFrame(rows)

def max_sharpe_ratio(eff_front):
    """
    Extracts the maximum Sharpe ratio value and row index from a 2-asset efficient frontier DataFrame.

    Args:
        eff_front (pd.DataFrame): Efficient frontier DataFrame containing a 'sharpe_ratio' column.

    Returns:
        tuple[float, int]: Maximum Sharpe ratio and corresponding DataFrame index.
    """
    # Find the index of the row with the highest Sharpe ratio
    max_idx = eff_front["sharpe_ratio"].idxmax()
    # Retrieve the maximum Sharpe ratio value
    max_sr = eff_front["sharpe_ratio"].max()
    # Return both value and index
    return max_sr, max_idx

def compute_weights(idx):
    """
    Converts an integer percentage index (0-100) into two complementary asset weights.

    Args:
        idx (int): Percentage weight assigned to asset A.

    Returns:
        tuple[float, float]: (weight_a, weight_b).
    """
    # Convert percentage to decimal weight for asset A
    weight_a = idx / 100
    # Compute complementary weight for asset B
    weight_b = 1 - weight_a
    # Return both weights
    return weight_a, weight_b

def capital_market_line(eff_front, tangency_idx, risk_free_rate = 0.03):
    """
    Computes X (volatility) and Y (return) coordinates for the Capital Market Line (2-asset version).

    Args:
        eff_front (pd.DataFrame): Efficient frontier DataFrame.
        tangency_idx (int): Row index of the tangency portfolio.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.03.

    Returns:
        tuple[list[float], list[float]]: X-coordinates (volatilities) and Y-coordinates (returns).
    """
    # Compute CML slope (Sharpe ratio of the tangency portfolio)
    slope = (eff_front["return"].iloc[tangency_idx]-risk_free_rate) / (eff_front["vol"].iloc[tangency_idx])
    # Extend upper X-coordinate 20% beyond maximum frontier volatility
    vol_max = eff_front["vol"].max() * 1.2
    # Compute extrapolated return along the CML at vol_max
    y_3 = slope*vol_max + risk_free_rate
    # Return three anchor points: Rf asset, Tangency portfolio, and leveraged extension
    return [0.0, eff_front["vol"].iloc[tangency_idx], vol_max], [risk_free_rate, eff_front["return"].iloc[tangency_idx], y_3]

def capital_market_line_matrix(tangency_ret, tangency_vol, max_plot_vol, risk_free_rate=0.03):
    """
    Computes X (volatility) and Y (return) coordinates for the Capital Market Line (N-asset version).

    Args:
        tangency_ret (float): Annualized return of the tangency (MSR) portfolio.
        tangency_vol (float): Annualized volatility of the tangency (MSR) portfolio.
        max_plot_vol (float): Maximum volatility displayed on the chart X-axis.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.03.

    Returns:
        tuple[list[float], list[float]]: X-coordinates (volatilities) and Y-coordinates (returns).
    """
    # Calculate slope of the Capital Market Line
    slope = (tangency_ret - risk_free_rate) / tangency_vol
    # Extend upper X-coordinate 20% past the highest plotted volatility
    vol_max = max_plot_vol * 1.2
    # Compute extrapolated return along the CML at vol_max
    y_max = slope * vol_max + risk_free_rate
    # Return three anchor points: Rf asset, Tangency portfolio, and leveraged extension
    return [0.0, tangency_vol, vol_max], [risk_free_rate, tangency_ret, y_max]

def target_volatility_weight(target_vol, portfolio_volatility):
    """
    Computes the risky portfolio weight required to achieve a target annual volatility along the CML.

    Args:
        target_vol (float): Desired annual portfolio volatility.
        portfolio_volatility (float): Annual volatility of the risky tangency portfolio.

    Returns:
        float: Weight allocated to the risky tangency portfolio (> 1 implies leverage).
    """
    # Ratio of target volatility to tangency portfolio volatility
    return target_vol / portfolio_volatility

def target_return_weight(target_ret, portfolio_ret, risk_free_rate = 0.03):
    """
    Computes the risky portfolio weight required to achieve a target annual return along the CML.

    Args:
        target_ret (float): Desired annual portfolio return.
        portfolio_ret (float): Annual return of the risky tangency portfolio.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.03.

    Returns:
        float: Weight allocated to the risky tangency portfolio (> 1 implies leverage).
    """
    # Solve target_ret = Rf + w * (portfolio_ret - Rf) for w
    return (target_ret - risk_free_rate) / (portfolio_ret - risk_free_rate)

def portfolio_historical_returns(returns_a, returns_b, weight_a, weight_b, risky_weight, rf_weight, risk_free_rate = 0.03, periods_per_year=12):
    """
    Reconstructs historical periodic returns for a 2-asset risky portfolio combined with a risk-free asset.

    Args:
        returns_a (pd.Series): Periodic returns of asset A.
        returns_b (pd.Series): Periodic returns of asset B.
        weight_a (float): Weight of asset A inside the risky bucket.
        weight_b (float): Weight of asset B inside the risky bucket.
        risky_weight (float): Total weight allocated to the risky bucket.
        rf_weight (float): Weight allocated to the risk-free asset.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.03.
        periods_per_year (int, optional): Number of return periods per year. Defaults to 12.

    Returns:
        pd.Series: Historical periodic returns of the complete portfolio.
    """
    # Convert annual risk-free rate into single-period rate
    rf_period = (1 + risk_free_rate)**(1 / periods_per_year) - 1
    # Compute effective portfolio weights for assets A and B
    weight_a_risk = risky_weight * weight_a
    weight_b_risk = risky_weight * weight_b
    # Return weighted sum of risky assets and risk-free asset returns
    return (weight_a_risk*returns_a + weight_b_risk * returns_b + rf_weight * rf_period)

def portfolio_historical_returns_matrix(returns_df, weights, risky_weight=1.0, rf_weight=0.0, risk_free_rate=0.03, periods_per_year=12):
    """
    Reconstructs historical periodic returns for an N-asset risky portfolio combined with a risk-free asset.

    Args:
        returns_df (pd.DataFrame): Aligned historical returns (T periods x N assets).
        weights (array-like): Vector of N risky asset weights.
        risky_weight (float, optional): Total weight allocated to the risky bucket. Defaults to 1.0.
        rf_weight (float, optional): Weight allocated to the risk-free asset. Defaults to 0.0.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.03.
        periods_per_year (int, optional): Number of return periods per year. Defaults to 12.

    Returns:
        pd.Series: Historical periodic returns of the complete portfolio indexed by date.
    """
    # Convert annual risk-free rate into single-period rate
    rf_period = (1 + risk_free_rate)**(1 / periods_per_year) - 1
    # Compute historical returns of the risky basket via matrix-vector product (R * w)
    risky_portf_returns = returns_df.values @ np.asarray(weights)
    # Combine risky basket returns and risk-free return into a dated Series
    return pd.Series(
        risky_weight * risky_portf_returns + rf_weight * rf_period,
        index=returns_df.index
    )

def skewness(returns):
    """
    Computes the sample skewness (third standardized moment) of a return series.
    Skewness = 0: symmetric; < 0: heavier left tail (extreme losses); > 0: heavier right tail.

    Args:
        returns (pd.Series): Periodic return series.

    Returns:
        float: Sample skewness.
    """
    # Subtract sample mean from returns
    centered = returns - returns.mean()
    # Compute population standard deviation (ddof=0)
    vol = returns.std(ddof=0)
    # Standardize returns to zero mean and unit variance
    standardized = (centered) / vol
    # Return mean of cubed standardized returns
    return (standardized**3).mean()

def kurtosis(returns):
    """
    Computes the excess kurtosis (fourth standardized moment minus 3) of a return series.
    Kurtosis = 0: normal tails; > 0: fat tails (leptokurtic); < 0: thin tails (platykurtic).

    Args:
        returns (pd.Series): Periodic return series.

    Returns:
        float: Excess kurtosis relative to the normal distribution.
    """
    # Subtract sample mean from returns
    centered = returns - returns.mean()
    # Compute population standard deviation (ddof=0)
    vol = returns.std(ddof=0)
    # Standardize returns to zero mean and unit variance
    standardized = (centered)/vol
    # Return mean of fourth powers minus 3 (Gaussian baseline)
    return (standardized**4).mean() - 3

def jarque_bera(returns, level=0.01):
    """
    Performs the Jarque-Bera statistical test for normality on a return series.

    Args:
        returns (pd.Series): Periodic return series.
        level (float, optional): Significance threshold (alpha). Defaults to 0.01.

    Returns:
        tuple[float, float, bool]: JB test statistic, p-value, and True if normality is not rejected.
    """
    # Remove any missing values
    returns = returns.dropna()
    # Count number of observations
    n = returns.shape[0]
    # Compute skewness and excess kurtosis
    s = skewness(returns)
    k = kurtosis(returns)

    # Compute Jarque-Bera statistic: (n / 6) * (S^2 + K^2 / 4)
    jb_stat = (n / 6.0) * (s**2 + (k**2) / 4.0)
    # Compute p-value from chi-squared distribution with 2 degrees of freedom
    p_value = 1.0 - stats.chi2.cdf(jb_stat, df=2)
    # Determine whether normality hypothesis holds at significance `level`
    is_normal = p_value > level
    # Return statistic, p-value, and boolean normality flag
    return jb_stat, p_value, is_normal

def parametric_var(returns, horizon_days, level = 0.01):
    """
    Computes parametric Gaussian Value-at-Risk (VaR) over a given trading-day horizon.

    Args:
        returns (pd.Series): Periodic return series.
        horizon_days (int): Risk horizon in trading days (e.g., 21 for 1 month).
        level (float, optional): Left-tail probability level (0.01 = 99% confidence). Defaults to 0.01.

    Returns:
        float: Positive VaR loss threshold over the horizon.
    """
    # Standard normal quantile at probability `level`
    z = norm.ppf(level)
    # Compute annualized return and volatility
    ann_ret = annualize_returns(returns)
    ann_vol = annualize_vol(returns)

    # Scale expected return and volatility to `horizon_days` (252 trading days/year)
    period_ret = (1 + ann_ret)**(horizon_days / 252) - 1
    period_vol = ann_vol / np.sqrt(252)
    # Return positive loss figure
    return float(-(period_ret + z * period_vol))    

def cornish_fisher_var(returns, level = 0.01):
    """
    Computes 1-month modified Cornish-Fisher VaR adjusted for skewness and excess kurtosis.

    Args:
        returns (pd.Series): Periodic return series.
        level (float, optional): Left-tail probability level (0.01 = 99% confidence). Defaults to 0.01.

    Returns:
        float: Positive 1-month Cornish-Fisher VaR loss threshold.
    """
    # Standard normal quantile, skewness, and excess kurtosis
    z = norm.ppf(level)
    s = skewness(returns)
    k = kurtosis(returns)
    # Apply fourth-order Cornish-Fisher expansion to adjust Gaussian quantile `z`
    z = (z + (s / 6) * ((z**2) - 1) + (k / 24) * ((z**3) - (3 * z)) - ((s**2) / 36) * ((2 * (z**3)) - (5 * z)))
    # Compute monthly expected return and monthly volatility
    r_m = (1 + annualize_returns(returns)) ** (1 / 12) - 1
    v_m = annualize_vol(returns) / np.sqrt(12)
    # Return positive VaR loss threshold
    return -(r_m + z * v_m)

def conditional_var(returns, level=0.01):
    """
    Computes historical Conditional Value-at-Risk (CVaR / Expected Shortfall).

    Args:
        returns (pd.Series): Periodic return series.
        level (float, optional): Left-tail probability cutoff (0.01 = worst 1%). Defaults to 0.01.

    Returns:
        float: Positive average loss beyond the VaR cutoff.
    """
    # Find empirical quantile threshold at `level`
    cutoff = returns.quantile(level)
    # Select all historical returns worse than or equal to the cutoff
    tail_losses = returns[returns <= cutoff]
    # Return the mean of tail losses as a positive value
    return - tail_losses.mean()

def max_drawdown(returns):
    """
    Computes the maximum peak-to-trough percentage decline of a return series.

    Args:
        returns (pd.Series): Periodic return series.

    Returns:
        tuple[Timestamp, float]: Date of the worst trough and maximum drawdown percentage (negative).
    """
    # Build cumulative wealth index starting at 100
    wealth_index = 100 * (1 + returns).cumprod()
    # Track running maximum peak wealth up to each date
    previous_meaks = wealth_index.cummax()
    # Compute percentage drawdown from running peaks
    drawdonws = (wealth_index - previous_meaks) / previous_meaks
    # Return the date and value of the deepest drawdown
    return drawdonws.idxmin(), drawdonws.min()

def run_cppi(risk_returns, initial_wealth, floor=0.8, multiplier=1.0, risk_free_rate=0.03, rebalance_freq = 3.0):
    """
    Backtests a standard Constant Proportion Portfolio Insurance (CPPI) strategy.

    Args:
        risk_returns (pd.Series): Periodic returns of the risky underlying portfolio.
        initial_wealth (float): Starting portfolio capital.
        floor (float, optional): Protected floor ratio relative to initial wealth. Defaults to 0.8.
        multiplier (float, optional): CPPI risk multiplier (m). Defaults to 1.0.
        risk_free_rate (float, optional): Annual risk-free interest rate. Defaults to 0.03.
        rebalance_freq (float, optional): Rebalancing interval in periods (months). Defaults to 3.0.

    Returns:
        pd.DataFrame: Historical trajectories for 'CPPI Wealth', 'Floor', 'Risky Wealth', and 'Cushion'.
    """
    # Number of historical steps
    n_step = len(risk_returns)
    # Convert annual risk-free rate into monthly rate
    rf_monthly = (1 + risk_free_rate)**(1 / 12) - 1
    # Compute initial monetary floor value
    floor_value = floor * initial_wealth

    # Pre-allocate NumPy tracking arrays
    wealth_hist = np.zeros(n_step + 1)
    risk_wealth_hist = np.zeros(n_step + 1)
    floor_hist = np.zeros(n_step + 1)
    cushion_hist = np.zeros(n_step + 1)

    # Set initial t = 0 values for wealth, floor, cushion, and risky allocation
    wealth_hist[0] = initial_wealth
    floor_hist[0] = floor_value
    cushion_hist[0] = initial_wealth-floor_value
    risk_wealth_hist[0] = np.minimum(wealth_hist[0], multiplier * cushion_hist[0])

    # Iterate through each period t
    for t in range(n_step):
        # Extract risky return at step t
        rets = risk_returns.iloc[t]
        # Accrue the guaranteed floor at the risk-free rate
        floor_hist[t + 1] = floor_hist[t] * (1 + rf_monthly)

        # Compute end-of-period value of risky and risk-free pockets
        risk_end = risk_wealth_hist[t] * (1 + rets)
        cash_end = (wealth_hist[t] - risk_wealth_hist[t]) * (1 + rf_monthly)
        # Update total portfolio wealth
        wealth_hist[t + 1] =  risk_end + cash_end
        # Rebalance cushion and risky exposure on scheduled dates
        if (t + 1) % int(rebalance_freq) == 0:
            cushion_hist[t + 1] = np.maximum(0, wealth_hist[t + 1] - floor_hist[t + 1])
            risk_wealth_hist[t + 1] = np.minimum(wealth_hist[t + 1], multiplier * cushion_hist[t + 1])
        else :
            # Between rebalancings, keep previous cushion and let risky wealth drift
            cushion_hist[t + 1] = cushion_hist[t]
            risk_wealth_hist[t + 1] = risk_end

    # Return trajectories as a DataFrame indexed by input dates
    return pd.DataFrame({
        "CPPI Wealth":wealth_hist[1:],
        "Floor":floor_hist[1:],
        "Risky Wealth":risk_wealth_hist[1:],
        "Cushion":cushion_hist[1:]
        }, index=risk_returns.index)

def run_cppi_lockin(risk_returns, initial_wealth, floor=0.8, multiplier=1.0, risk_free_rate=0.03, rebalance_freq = 3.0):
    """
    Backtests a CPPI strategy with dynamic profit lock-in (ratcheting floor based on peak wealth).

    Args:
        risk_returns (pd.Series): Periodic returns of the risky underlying portfolio.
        initial_wealth (float): Starting portfolio capital.
        floor (float, optional): Protected floor percentage applied to running peak wealth. Defaults to 0.8.
        multiplier (float, optional): CPPI risk multiplier (m). Defaults to 1.0.
        risk_free_rate (float, optional): Annual risk-free interest rate. Defaults to 0.03.
        rebalance_freq (float, optional): Rebalancing interval in periods (months). Defaults to 3.0.

    Returns:
        pd.DataFrame: Historical trajectories for 'CPPI Wealth', ratcheting 'Floor', 'Risky Wealth', and 'Cushion'.
    """
    # Number of historical steps
    n_step = len(risk_returns)
    # Convert annual risk-free rate into monthly rate
    rf_monthly = (1 + risk_free_rate)**(1 / 12) - 1
    # Compute initial monetary floor value
    floor_value = floor * initial_wealth

    # Pre-allocate NumPy tracking arrays
    wealth_hist = np.zeros(n_step + 1)
    risk_wealth_hist = np.zeros(n_step + 1)
    floor_hist = np.zeros(n_step + 1)
    cushion_hist = np.zeros(n_step + 1)

    # Set initial t = 0 values and initialize running peak wealth
    wealth_hist[0] = initial_wealth
    floor_hist[0] = floor_value
    cushion_hist[0] = initial_wealth - floor_value
    risk_wealth_hist[0] = np.minimum(wealth_hist[0], multiplier * cushion_hist[0])
    peak_wealth = initial_wealth

    # Iterate through each period t
    for t in range(n_step):
        # Extract risky return at step t
        rets = risk_returns.iloc[t]

        # Compute end-of-period value of risky and risk-free pockets
        risk_end = risk_wealth_hist[t] * (1 + rets)
        cash_end = (wealth_hist[t] - risk_wealth_hist[t]) * (1 + rf_monthly)
        # Update total portfolio wealth
        wealth_hist[t + 1] = risk_end + cash_end

        # Update highest observed wealth and ratchet up the guaranteed floor
        peak_wealth = np.maximum(peak_wealth, wealth_hist[t + 1])
        floor_hist[t + 1] = np.maximum(floor_hist[t] * (1 + rf_monthly), floor * peak_wealth)

        # Rebalance cushion and risky exposure on scheduled dates
        if (t  +1) % int(rebalance_freq) == 0:    
            cushion_hist[t + 1] = np.maximum(0, wealth_hist[t + 1] - floor_hist[t + 1])
            risk_wealth_hist[t + 1] = np.minimum(wealth_hist[t + 1], multiplier * cushion_hist[t + 1])                                
        else:
            # Between rebalancings, keep previous cushion and let risky wealth drift
            cushion_hist[t + 1] = cushion_hist[t]
            risk_wealth_hist[t + 1] = risk_end               

    # Return trajectories as a DataFrame indexed by input dates
    return pd.DataFrame({
        "CPPI Wealth":wealth_hist[1:],
        "Floor":floor_hist[1:],
        "Risky Wealth":risk_wealth_hist[1:],
        "Cushion":cushion_hist[1:]
        }, index=risk_returns.index)

def run_buy_and_hold(risk_returns, initial_wealth):
    """
    Simulates a passive 100% Buy & Hold strategy across historical returns.

    Args:
        risk_returns (pd.Series): Periodic return series.
        initial_wealth (float): Starting capital.

    Returns:
        pd.DataFrame: Time series of Buy & Hold portfolio 'Wealth'.
    """
    # Count total number of periods
    n_steps = len(risk_returns)
    # Pre-allocate wealth array
    wealth_values = np.zeros(n_steps + 1)
    # Set initial capital at t = 0
    wealth_values[0] = initial_wealth
    # Compound wealth period by period
    for t, date in enumerate(risk_returns.index):
        wealth_values[t+1] = wealth_values[t] * (1 + risk_returns.iloc[t])

    # Return simulated wealth series in a DataFrame
    return pd.DataFrame({"Wealth": wealth_values[1:]}, index=risk_returns.index)

def run_monte_carlo_gbm(returns, n_scenarios=100, steps_per_year=12, n_years = 10, initial_wealth=1000):
    """
    Simulates forward wealth paths using a parametric Geometric Brownian Motion (GBM).

    Args:
        returns (pd.Series): Historical periodic returns used to estimate drift and volatility.
        n_scenarios (int, optional): Number of simulated trajectories. Defaults to 100.
        steps_per_year (int, optional): Time steps per year. Defaults to 12.
        n_years (int, optional): Simulation horizon in years. Defaults to 10.
        initial_wealth (float, optional): Starting wealth. Defaults to 1000.

    Returns:
        pd.DataFrame: Simulated wealth paths (rows = time steps, columns = scenarios).
    """
    # Estimate annualized volatility and annualized drift from historical returns
    sigma = (returns.std()) * np.sqrt(steps_per_year)
    mu = (1 + returns.mean())**steps_per_year - 1
    # Total number of time points including t = 0
    n_steps = (n_years * steps_per_year) + 1

    # Draw standard normal random shocks Z ~ N(0, 1)
    z = np.random.normal(0.0, 1.0, size=(n_steps, n_scenarios))
    # Compute discrete GBM growth factors exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    growth_factors = np.exp((mu - 0.5 * sigma**2)*(1 / steps_per_year) + sigma * np.sqrt(1 / steps_per_year) * z)
    # Set t = 0 factor to 1.0 so every path starts at `initial_wealth`
    growth_factors[0] = 1.0
    # Compound growth factors across time and scale by initial capital
    return initial_wealth * pd.DataFrame(growth_factors).cumprod()

def run_monte_carlo_bootstrap(returns, n_scenarios=100, steps_per_year=12, n_years=10, initial_wealth=1000):
    """
    Simulates forward wealth paths using non-parametric historical bootstrap resampling.

    Args:
        returns (pd.Series): Historical periodic returns pool to sample from.
        n_scenarios (int, optional): Number of simulated scenarios. Defaults to 100.
        steps_per_year (int, optional): Time steps per year. Defaults to 12.
        n_years (int, optional): Simulation horizon in years. Defaults to 10.
        initial_wealth (float, optional): Starting wealth. Defaults to 1000.

    Returns:
        pd.DataFrame: Simulated wealth paths (rows = time steps, columns = scenarios).
    """
    # Total number of time points including t = 0
    n_step = (n_years * steps_per_year) + 1
    # Randomly sample historical returns with replacement and convert to gross return factors (1 + r)
    norm = 1 + np.random.choice(np.asarray(returns).ravel(), size=(n_step, n_scenarios))
    # Set t = 0 factor to 1.0 so every path starts at `initial_wealth
    norm[0] = 1.0
    # Compound resampled returns across time and scale by initial capital
    return initial_wealth * pd.DataFrame(norm).cumprod()

def run_cppi_monte_carlo(risk_returns, initial_wealth=10000, n_scenarios=100, steps_per_year=12, n_years=10, floor=0.8, multiplier=1.0, risk_free_rate=0.03, rebalance_freq=3):
    """
    Simulates vectorized forward-looking CPPI wealth and floor paths via historical bootstrap resampling.

    Args:
        risk_returns (pd.Series): Historical periodic returns of the underlying risky portfolio.
        initial_wealth (float, optional): Starting capital. Defaults to 10000.
        n_scenarios (int, optional): Number of simulated trajectories. Defaults to 100.
        steps_per_year (int, optional): Time steps per year. Defaults to 12.
        n_years (int, optional): Projection horizon in years. Defaults to 10.
        floor (float, optional): Protected floor ratio relative to initial wealth. Defaults to 0.8.
        multiplier (float, optional): CPPI risk multiplier (m). Defaults to 1.0.
        risk_free_rate (float, optional): Annual risk-free interest rate. Defaults to 0.03.
        rebalance_freq (int, optional): Rebalancing interval in steps. Defaults to 3.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Simulated CPPI wealth paths and floor paths indexed by future dates.
    """
    # Compute total simulation steps
    n_steps = (n_years * steps_per_year)
    # Resample historical returns with replacement across all steps and scenarios
    simulated_returns = np.random.choice(np.asarray(risk_returns).ravel(), size=(n_steps, n_scenarios))
    # Compute single-period risk-free rate
    rf_period = (1 + risk_free_rate)**(1 / steps_per_year) - 1
    # Compute initial monetary floor value
    floor_value = floor * initial_wealth

    # Pre-allocate 2D arrays (time steps x scenarios)
    wealth_hist = np.zeros((n_steps + 1, n_scenarios))
    risky_wealth_hist = np.zeros((n_steps + 1, n_scenarios))
    floor_hist = np.zeros((n_steps + 1, n_scenarios))
    cushion_hist = np.zeros((n_steps + 1, n_scenarios))

    # Initialize t = 0 values across all scenarios simultaneously
    wealth_hist[0] = initial_wealth
    floor_hist[0] = floor_value
    cushion_hist[0] = initial_wealth - floor_value
    risky_wealth_hist[0] = np.minimum(wealth_hist[0], multiplier * cushion_hist[0])

    # Step forward in time, updating all scenarios vectorized
    for t in range(n_steps):
        # Simulated returns vector for step t across all scenarios
        rets = simulated_returns[t]
        # Accrue floor at the periodic risk-free rate
        floor_hist[t + 1] = floor_hist[t] * (1 + rf_period)

        # Evolve risky and safe pockets
        risk_end = risky_wealth_hist[t] * (1 + rets)
        cash_end = (wealth_hist[t] - risky_wealth_hist[t]) * (1 + rf_period)
        wealth_hist[t + 1] = risk_end + cash_end

        # Rebalance cushion and risky exposure on scheduled dates
        if (t + 1) % int(rebalance_freq) == 0:
            cushion_hist[t + 1] = np.maximum(0, wealth_hist[t + 1] - floor_hist[t + 1])
            risky_wealth_hist[t + 1] = np.minimum(wealth_hist[t + 1], multiplier * cushion_hist[t + 1])
        else :
            cushion_hist[t + 1] = cushion_hist[t]
            risky_wealth_hist[t + 1] = risk_end

    # Anchor future date index to the last historical observation date
    last_date = pd.to_datetime(risk_returns.index[-1])
    # Map steps_per_year to pandas date offset aliases
    freq_map = {12: "ME", 52: "W", 252: "B", 4: "QE", 1: "YE"}
    freq = freq_map.get(int(steps_per_year), "ME")

    # Generate future date range and wrap arrays into DataFrames
    future_index = pd.date_range(start=last_date, periods=n_steps + 1, freq=freq)
    df_cppi_wealth = pd.DataFrame(wealth_hist, index=future_index)
    df_cppi_floor = pd.DataFrame(floor_hist, index=future_index)
    return df_cppi_wealth, df_cppi_floor

def plot_monte_carlo_paths(paths_df, n_paths):
    """
    Plots individual simulated Monte Carlo wealth trajectories on a Plotly figure.

    Args:
        paths_df (pd.DataFrame): DataFrame of simulated paths (columns = scenarios).
        n_paths (int): Maximum number of trajectories to render.

    Returns:
        go.Figure: Plotly Figure containing the plotted lines.
    """
    # Initialize an empty Plotly Figure
    fig = go.Figure()
    # Add up to `n_paths` individual line traces without legend clutter
    for i in range(min(n_paths, paths_df.shape[1])):
        fig.add_trace(go.Scatter(x=paths_df.index, y=paths_df.iloc[:,i], mode='lines', showlegend=False))
    # Return the figure
    return fig

def style_plotly(fig, title="", x_title="", y_title="", x_format=None, y_format=None):
    """
    Applies a standardized dark theme, transparent background, grid styling, and unified hover layout to a Plotly figure.

    Args:
        fig (go.Figure): Plotly Figure object to style.
        title (str, optional): Chart title. Defaults to "".
        x_title (str, optional): X-axis title. Defaults to "".
        y_title (str, optional): Y-axis title. Defaults to "".
        x_format (str, optional): D3 tick format string for X-axis (e.g., '.1%'). Defaults to None.
        y_format (str, optional): D3 tick format string for Y-axis (e.g., '.1%'). Defaults to None.

    Returns:
        go.Figure: Styled Plotly Figure.
    """
    # Update global layout, dark template, transparent backgrounds, horizontal legend, and axes styling
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="sans-serif"), y=0.92),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.45)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=100, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            tickformat=x_format
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            tickformat=y_format
        ),
    )
    # Return styled figure
    return fig

def compute_risk_table(series_returns, risk_free_rate=0.03, is_normal=False):
    """
    Computes a summary dictionary of key performance and risk metrics for a return series.

    Args:
        series_returns (pd.Series): Periodic return series to evaluate.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.03.
        is_normal (bool, optional): Uses Gaussian Parametric VaR if True, Cornish-Fisher VaR if False. Defaults to False.

    Returns:
        dict[str, float]: Dictionary mapping metric names to their computed numerical values.
    """
    # Drop any missing observations
    r = series_returns.dropna()
    # Compute annualized return, annualized volatility, Sharpe ratio, CVaR, and Max Drawdown
    ret_ann = annualize_returns(r)
    vol_ann = annualize_vol(r)
    sr = sharpe_ratio(ret_ann, vol_ann, risk_free_rate=risk_free_rate)
    cvar = conditional_var(r)
    _, mdd = max_drawdown(r)
    # Use Parametric Gaussian VaR when returns follow a normal distribution
    if is_normal:
        var = parametric_var(r, horizon_days=21)
        return {
                "Annualized Return": ret_ann,
                "Annualized Volatility": vol_ann,
                "Sharpe Ratio": sr,
                "Parametric VaR (99%, 1M)": var,
                "CVaR / Expected Shortfall (99%)": cvar,
                "Max Drawdown": mdd 
            }
    # Otherwise use skewness/kurtosis-adjusted Cornish-Fisher VaR
    else:
        var_cf = cornish_fisher_var(r)
        return {
                "Annualized Return": ret_ann,
                "Annualized Volatility": vol_ann,
                "Sharpe Ratio": sr,
                "Cornish-Fisher VaR (99%, 1M)": var_cf,
                "CVaR / Expected Shortfall (99%)": cvar,
                "Max Drawdown": mdd 
            }
    
    
