#problem 1
import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma = 0.1  # Standard deviation
P_t_minus_1 = 100  # Initial price
num_simulations = 10000  # Number of simulations

# Simulate returns
returns = np.random.normal(0, sigma, num_simulations)

# 1. Classical Brownian Motion
P_t_classical = P_t_minus_1 + returns

# 2. Arithmetic Return System
P_t_arithmetic = P_t_minus_1 * (1 + returns)

# 3. Geometric Brownian Motion (Log Return)
P_t_log = P_t_minus_1 * np.exp(returns)

# Calculate theoretical expectations
expected_classical = P_t_minus_1
std_classical = sigma

expected_arithmetic = P_t_minus_1
std_arithmetic = P_t_minus_1 * sigma

expected_log = P_t_minus_1 * np.exp(sigma**2 / 2)
std_log = P_t_minus_1 * np.sqrt(np.exp(sigma**2) - 1)

# Simulated mean and std
simulated_classical_mean = np.mean(P_t_classical)
simulated_classical_std = np.std(P_t_classical)

simulated_arithmetic_mean = np.mean(P_t_arithmetic)
simulated_arithmetic_std = np.std(P_t_arithmetic)

simulated_log_mean = np.mean(P_t_log)
simulated_log_std = np.std(P_t_log)

# Print results
print(f"Classical Brownian Motion: Theoretical mean = {expected_classical}, Simulated mean = {simulated_classical_mean}")
print(f"Classical Brownian Motion: Theoretical std = {std_classical}, Simulated std = {simulated_classical_std}")

print(f"Arithmetic Return System: Theoretical mean = {expected_arithmetic}, Simulated mean = {simulated_arithmetic_mean}")
print(f"Arithmetic Return System: Theoretical std = {std_arithmetic}, Simulated std = {simulated_arithmetic_std}")

print(f"Log Return (Geometric Brownian Motion): Theoretical mean = {expected_log}, Simulated mean = {simulated_log_mean}")
print(f"Log Return (Geometric Brownian Motion): Theoretical std = {std_log}, Simulated std = {simulated_log_std}")

# Plot histograms for visualization
plt.figure(figsize=(12, 8))
plt.hist(P_t_classical, bins=50, alpha=0.5, label='Classical Brownian Motion')
plt.hist(P_t_arithmetic, bins=50, alpha=0.5, label='Arithmetic Return System')
plt.hist(P_t_log, bins=50, alpha=0.5, label='Geometric Brownian Motion')
plt.legend()
plt.title("Simulated Price Distributions for Different Return Systems")
plt.show()



#problem 2
#2.1

import pandas as pd
import numpy as np

def calculate_returns(prices: pd.DataFrame, method="DISCRETE", date_column="Date"):
    # Extract the stock names (the first row) and remove the date column
    stock_names = prices.columns[1:]  # Stock names are in the first row (skip the date column)
    
    # Extract the price data (everything except the first row and first column)
    price_data = prices.iloc[:, 1:].values.astype(float)  # Convert to float for calculations
    
    # Get the number of rows and columns
    n, m = price_data.shape
    
    # Initialize the return matrix
    p2 = np.empty((n-1, m), dtype=float)
    
    # Calculate the returns
    for i in range(n-1):
        for j in range(m):
            p2[i, j] = price_data[i+1, j] / price_data[i, j]
    
    # Adjust the returns based on the method specified
    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"Method {method} must be either 'LOG' or 'DISCRETE'.")
    
    # Extract the dates (adjust length to match returns)
    dates = prices[date_column].values[1:]  # All dates except the first one (to match return matrix)
    
    # Create the output DataFrame with dates and calculated returns
    return_df = pd.DataFrame(p2, columns=stock_names)
    return_df[date_column] = dates  # Add back the dates
    
    return return_df

# Load the CSV file
prices = pd.read_csv('DailyPrices.csv')

# Calculate arithmetic returns (DISCRETE method)
arithmetic_returns = calculate_returns(prices, method="DISCRETE")

# Display the first few rows of the returns
print(arithmetic_returns.head())

#2.2
import pandas as pd
import numpy as np
from scipy.stats import norm, t  # Import T-distribution correctly from scipy.stats
import statsmodels.api as sm

prices = pd.read_csv('DailyPrices.csv')
dates = prices.iloc[:, 0]  # First column contains the dates
meta_prices = prices['META']  # Replace 'META' with the correct column name for META in your file

meta_returns = meta_prices.pct_change().dropna()

meta_returns = meta_returns - meta_returns.mean()

confidence_level = 0.05
days = 1  # 1-day VaR

# 1. Using a normal distribution
mean_return = np.mean(meta_returns)
std_dev = np.std(meta_returns)
var_normal = norm.ppf(confidence_level) * std_dev
print(f"VaR (Normal Distribution): {var_normal}")

# 2. Using a normal distribution with Exponentially Weighted variance (lambda = 0.94)
lambda_ewma = 0.94
ewma_variance = np.zeros_like(meta_returns)
ewma_variance[0] = np.var(meta_returns)
for i in range(1, len(meta_returns)):
    ewma_variance[i] = lambda_ewma * ewma_variance[i-1] + (1 - lambda_ewma) * meta_returns.iloc[i-1] ** 2
ewma_std_dev = np.sqrt(ewma_variance[-1])
var_ewma = norm.ppf(confidence_level) * ewma_std_dev
print(f"VaR (EWMA): {var_ewma}")

# 3. Using a MLE fitted T distribution
t_distribution_params = t.fit(meta_returns)  # Fit T-distribution to the returns
df_t, loc_t, scale_t = t_distribution_params
var_t = t.ppf(confidence_level, df_t, loc=loc_t, scale=scale_t)
print(f"VaR (T Distribution): {var_t}")

# 4. Using a fitted AR(1) model
model = sm.tsa.ARIMA(meta_returns, order=(1, 0, 0))
ar_result = model.fit()
ar_std_dev = np.std(ar_result.resid)
var_ar = norm.ppf(confidence_level) * ar_std_dev
print(f"VaR (AR(1) Model): {var_ar}")

# 5. Using a Historical Simulation
var_historical = np.percentile(meta_returns, confidence_level * 100)
print(f"VaR (Historical Simulation): {var_historical}")

# Compare the 5 values
print(f"Comparison of VaR values:\n"
      f"1. Normal Distribution: {var_normal}\n"
      f"2. EWMA: {var_ewma}\n"
      f"3. MLE T Distribution: {var_t}\n"
      f"4. AR(1) Model: {var_ar}\n"
      f"5. Historical Simulation: {var_historical}")


#problem 3
import pandas as pd
import numpy as np

#
def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
    vars_ = prices.columns
    nVars = len(vars_)
    vars_ = [var for var in vars_ if var != dateColumn]
    if nVars == len(vars_):
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {vars_}")
    nVars = nVars - 1

    p = prices[vars_].to_numpy()
    n, m = p.shape
    p2 = np.empty((n-1, m))

    for i in range(n-1):
        for j in range(m):
            p2[i, j] = p[i+1, j] / p[i, j]

    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")

    dates = prices[dateColumn].iloc[1:n].to_numpy()
    out = pd.DataFrame({dateColumn: dates})
    for i in range(nVars):
        out[vars_[i]] = p2[:, i]

    return out
    
def exp_weighted_cov(returns, lambda_):  
    # Preprocess the data
    returns = returns.values
    mean_return = np.mean(returns, axis=0)
    normalized_returns = returns - mean_return
    
    # Initializing the covariance matrix
    n_timesteps = normalized_returns.shape[0]
    Var = np.cov(returns, rowvar=False)
    
    # Updating the covariance matrix
    for t in range(1, n_timesteps):
        Var = lambda_ * Var + (1 - lambda_) * np.outer(normalized_returns[t], normalized_returns[t])
    return Var
    

#
portfolio = pd.read_csv("portfolio.csv")
prices = pd.read_csv("DailyPrices.csv")

#
def get_portfolio_price(portfolio, prices, portfolio_code, Delta=False):
    """Get the price for each asset in portfolio and calculate the current price."""
    
    if portfolio_code == "All":
        assets = portfolio.drop('Portfolio',axis=1)
        assets = assets.groupby(["Stock"], as_index=False)["Holding"].sum()
    else:
        assets = portfolio[portfolio["Portfolio"] == portfolio_code]
        
    stock_codes = list(assets["Stock"])
    assets_prices = pd.concat([prices["Date"], prices[stock_codes]], axis=1)
    
    current_price = np.dot(prices[assets["Stock"]].tail(1), assets["Holding"])
    holdings = assets["Holding"]
    
    if Delta == True:
        asset_values = assets["Holding"].values.reshape(-1, 1) * prices[assets["Stock"]].tail(1).T.values
        delta = asset_values / current_price
        
        return current_price, assets_prices, delta
    
    return current_price, assets_prices, holdings
    
#Calculate VAR ew
def calculate_delta_var(portfolio, prices, alpha=0.05, lambda_=0.94, portfolio_code="All"):
    
    current_price, assets_prices, delta = get_portfolio_price(portfolio, prices, portfolio_code, Delta=True)

    returns = return_calculate(assets_prices).drop('Date', axis=1)
    assets_cov = exp_weighted_cov(returns, lambda_)

    p_sig = np.sqrt(np.transpose(delta) @ assets_cov @ delta)
    
    var_delta = -current_price * norm.ppf(alpha) * p_sig
    
    return current_price[0], var_delta[0][0]

#Calculate VaR Monte Carl
def multivariate_normal_simulation(covariance_matrix, n_samples, explained_variance=1.0):

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Only consider eigenvalues greater than 0
        idx = eigenvalues > 1e-8
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Sort the eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Update the explained_variance incase the explained_variance is higher than the cumulative sum of the eigenvalue
        if explained_variance == 1.0:
            explained_variance = (np.cumsum(eigenvalues)/np.sum(eigenvalues))[-1]
        
        # Determine the number of components to keep based on the explained variance ratio
        n_components = np.where((np.cumsum(eigenvalues)/np.sum(eigenvalues))>= explained_variance)[0][0] + 1
        eigenvectors = eigenvectors[:,:n_components]
        eigenvalues = eigenvalues[:n_components]

        normal_samples = np.random.normal(size=(n_components, n_samples))
        
        # Simulate the multivariate normal samples by multiplying the eigenvectors with the normal samples
        B = np.dot(eigenvectors, np.diag(np.sqrt(eigenvalues)))
        samples = np.transpose(np.dot(B, normal_samples))
        
        return samples
        
def calculate_MC_var(portfolio, prices, alpha=0.05, lambda_=0.97, n_simulation = 10000, portfolio_code="All"):
    current_price, assets_prices, holdings = get_portfolio_price(portfolio, prices, portfolio_code)
    returns = return_calculate(assets_prices).drop('Date',axis=1)
    returns_norm = returns - returns.mean()
    assets_cov = exp_weighted_cov(returns_norm, lambda_)
    assets_prices = assets_prices.drop('Date',axis=1)
    np.random.seed(0)
    sim_returns = np.add(multivariate_normal_simulation(assets_cov, n_simulation), returns.mean().values)
    sim_prices = np.dot(sim_returns* assets_prices.tail(1).values.reshape(assets_prices.shape[1],),holdings)
    var_MC = -np.percentile(sim_prices, alpha*100)
    return current_price[0], var_MC, sim_prices
    
def calculate_historic_var(portfolio, prices, alpha=0.05,n_simulation=1000, portfolio_code="All"):
   
    current_price, assets_prices, holdings = get_portfolio_price(portfolio, prices, portfolio_code)
    
    returns = return_calculate(assets_prices).drop("Date", axis=1)
    
    assets_prices = assets_prices.drop('Date',axis=1)
    sim_returns = returns.sample(n_simulation, replace=True)
    sim_prices = np.dot(sim_returns* assets_prices.tail(1).values.reshape(assets_prices.shape[1],),holdings)
    
    var_hist = -np.percentile(sim_prices, alpha*100)
    
    return current_price[0], var_hist, sim_prices

code = 'A'
current_price, delta_var = calculate_delta_var(portfolio, prices, portfolio_code=code)
current_price, mc_var, mc_sim_prices = calculate_MC_var(portfolio, prices, portfolio_code=code)
current_price, hist_var, hist_sim_prices = calculate_historic_var(portfolio, prices, portfolio_code=code)
print("For porfolio_{}".format(code))
print("The current value is: {:.2f}".format(current_price))
print("VaR (Delta Normal) is: {:.2f}".format(delta_var))
print("VaR (Monte Carlo) is: {:.2f}".format(mc_var))
print("VaR (Historic) is: {:.2f}\n".format(hist_var))

code = 'B'
current_price, delta_var = calculate_delta_var(portfolio, prices, portfolio_code=code)
current_price, mc_var, mc_sim_prices = calculate_MC_var(portfolio, prices, portfolio_code=code)
current_price, hist_var, hist_sim_prices = calculate_historic_var(portfolio, prices, portfolio_code=code)
print("For porfolio_{}".format(code))
print("The current value is: {:.2f}".format(current_price))
print("VaR (Delta Normal) is: {:.2f}".format(delta_var))
print("VaR (Monte Carlo) is: {:.2f}".format(mc_var))
print("VaR (Historic) is: {:.2f}\n".format(hist_var))

# Get the list of stock names from the DailyPrices.csv
available_stocks = set(prices.columns[1:])  # Exclude the 'Date' column

# Filter out rows in portfolio that contain stocks not in DailyPrices
missing_stocks = portfolio[~portfolio['Stock'].isin(available_stocks)]
if not missing_stocks.empty:
    print("The following stocks are missing from the price data and will be excluded:")
    print(missing_stocks)
    portfolio = portfolio[portfolio['Stock'].isin(available_stocks)]


code = 'C'
current_price, delta_var = calculate_delta_var(portfolio, prices, portfolio_code=code)
current_price, mc_var, mc_sim_prices = calculate_MC_var(portfolio, prices, portfolio_code=code)
current_price, hist_var, hist_sim_prices = calculate_historic_var(portfolio, prices, portfolio_code=code)
print("For porfolio_{}".format(code))
print("The current value is: {:.2f}".format(current_price))
print("VaR (Delta Normal) is: {:.2f}".format(delta_var))
print("VaR (Monte Carlo) is: {:.2f}".format(mc_var))
print("VaR (Historic) is: {:.2f}\n".format(hist_var))

code = 'All'
current_price, delta_var = calculate_delta_var(portfolio, prices, portfolio_code=code)
current_price, mc_var, mc_sim_prices = calculate_MC_var(portfolio, prices, portfolio_code=code)
current_price, hist_var, hist_sim_prices = calculate_historic_var(portfolio, prices, portfolio_code=code)
print("For porfolio_{}".format(code))
print("The current value is: {:.2f}".format(current_price))
print("VaR (Delta Normal) is: {:.2f}".format(delta_var))
print("VaR (Monte Carlo) is: {:.2f}".format(mc_var))
print("VaR (Historic) is: {:.2f}\n".format(hist_var))
