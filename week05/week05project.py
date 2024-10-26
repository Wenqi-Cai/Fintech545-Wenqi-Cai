import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t, kurtosis, spearmanr
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize 
from scipy.integrate import quad

#problem 2
data = pd.read_csv('problem1.csv')

# 1. a normal distribution with an exponentially weighted variance
def exponentially_weighted_variance(returns, lambd=0.97):
    """Calculate exponentially weighted standard deviation"""
    weights = np.array([(1 - lambd) * lambd**i for i in range(len(returns))])
    weights /= weights.sum()
    weighted_variance = np.dot(weights, (returns - np.mean(returns))**2)
    return np.sqrt(weighted_variance)

def calculate_var_es_normal(returns, alpha=0.05, lambd=0.97):
    """Calculate VaR and ES under a normal distribution with exponentially weighted variance"""
    mean = np.mean(returns)
    std_dev = exponentially_weighted_variance(returns, lambd)
    var = norm.ppf(alpha) * std_dev + mean
    es = mean - std_dev * norm.pdf(norm.ppf(alpha)) / alpha
    return var, es

# 2. a MLE fitted T distribution
def fit_t_distribution(returns):
    """Fit a t-distribution to the data using MLE"""
    params = t.fit(returns)
    return params  # Returns df, loc, scale for the t-distribution

def calculate_var_es_t_dist(returns, alpha=0.05):
    """Calculate VaR and ES under a fitted t-distribution"""
    df, loc, scale = fit_t_distribution(returns)
    var = t.ppf(alpha, df, loc=loc, scale=scale)
    es = t.expect(lambda x: x, args=(df,), loc=loc, scale=scale, lb=-np.inf, ub=var) / alpha
    return var, es

# 3. Historical simulation
def calculate_var_es_historic(returns, alpha=0.05):
    """Calculate VaR and ES using historical simulation"""
    sorted_returns = np.sort(returns)
    index = int(alpha * len(sorted_returns))
    var = sorted_returns[index]
    es = sorted_returns[:index].mean()
    return var, es

# Calculate VaR and ES using each method
var_normal, es_normal = calculate_var_es_normal(data)
var_t, es_t = calculate_var_es_t_dist(data)
var_hist, es_hist = calculate_var_es_historic(data)

#(var_normal, es_normal), (var_t, es_t), (var_hist, es_hist)
print(f"Exponentially weighted variance for normal distribution: VaR={var_normal},ES={es_normal}")
print(f"MLE fitted T distribution: VaR={var_t},ES={es_t}")
print(f"Historic Simulation: VaR={var_hist},ES={es_hist}")

#problem 3


#functions
def ES_norm(mu, sigma, alpha=0.05):
    # Calculate the VaR at the given alpha level using the provided VaR function
    v = VaR_norm(mu, sigma, alpha)
    
    # Define the normal distribution with the specified mean and standard deviation
    d = norm(loc=mu, scale=sigma)
    
    # Define the function to integrate
    f = lambda x: x * d.pdf(x)
    
    # Determine the starting point for integration (close to the minimum of the distribution's support)
    st = d.ppf(1e-12)
    print(st)
    
    # Perform the integration from st to -v
    result, _ = quad(f, st, -v) 
    print(result)
    # Note: the integration bounds have been corrected to 'v' instead of '-v'
    
    # Calculate and return the Expected Shortfall
    return -result / alpha

def VaR_norm(mu, sigma, alpha=0.05):
    return -norm.ppf(alpha, loc=mu, scale=sigma)

# Fit Normal Distribution
def fit_normal(data, lambda_=0.97, method = 'normal'):
    if method == 'ew':
        mu = np.mean(data)
        sigma_ = np.sqrt(ew_cov(data, lambda_=lambda_))
        sigma = sigma_[0][0]
        return mu, sigma
    else:
        mu, sigma = norm.fit(data)
        return mu, sigma
        
def VaR_t(df, loc, scale, alpha = 0.05):
    t_dist = t(df, loc, scale)
    return -t_dist.ppf(alpha)

def ES_t(df, loc, scale, alpha=0.05):
    # Calculate the VaR at the given alpha level using the provided VaR function
    v = VaR_t(df, loc, scale, alpha)
    
    # Define the normal distribution with the specified mean and standard deviation
    d = t(df=df, loc = loc, scale = scale)
    
    # Define the function to integrate
    f = lambda x: x * d.pdf(x)
    
    # Determine the starting point for integration (close to the minimum of the distribution's support)
    st = d.ppf(1e-12)
    print(st)
    
    # Perform the integration from st to -v
    result, _ = quad(f, st, -v) 
    print(result)
    # Note: the integration bounds have been corrected to 'v' instead of '-v'
    
    # Calculate and return the Expected Shortfall
    return -result / alpha

def return_calculate(prices, method="DISCRETE", date_column="Date"):
    if date_column not in prices.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame: {list(prices.columns)}")
    
    # Exclude the date column from the calculation
    vars = [col for col in prices.columns if col != date_column]
    
    # Convert prices to a matrix (numpy array) for calculations
    p = prices[vars].values
    n, m = p.shape
    p2 = np.empty((n-1, m))
    
    # Vectorized calculation for performance
    p2 = p[1:, :] / p[:-1, :]
    
    if method.upper() == "DISCRETE":
        p2 = p2 - 1
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\", \"DISCRETE\")")
    
    # Create a new DataFrame to store the returns along with the corresponding dates
    dates = prices[date_column].iloc[1:].reset_index(drop=True)
    out = pd.DataFrame(data=p2, columns=vars)
    out.insert(0, date_column, dates)
    
    return out

def fit_t(data):
    constraints=({"type":"ineq", "fun":lambda x: x[0]-1}, {"type":"ineq", "fun":lambda x: x[2]})
    returns_t = minimize(MLE_T, x0=[10, np.mean(data), np.std(data, ddof=1).item() ], args=data, constraints=constraints)
    df, loc, scale = returns_t.x[0], returns_t.x[1], returns_t.x[2]
    return df, loc, scale
    
def MLE_T(params, returns):
    negLL = -1 * np.sum(t.logpdf(returns, df=params[0], loc=params[1], scale=params[2]))
    return(negLL)

def multivariate_normal_simulation(covariance_matrix, n_samples, method='direct', fix_method='chol_psd', mean = 0, explained_variance=1.0):
    
    # If the method is 'direct', simulate directly from the covariance matrix
    if method == 'direct' and fix_method=='chol_psd':
        
        L = chol_psd(covariance_matrix)
        normal_samples = np.random.normal(size=(covariance_matrix.shape[0], n_samples))
        
        samples = np.transpose(np.dot(L, normal_samples) + mean)
        
        return samples
    
    # If the method is 'pca', simulate using PCA
    elif method == 'pca' and fix_method=='chol_psd':
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

    elif method == 'direct' and fix_method=='near_psd':
        psd = near_psd(covariance_matrix)
        L = chol_psd(psd)
        normal_samples = np.random.normal(size=(covariance_matrix.shape[0], n_samples))
        
        samples = np.transpose(np.dot(L, normal_samples) + mean)
        
        return samples

    elif method == 'direct' and fix_method == 'higham':
        psd = Higham_psd(covariance_matrix)
        L = chol_psd(psd)
        normal_samples = np.random.normal(size=(covariance_matrix.shape[0], n_samples))
        
        samples = np.transpose(np.dot(L, normal_samples) + mean)
        
        return samples

def VaR_t(df, loc, scale, alpha = 0.05):
    t_dist = t(df, loc, scale)
    return -t_dist.ppf(alpha)
    
def ES_t(df, loc, scale, alpha=0.05):
    # Calculate the VaR at the given alpha level using the provided VaR function
    v = VaR_t(df, loc, scale, alpha)
    
    # Define the normal distribution with the specified mean and standard deviation
    d = t(df=df, loc = loc, scale = scale)
    
    # Define the function to integrate
    f = lambda x: x * d.pdf(x)
    
    # Determine the starting point for integration (close to the minimum of the distribution's support)
    st = d.ppf(1e-12)
    print(st)
    
    # Perform the integration from st to -v
    result, _ = quad(f, st, -v) 
    print(result)
    # Note: the integration bounds have been corrected to 'v' instead of '-v'
    
    # Calculate and return the Expected Shortfall
    return -result / alpha

prices = pd.read_csv('DailyPrices.csv')
# current_prices = prices.iloc[-1,:]
returns_all = return_calculate(prices)
portfolio = pd.read_csv("portfolio.csv")
returns_all.iloc[:, 1:] = returns_all.iloc[:, 1:].apply(lambda x: x - x.mean(), axis=0)
returns_all.drop('Date', axis=1, inplace=True)

stocks = portfolio[portfolio['Portfolio'] == 'A']['Stock']
current_prices = prices.loc[:,stocks].tail(1)

def calculate_portfolio_VaR_ES(portfolio_name, prices, portfolio_info,model ='t'):
    returns_all = return_calculate(prices)
    returns_all.iloc[:, 1:] = returns_all.iloc[:, 1:].apply(lambda x: x - x.mean(), axis=0)
    returns_all.drop('Date', axis=1, inplace=True)

    if portfolio_name.upper() == 'ALL':
        portfolio_holding = portfolio.loc[:,['Stock','Holding']]
        stocks = portfolio['Stock']
    else:      
        portfolio_holding = portfolio[portfolio['Portfolio'] == portfolio_name].loc[:,['Stock','Holding']]
        stocks = portfolio[portfolio['Portfolio'] == portfolio_name]['Stock']
    current_prices = prices.loc[:,stocks].tail(1)
    stocks = stocks.reset_index(drop=True)
    fittedModel = {}
    
    if model == 't':
        for stock in stocks:
            fittedModel[stock] = fit_t(returns_all[stock])
    elif model == 'normal':
        for stock in stocks:
            fittedModel[stock] = fit_normal(returns_all[stock])
    U = pd.DataFrame()
    
    if model == 't':
        for stock, arg in fittedModel.items():
            stock_return = returns_all[stock]
            df, loc, scale = arg
            U[stock] = t.cdf(stock_return,df, loc, scale)
    elif model == 'normal':
        for stock, arg in fittedModel.items():
            stock_return = returns_all[stock]
            mu, sigma = arg
            U[stock] = norm.cdf(stock_return,mu, sigma)

    spcor = spearmanr(U, axis = 0)[0]
    nSim = 1000
    uSim = multivariate_normal_simulation(spcor, nSim,method = 'pca')
    uSim = norm.cdf(uSim,loc=0,scale=1)
    simulatedReturns = pd.DataFrame()
    
    if model == 't':
        for i in range(uSim.shape[1]):
            stock_name = stocks[i]
            df, loc, scale = fittedModel[stock_name]
            simulatedReturns[stock_name] = t.ppf(uSim[:,i],df, loc, scale)
    elif model == 'normal':
        for i in range(uSim.shape[1]):
            stock_name = stocks[i]
            mu, sigma = fittedModel[stock_name]
            simulatedReturns[stock_name] = norm.ppf(uSim[:,i],mu, sigma)
            
    sim_prices = simulatedReturns.mul(current_prices.values.reshape(-1),axis = 1)
    sim_holdings = sim_prices.dot(portfolio_holding['Holding'].values.reshape(-1, 1))
    
    iterations = pd.DataFrame({'iteration': [i for i in range(1, nSim + 1)]})
    values = pd.merge(portfolio_holding, iterations,how='cross')

    nv = len(values)  # Assuming 'values' is a DataFrame as constructed before
    simulatedValue = [0] * nv  # Initialize a list with zeros
    pnl = [0] * nv  # Initialize a list with zeros
    
    for i in range(nv):
        iteration_raw = values.iloc[i]['iteration']
        # Reset iteration to 1 after reaching 100000
        iteration = (iteration_raw % nSim) if iteration_raw == nSim else iteration_raw
        stock = values.iloc[i]['Stock']
        price = prices.loc[0,stock]
        currentValue = values.iloc[i]['Holding']*price
        
        # Ensure that 'simRet' is indexed or accessed correctly; this might need adjustment
        # Assuming 'simRet' has a multi-level index of 'iteration' and 'Stock' or a similar structure
        ret = simulatedReturns.loc[iteration, stock]
        simulatedValue[i] = currentValue * (1 + ret)
        pnl[i] = simulatedValue[i] - currentValue

    values['pnl'] = pnl
    values['simulatedValue'] = simulatedValue

    gdf = values.groupby('iteration')
    totalValues = gdf.aggregate({
        'simulatedValue': 'sum',
        'pnl': 'sum'
    }).reset_index()
    pnl_sum = totalValues['pnl']
    if model == 't':
        Var = VaR_t(fit_t(pnl_sum)[0], fit_t(pnl_sum)[1],fit_t(pnl_sum)[2], alpha=0.05)
        ES = ES_t(fit_t(pnl_sum)[0], fit_t(pnl_sum)[1],fit_t(pnl_sum)[2], alpha=0.05)
    if model == 'normal':
        mu, sigma = fit_normal(pnl_sum)
        Var = VaR_norm(mu, sigma, alpha=0.05)
        ES = ES_norm(mu, sigma, alpha=0.05)

    return Var, ES, sim_holdings

prices = pd.read_csv('DailyPrices.csv')
portfolio = pd.read_csv("portfolio.csv")

var_A, es_A, sim_prices_A= calculate_portfolio_VaR_ES('A', prices, portfolio)
print("For Portfolio A," )
print(f"VaR is {var_A}")
print(f"ES is {es_A}" )
var_B, es_B, sim_prices_B = calculate_portfolio_VaR_ES('B', prices, portfolio)
print("For Portfolio B," )
print(f"VaR is {var_B}")
print(f"ES is {es_B}" )
var_C, es_C, sim_prices_C = calculate_portfolio_VaR_ES('C', prices, portfolio, model ='normal')
print("For Portfolio C," )
print(f"VaR is {var_C}")
print(f"ES is {es_C}" )
var_All, es_All, sim_prices_All = calculate_portfolio_VaR_ES('ALL', prices, portfolio)
print("For Portfolio ALL," )
print(f"VaR is {var_All}")
