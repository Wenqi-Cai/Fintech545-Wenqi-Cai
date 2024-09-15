import pandas as pd
import numpy as np

# problem 1

#1.1
df = pd.read_csv('problem1.csv')
data = df['x']
n = len(data)
mean_1 = sum(data) / n
variance_1 = sum((x - mean_1)**2 for x in data) / (n-1)
std_1 = variance_1**0.5
skewness_1 = sum(((x - mean_1)/std_1)**3 for x in data) * n / ((n-1)*(n-2))
skewness_1_nor = skewness_1 / (std_1**3)
kurtosis_1 = np.mean(((data - mean_1) / std_1) ** 4)
kurtosis_1_nor = (n * (n + 1) * kurtosis_1 - 3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
print(f"mean: {mean_1}")
print(f"varience: {variance_1}")
print(f"normalized_skewness: {skewness_1_nor}")
print(f"normalized_kurtosis: {kurtosis_1_nor}")

#1.2
import scipy
from scipy.stats import skew, kurtosis
mean = data.mean()
variance = data.var()
std_data = (data - mean)/ data.std()
skewness = skew(std_data)
kurtosis = kurtosis(std_data, fisher=False)
print(f"Mean : {mean}")
print(f"Variance : {variance}")
print(f"Skewness : {skewness}")
print(f"Kurtosis : {kurtosis}")


# problem 2
import numpy as np
import pandas as pd
import statsmodels.api as sm
df2 = pd.read_csv('problem2.csv')
#print(df2.head())

#1/OLS
Y = df2['y']
X = df2['x']
X = sm.add_constant(X)  
#print(X.shape)
ols_model = sm.OLS(Y,X).fit()  
print(ols_model.summary())  
print(f"ols_beta:{ols_model.params}") 
print(f"ols_stderror:{ols_model.resid.std()}") 

#2/MLE given the assumption of normality
def negative_log_likelihood(params): 
	beta = params[:-1]  
	sigma = params[-1]
	n = len(Y)
	residual = Y - X@beta
	nll = n * np.log(sigma * np.sqrt(2 * np.pi)) + np.sum(residual**2) / (2 * sigma**2)
	
	return nll
initial_params = np.array([0.0, 0.0, 1.0])
from scipy.optimize import minimize
mle_result = minimize(negative_log_likelihood, initial_params, method='BFGS')
mle_beta = mle_result.x[:-1]
mle_sigma = mle_result.x[-1]

print(f"MLE Beta Coefficients:{mle_beta}")
print(f"MLE Sigma (Standard Deviation): {mle_sigma}")

print("\nComparison:")
print(f"Difference in Beta Coefficients: {ols_model.params - mle_beta}")
print(f"Difference in Standard Deviation: {ols_model.resid.std() - mle_sigma}")



#3/MLE given the assumption of a T distribution of errors
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t

df2 = pd.read_csv('problem2.csv')

Y = df2['y']
X = df2['x'].values.reshape(-1, 1)

def negative_log_likelihood_t(params):
    beta = params[0]  
    sigma = params[1]
    nu = params[2]
    
    if sigma <= 0:
        return np.inf

    residual = Y - X.flatten() * beta 
    nll_t = -np.sum(t.logpdf(residual, df=nu, scale=sigma)) 
    return nll_t

initial_params_t = np.array([0.0, 1.0, 2.0])

t_result = minimize(negative_log_likelihood_t, initial_params_t, method='BFGS')

t_beta = t_result.x[0]
t_sigma = t_result.x[1]
t_nu = t_result.x[2]

print(f"MLE-t Beta Coefficient: {t_beta}")
print(f"MLE-t Sigma (Standard Deviation): {t_sigma}")
print(f"MLE-t Nu (Degrees of Freedom): {t_nu}")


#AIC
Y = df2['y']
X = df2['x']
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
model_normal = ARIMA(Y, exog=X, order=(1, 0, 0)) 
results_normal = model_normal.fit()
print(f'Normal Distribution Model AIC: {results_normal.aic}')
log_likelihood_t=-t_result.fun
n = len(Y)
aic_t = 2 * len(initial_params) - 2 * log_likelihood_t
print(f'T Distribution Model AIC: {aic_t}')

#2.3
#Fit a multivariate distribution
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
df = pd.read_csv('problem2_x.csv')
X1 = df['x1'].values
X2 = df['x2'].values
mean = [np.mean(X1), np.mean(X2)]
cov = np.cov(X1, X2)
mvn = multivariate_normal(mean=mean, cov=cov) #a multivariate distribution

#the conditional distributions
cov_11 = cov[0, 0]  
cov_22 = cov[1, 1]  
cov_12 = cov[0, 1]  
print(f'cov_11: {cov_11}')
print(f'cov_12: {cov_12}')
print(f'cov_22: {cov_22}')

def con_dis(x1_value):
	conditional_mean = mean[1] + cov_12 / cov_11 * (x1_value - mean[0]) 
	conditional_cov = cov_22 - cov_12**2 / cov_11  
	return conditional_mean, conditional_cov
conditional_means = []
conditional_stds = []

for x1 in X1:
    mu_cond, sigma_cond = con_dis(x1) 
    conditional_means.append(mu_cond)
    conditional_stds.append(np.sqrt(sigma_cond))

#plot
import matplotlib.pyplot as plt

# 
confidence_interval = 1.96  
upper_bound = np.array(conditional_means) + confidence_interval * np.array(conditional_stds)
lower_bound = np.array(conditional_means) - confidence_interval * np.array(conditional_stds)

#
plt.figure(figsize=(12, 6))
plt.scatter(X1, X2, color='red', label='Observed Values') 
plt.plot(X1, conditional_means, color='blue', label='Expected Value') 
plt.fill_between(X1, lower_bound, upper_bound, color='lightblue', alpha=0.5, label='95% Confidence Interval')  
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Conditional Distribution of X2 given X1')
plt.legend()
plt.show()

#2.4 
df2 = pd.read_csv('problem2.csv')
Y = df2['y']
X = df2['x']
X = X.to_numpy().reshape(-1, 1)  
Y = Y.to_numpy().reshape(-1, 1) 
#print(Y.shape)
#print(X.shape)
def mle_estimation(Y, X):
	beta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y
	residuals = Y - X @ beta_hat
	sigma2_hat = np.sum(residuals**2) / len(Y)
	return beta_hat, np.sqrt(sigma2_hat)
beta_hat, sigma_hat = mle_estimation(Y, X)
print("Estimated Beta:", beta_hat)
print("Estimated Sigma:", sigma_hat)

#problem3
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
data = pd.read_csv('problem3.csv')

#ACF
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(data, lags=40, ax=plt.gca(),color='red', title='ACF')
plt.xlabel('Lags')
plt.ylabel('ACF')

#PACF
plt.subplot(1, 2, 2)
plot_pacf(data, lags=40, ax=plt.gca(),color='red', title='PACF')
plt.xlabel('Lags')
plt.ylabel('PACF')

plt.tight_layout()
plt.show()
#AR
def fit_ar_model(data, p):
    model = ARIMA(data, order=(p, 0, 0))  # AR(p) 
    results = model.fit()
    return results
p_values = [1, 2, 3]
ar1 = AutoReg(data, lags=1).fit()
ar2 = AutoReg(data, lags=2).fit()
ar3 = AutoReg(data, lags=3).fit()
print("AR1 Coefficients:", ar1.params)
print("AR2 Coefficients:", ar2.params)
print("AR3 Coefficients:", ar3.params)
#MA
def fit_ma_model(data, q):
    model = ARIMA(data, order=(0, 0, q))  # MA(q) 
    results = model.fit()
    return results
q_values = [1, 2, 3]
ma1 = ARIMA(data, order=(0, 0, 1)).fit()  
ma2 = ARIMA(data, order=(0, 0, 2)).fit()
ma3 = ARIMA(data, order=(0, 0, 3)).fit()
print("MA Coefficients:", ma1.params)
print("MA Coefficients:", ma2.params)
print("MA Coefficients:", ma3.params)

aic_values_ar = []
for p in p_values:
    results = fit_ar_model(data, p)
    aic_values_ar.append(results.aic)
    print(f'AR({p}) Model AIC: {results.aic}')
    
aic_values_ma = []
for q in q_values:
    results = fit_ma_model(data, q)
    aic_values_ma.append(results.aic)
    print(f'MA({q}) Model AIC: {results.aic}')

