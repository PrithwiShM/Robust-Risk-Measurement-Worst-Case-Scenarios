# %%
import pandas as pd
import numpy as np
from datetime import datetime as dt

from numpy.linalg import det, inv, eigvalsh
import matplotlib.pyplot as plt
from MVP_RobustRisk_Calculator import DataFetcher, MeanVarianceModel, RobustModel

# %% 

class MeanVarianceModelE(MeanVarianceModel):
    def portfolio_risk(self, a):
        return -a.T @ self._mu + (self._gamma / 2) * a.T @ self._Sigma @ a
    
class RobustModelE(RobustModel):
    def resetTheta(self, theta):
        self._theta = theta
    
    def portfolio_risk(self, a):
        return -a.T @ self._mu + (self._gamma / 2) * a.T @ self._Sigma_tilda @ a

    def calculateEntropy(self, Sigma_tilda):
        return 0.5*(np.log(det(self._Sigma @ inv(Sigma_tilda) )) + np.trace(self._invSigma @ Sigma_tilda - self._I))


# %%
if __name__ == "__main__":
    # Yfinance method to find the most traded stocks
    # List of stock tickers
    tickers =["AAPL", "MSFT", "AMZN", "FB", "GOOGL", "BRK-B", "JNJ",
    "WMT", "V", "PG", "NVDA", "JPM", "UNH", "HD", "MA", "DIS", "PFE", "VZ", 
    "INTC", "CMCSA", "T", "KO", "MRK", "PEP", "ABT", "CVX", "ORCL", "CSCO", 
    "NFLX", "XOM", "IBM", "NKE", "CRM", "ADBE", "AVGO", "GS", "QCOM", "BA", 
    "PYPL", "TXN", "AXP", "AMD", "C", "MS", "CAT", "HON", "SBUX", "COST", "MCD"] 
    # 


    # ['AVGO', 'V', 'FB', 'PYPL']: YFPricesMissingError('possibly delisted; no price data found
    # 'CRM', 'FB', 'GOOGL', 'MA' have null values
    bad_data = ['AVGO', 'PYPL', 'V', 'CRM', 'FB', 'GOOGL', 'MA']

    fetcher = DataFetcher(tickers)
    daily_returns = fetcher.fetch_returns(
        dt(1999, 1, 1),
        dt(2009, 12, 31),
        drop_tickers=bad_data,
    )    


    # %%
    #training and testing dates

    training_period_start = "1999-01-01"
    training_period_end = "2007-12-31"
    test_period_end = "2009-12-31"

    #  Calculate the mean and covariance of daily returns for the first 12 years
    h_data = daily_returns.loc[training_period_start:training_period_end].iloc[1:]
    testing_data = daily_returns.loc[training_period_end:test_period_end]

    n = daily_returns.shape[1]

    # %% [markdown]
    # ## Nominal Case
    gamma = 2.0

    nominal = MeanVarianceModelE(gamma=gamma)
    nominal.fitMeanVariance(h_data)
    a_hat = nominal.NominalOptimalPortfolio()

    risk_nominal = nominal.portfolio_risk(a_hat)
    print("Performance in Nominal Model:", risk_nominal)

    # %% [markdown]
    # ## Worst Case Portfolio

    theta_ = 200
    # Calculate worst-case covariance matrix
    robust = RobustModelE(gamma=gamma, theta=theta_)
    robust.fitMeanVariance(h_data)

    initial_weights = [np.ones(n)/n]  #robust.GenerateInitialGuesses(5)
    a_star = robust.constraintOptimization(initial_weights)
    Sigma_tilda = robust.worst_case_covariance_matrix(a_star)

    # Calculate performance in the worst-case model
    risk_worstcase = robust.portfolio_risk(a_hat)
    print("Performance of NP in Worst-Case Model:", risk_worstcase)

    Rel_entropy = robust.calculateEntropy(Sigma_tilda)

    RPs_nominal = nominal.portfolio_risk(a_star)
    RPs_worstcase = robust.portfolio_risk(a_star)
    # Both of them are the same

    print('RPs_nominal ', RPs_nominal)
    print('RPs_worstcase ', RPs_worstcase)
    print('Rel_entropy ', Rel_entropy)

    # %%
    results_ = {}

    thetas = np.exp(np.linspace(0, 6, 40))
    for theta_ in thetas:
        
        #computaion of entropy, risk_measurement
        #mvo = mean variance objective
        robust.resetTheta(theta_)
        a_star = robust.constraintOptimization(initial_weights )
        Sigma_tilda = robust.worst_case_covariance_matrix(a_star)
        
        Rel_entropy = robust.calculateEntropy(Sigma_tilda)
        
        RPs_nominal = nominal.portfolio_risk(a_star)
        NPs_worstcase = robust.portfolio_risk(a_hat)
        RPs_worstcase = robust.portfolio_risk(a_star)

        results_[theta_] = (Rel_entropy, NPs_worstcase, RPs_nominal, RPs_worstcase)

    # %%
    # Convert the dictionary to a list of tuples
    data_list = [(key, *value) for key, value in results_.items()]

    # Create a DataFrame
    df_wc = pd.DataFrame(data_list, columns=['Theta', 'Entropy', 'NPs_worstcase', 'RPs_nominal',  'RPs_worstcase'])

    # No bounds no linear constraints just simple results
    fig = plt.figure()
    #fig.patch.set_facecolor('lightskyblue')
    ax = plt.gca()
    ax.set_facecolor('lightskyblue')
    ax.axhline(y = risk_nominal, color = 'b', linestyle = (0, (3, 5, 1, 5)))  # NP Nominal
    df_wc.plot(ax = ax, x='Entropy',y='RPs_nominal', linestyle=(0, (5, 5)), linewidth=1.5, color='green') # RP Nominal

    df_wc.plot(ax = ax, x='Entropy',y='RPs_worstcase', linestyle='dotted', linewidth=2, color='darkorange') # RP Robust
    df_wc.plot(ax = ax, x='Entropy',y='NPs_worstcase', linestyle='solid', linewidth=1.5, color='blue') # NP Robust

    # Save the plot to a file
    plt.savefig("EntropyvsRiskCurve.png") 
    plt.close() 
    # %%
    real = MeanVarianceModelE(gamma=gamma)
    real.fitMeanVariance(testing_data)

    Realized_risk = real.portfolio_risk(a_hat)
    print('Realised_portfolio_risk: ', "{:.2e}".format(Realized_risk))

    # %%
