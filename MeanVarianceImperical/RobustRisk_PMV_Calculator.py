
# %%
import pytz
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime as dt
import time

from numpy.linalg import det, inv, eigvals
from scipy.optimize import minimize, LinearConstraint, Bounds

class DataFetcher:
    """Download price data and return daily percentage returns."""
    def __init__(self, tickers, tz="America/New_York"):
        self.tickers = tickers
        self.tz = pytz.timezone(tz)

    def fetch_returns(
        self, start_dt: dt, end_dt: dt,
        drop_tickers=None,
    ) -> pd.DataFrame:
        start = self.tz.localize(start_dt)
        end = self.tz.localize(end_dt)

        # Download the stock price data
        data = yf.download(self.tickers, start=start, end=end, auto_adjust=False).xs('Adj Close', level=0, axis=1)

        data = data.drop(columns=drop_tickers) # renamed Tickers, very recenlty released IPOs

        daily_returns = data.pct_change(fill_method='ffill')
        return daily_returns

# NOMINAL MEAN–VARIANCE MODEL
class MeanVarianceModel:
    def __init__(self, gamma: float):
        self._gamma = gamma
        self.fitted = False

    def fitMeanVariance(self, data):
        self._mu = data.mean().to_numpy()
        self._Sigma = data.cov().to_numpy()
        self.n = self._mu.size
        self.fitted = True

    def NominalOptimalPortfolio(self)-> np.ndarray:
        # Vector of ones
        if not self.fitted:
            raise RuntimeError("call .fit() first")
        ones = np.ones(self.n)

        # Note here, there is sum constraint, but no bound constraints
        # Calculate (gamma * Sigma)^-1
        gamma_sigma_inv = inv(self._gamma * self._Sigma)

        # Calculate lambda
        lambda_val = (ones.T @ gamma_sigma_inv @ self._mu - 1) / (ones.T @ gamma_sigma_inv @ ones)

        # Calculate optimal portfolio weights
        a = gamma_sigma_inv @ (self._mu - lambda_val * ones)
        return a

    def calculateVariance(self, _a, days):
        variance = _a.T @ self._Sigma @ _a
        
        std_error = np.sqrt((2 * _a.T @ self._Sigma @ self._Sigma @ _a) / days) 

        return variance, std_error

# ROBUST MODEL
class RobustModel(MeanVarianceModel): # Inheritance
    def __init__(self, gamma: float, theta: float):
        super().__init__(gamma)
        self._theta = theta

    def objective_a_oftheta(self, a: np.ndarray):
            I = np.eye(self.n)
            term1 = 1 / np.sqrt(det(I - self._theta * self._gamma * np.outer(a,a) @ self._Sigma))
            return (1/self._theta)*np.log(term1) + a.T @ self._mu

    def positive_definite_constraint(self, a: np.ndarray):
        eigenvalues = eigvals(inv(self._Sigma) - self._theta * self._gamma * np.outer(a,a) )
        return np.min(eigenvalues) - 1e-10  # Ensure all eigenvalues are slightly greater than zero


    def GenerateInitialGuesses(self, _n_points):
        points_on_hyperplane = np.zeros((_n_points, self.n))

        for j in range(_n_points):
            random_numbers = np.random.rand(self.n)
            # Normalize the numbers so their sum is 10
            normalized_numbers = random_numbers / random_numbers.sum() 

            points_on_hyperplane[j] = normalized_numbers

        # Initial guesses for 'a'
        initial_guesses_if = []
        for a0 in points_on_hyperplane:
            initial_guesses_if.extend([np.random.permutation(a0) for _ in range(100)])
        initial_guesses = [aa for aa in initial_guesses_if if self.positive_definite_constraint(aa)>0]

        print("Generated total ", len(initial_guesses), " Initial Guesses")
        return initial_guesses

    def constraintOptimization(self, _initial_guesses):
        start_time = time.time()

        #extra constraints
        ones_for_sum = np.ones((1, self.n))
        linear_constraint = LinearConstraint(ones_for_sum, [1], [1])
        bounds = Bounds(0, 1)
        # Initialize variables to track the best result
        best_result = None

        # Iterate over each initial guess
        for ax in _initial_guesses:

            # Define constraints in the format required by 'minimize'
            constraints = [{'type': 'ineq', 'fun': self.positive_definite_constraint, 'args': ()},
                        linear_constraint]
            
            # Perform the optimization
            result = minimize(self.objective_a_oftheta, ax, method='SLSQP', constraints=constraints, bounds=None, args = ())
            if ~result.success:
                print("Optimization failed:", result.message)
            # Update the best result if necessary
            if best_result is None or result.fun < best_result.fun:
                best_result = result
                
        # Check if the optimization was successful
        if best_result.success:
            a_star = best_result.x
            print("Optimized a:", a_star)
            print(" Minimum objective reached at", np.real(best_result.fun), "Not portfolio objective!")
        else:
            print("Optimization failed:", best_result.message)

        print("--- %s Optimization time :seconds ---" % (time.time() - start_time))
        return a_star, best_result

    def worst_case_covariance_matrix(self,  a):
        inv_Sigma_worst_case = inv(self._Sigma) - self._theta * self._gamma * np.outer(a, a)
        return inv(inv_Sigma_worst_case)

    def variance_and_se_from_cov(self, a, _Sigma, n_days):
        var = a.T @ _Sigma @ a
        se  = np.sqrt((2 * a.T @ _Sigma @ _Sigma @ a) / n_days)
        return var, se
# %%

if __name__ == "__main__":
    # Define the stock tickers and the time period
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JNJ",
    "WMT", "V", "PG", "NVDA", "JPM", "UNH", "HD", "MA", "DIS", "PFE", "VZ", 
    "INTC", "CMCSA", "T", "KO", "MRK", "PEP", "ABT", "CVX", "ORCL", "CSCO", 
    "NFLX", "XOM", "IBM", "NKE", "CRM", "ADBE", "AVGO", "GS", "QCOM", "BA", 
    "PYPL", "TXN", "AXP", "AMD", "C", "MS", "CAT", "HON", "SBUX", "COST", "MCD"]  # Replace with S&P 500 tickers
    fetcher = DataFetcher(tickers)
    daily_returns = fetcher.fetch_returns(
        dt(1996, 1, 1),
        dt(2020, 12, 31),
        drop_tickers=["PYPL", "MA", "AVGO", "TSLA", "META", "V"],
    )    
    # %%
    #training and testing dates
    flag = 2008

    if flag == 2008:
        training_period_start = "1996-01-02" # +1, As there is no data on the first day
        training_period_end = "2007-12-31"
        test_period_end = "2008-12-31"
    else:
        training_period_start = "2009-01-01"
        training_period_end = "2019-12-31"
        test_period_end = "2020-12-31"

    # %%
    # Calculate the mean and covariance of daily returns for the first 12 years
    training_data = daily_returns.loc[training_period_start:training_period_end]
    testing_data = daily_returns.loc[training_period_end:test_period_end]
    
    # Risk aversion parameter (gamma)
    gamma = 10.0

    nominal = MeanVarianceModel(gamma=gamma)
    nominal.fitMeanVariance(training_data)
    a = nominal.NominalOptimalPortfolio()
    no_training_days = training_data.shape[0]
    forcasted_variance, std_error_fv = nominal.calculateVariance(a, no_training_days)
    print('forcasted_variance: ', "{:.2e}".format(forcasted_variance))
    print('2x Std error: ', "{:.2e}".format(2*std_error_fv))
    
   
     # %%
    #check realized variance
    real = MeanVarianceModel(gamma=gamma)
    real.fitMeanVariance(testing_data)

    no_working_days = testing_data.shape[0]
    Realized_variance, std_error_rv= real.calculateVariance(a, no_working_days)
    print('Realised_variance: ', "{:.2e}".format(Realized_variance))

    # %%
    #specific case for analysis
    theta = 700
    robust = RobustModel(gamma=gamma, theta=theta)
    robust.fitMeanVariance(training_data)
    initial_guesses = robust.GenerateInitialGuesses(2)
    a_star, best_result = robust.constraintOptimization(initial_guesses)
   
    Sigma_tilda = robust.worst_case_covariance_matrix(a_star)
    worst_variance, std_error_wv = robust.variance_and_se_from_cov(a, Sigma_tilda, no_training_days)
    print('worst_forcasted_variance ', "{:.2e}".format(worst_variance))
    print('std_error_wv ', "{:.2e}".format(std_error_wv))

    Model_Error = np.abs(forcasted_variance - worst_variance)
    print('Model_Error ', "{:.2e}".format(Model_Error))

    print('forcasted variance with both errors in confidence interval-')
    t_err = 2*(std_error_wv+Model_Error)
    print('(',"{:.2e},".format(worst_variance-t_err), "{:.2e}".format(worst_variance+t_err), ')')

    # %%
    print("Increase in std error: ", "{:.2f}%".format((t_err - std_error_fv)*100/std_error_fv))

# %%
