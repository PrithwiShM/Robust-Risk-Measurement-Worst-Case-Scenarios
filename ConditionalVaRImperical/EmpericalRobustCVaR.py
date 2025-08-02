# %%
import yfinance as yf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arch import arch_model
from scipy.stats import genpareto
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

class CVaRmodels:
    def __init__(self, rets):
        self._res_bt = rets
        self._losses = np.array(-rets.values, dtype=float)

    def resetperiod(self, i, W):
        self._losses = self._res_bt.iloc[i-W:i].values

    # Historical CVaR
    def cvar_hist(self, alpha=0.95):
        var = np.quantile(self._losses, alpha)          # empirical VaR

        if self._losses[self._losses > var].size == 0:
            return self._losses.max()
        else:
            return self._losses[self._losses > var].mean()  # CVaR = ES

    def cvar_GarchFiltered(self, alpha=0.95, B=10_000):
        ret_pc = - 100 * self._losses
        # 1. Fit the filter --------------------------------------------------------
        am      = arch_model(ret_pc, mean="Constant", vol="GARCH",
                            p=1, q=1, dist='t')
        res     = am.fit(disp="off")
        # 2. Obtain one‑step‑ahead mean & σ ---------------------------------------
        fc      = res.forecast(horizon=1, reindex=False)
        mu_hat  = fc.mean.values[-1, 0]/ 100     # μ̂_{T+1},    # back to raw return units
        sigma2  = fc.variance.values[-1, 0] /100**2       
        sigma_hat = np.sqrt(sigma2)                  # σ̂_{T+1}

        # 3. Build the pool of *standardised* residuals z_t -----------------------
        z = res.std_resid         # (r_t − μ̂_t) / σ̂_t
        
        # 4. Bootstrap: draw B residuals with replacement -------------------------
        rng = np.random.default_rng()
        z_star = rng.choice(z, size=B, replace=True) # {z*}
        
        # 5. Convert to *future* return draws -------------------------------------
        r_star = mu_hat + sigma_hat * z_star         # r*_{T+1}
        # 6. Compute empirical VaR & CVaR of loss ---------------------------------
        var = np.quantile(r_star, alpha)          # empirical VaR_α
        if self._losses[self._losses > var].size == 0:
            index_2 = np.argsort(self._losses)[-2]
            cvar = self._losses[index_2]
        else:
            cvar = self._losses[self._losses > var].mean()    # ES = CVaR_α
        return cvar

    def fit_gpd_cvar(self, alpha=0.99, thresh_q=0.95):
        """CVaR from a POT‑GPD fit.

        thresh_q : percentile to set threshold u (e.g. 0.95)

        Returns
        -------
        var_alpha  : VaR_α
        cvar_alpha : CVaR_α
        """
        u      = np.quantile(self._losses, thresh_q)       # threshold
        exc    = self._losses[self._losses > u] - u              # exceedances, only a subset
        n, Nu  = len(self._losses), len(exc)
        p_u    = Nu / n

        # 1. Fit GPD (ξ = shape, β = scale) on exceedances
        xi, loc, beta = genpareto.fit(exc, floc=0)

        # 2. VaR
        if np.isclose(xi, 0.0): 
            var_alpha = u + beta * np.log(p_u / (1 - alpha))
            cvar_alpha = var_alpha + beta
        else:
            var_alpha = u + beta / xi * (( (1 - alpha) / p_u )**(-xi) - 1)
            if xi >= 1:
                raise ValueError("xi must be <1 for finite CVaR")
            cvar_alpha = u + (var_alpha - u + beta) / (1 - xi)

        return var_alpha, cvar_alpha

    def cvar_dro_kl(self, alpha=0.99, rho=0.05):
        a_star = np.quantile(self._losses, alpha)
        tail   = np.maximum(self._losses - a_star, 0.0)

        def phi(lmb):
            if lmb <= 0:
                return np.inf
            log_mgf = np.log(np.mean(np.exp(lmb * tail)))
            return rho / lmb + log_mgf / (1 - alpha)

        res = minimize_scalar(phi, bracket=(1e-6, 20.0), method="brent")
        return a_star + res.fun             # robust CVaR

    def cvar_dro_w1(self, alpha=0.99, eps=0.002):
        a_star = np.quantile(self._losses, alpha)
        tail_mean = np.maximum(self._losses - a_star, 0.0).mean()
        return a_star + (tail_mean + eps) / (1 - alpha)
    
    def CVaR_Time_Series_Plot(self, alpha= 0.99, rho_kl= 0.05):
        window_years = 6
        days_per_yr  = 252
        W            = window_years * days_per_yr      # rolling window length
        B            = 10_000                          # bootstrap draws for FHS

        records = []
        for i in range(W, len(self._res_bt)):
            t      = self._res_bt.index[i]
            #sample = self._res_bt.iloc[i-W:i].values  # six-year tail sample
            self.resetperiod(i, W)
            l_t    =  -self._res_bt.iloc[i]             # realised loss on date t

            rec = dict(
                date         = t,
                loss         = l_t,
                cvar_hist    = self.cvar_hist(alpha),
                cvar_fhs     = self.cvar_GarchFiltered(alpha=alpha, B=B),
                cvar_gpd_n   = self.fit_gpd_cvar(alpha, 0.95)[1],
                cvar_dro_kl     = self.cvar_dro_kl(alpha, rho=rho_kl),     # VaR & CVaR from GPD
                cvar_dro_w1  = self.cvar_dro_w1(alpha, eps=0.002),
            )
            print(i, end=" ")
            records.append(rec)

        df = pd.DataFrame.from_records(records).set_index("date")
        

        metrics = {}

        for col in df.filter(like="cvar_").columns:
            exceed = df["loss"] >= df[col]
            hit_rate = exceed.mean()                       # unconditional coverage
            
            # Kupiec LR test (H0: true exceedance prob = 1-alpha)
            N      = len(exceed)
            n_exc  = exceed.sum()
            p_hat  = n_exc / N if N else 0.0
            LR     = -2 * ( (n_exc*np.log(1-alpha) + (N-n_exc)*np.log(alpha))
                            - (n_exc*np.log(p_hat) + (N-n_exc)*np.log(1-p_hat)) )
            p_val  = 1 - chi2.cdf(LR, df=1)
            
            # Average gap when the model fails ( realised loss – forecast )
            gap = (df["loss"] - df[col])[exceed].mean()
            
            metrics[col] = dict(hit_rate=hit_rate, kupiec_p=p_val, avg_gap=gap)

        metrics = (
            pd.DataFrame(metrics)
            .T.rename(columns={"hit_rate":"hit-rate", "kupiec_p":"Kupiec p-value",
                                "avg_gap":"avg tail gap"})
        )

        return df, metrics
            

# %%
if __name__ == "__main__":
    # Download daily SPY total-return series
    spy_total_return = yf.download("SPY", start="1999-01-01", end="2009-12-31", auto_adjust=False)
    print("\nSPY Total Return Series:")
    print(spy_total_return.head())

    spy_daily_returns = spy_total_return['Adj Close'].ffill().pct_change().iloc[1:]
    alpha = 0.95
    cVaRmodel = CVaRmodels(spy_daily_returns['SPY'])
    CVaR_historical = cVaRmodel.cvar_hist(0.95)
    print("CVaR historical: {:.3}%".format(CVaR_historical*100))

    CVaR_GarchFiltered = cVaRmodel.cvar_GarchFiltered(0.95)
    print("CVaR GarchFiltered: {:.3}%".format(CVaR_GarchFiltered*100))

    GPD_var, GPD_cvar = cVaRmodel.fit_gpd_cvar(alpha=0.95, thresh_q=0.90)
    print("CVaR GPD: {:.3}%".format(GPD_cvar*100))

    CVaR_dro_kl = cVaRmodel.cvar_dro_kl(alpha=0.95, rho=0.04)
    print(f"CVaR DRO-KL : {CVaR_dro_kl:.3%}")

    CVaR_dro_w1 = cVaRmodel.cvar_dro_w1(alpha=0.95, eps=0.002)
    print(f"CVaR DRO-W1 : {CVaR_dro_w1:8.4%}")

    df, metrics = cVaRmodel.CVaR_Time_Series_Plot(alpha= 0.98)

# %%
    fig = plt.figure()
    #fig.patch.set_facecolor('lightskyblue')
    ax = plt.gca()

    df.plot(ax = ax, figsize=(10, 6))
    plt.title('CVaR Time Series Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(title='Series')
    plt.grid(True)
    # Save the plot to a file
    plt.savefig("CVaR_Time_Series_Plot.png") 
    plt.close() 

    pd.set_option('display.float_format', '{:.3f}'.format)
    print(metrics)
# %%
