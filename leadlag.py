## Lead-Lag metrics

import distcorr as corr
import numpy as np


def ccf(x, y, lags=None):
    """
    Cross-correlation function
    assume x,y are already loaded in for lookback window length
    """
    if lags is None:
        lags = np.arange(-len(x) + 1, len(y))
    
    ccf_values = []
    for lag in lags:
        if lag < 0: #this refers to x leading y
            cross_corr = corr.distance_correlation(x[:lag], y[-lag:]) #can modify this to use a different correlation measure
            ccf_values.append(cross_corr)
        elif lag == 0: #no lag
            cross_corr = corr.distance_correlation(x, y)
            ccf_values.append(cross_corr)
        else: #this refers to y leading x
            cross_corr = corr.distance_correlation(x[lag:], y[:-lag])
            ccf_values.append(cross_corr)
    
    return np.array(ccf_values)

def ccf_auc(x, y, max_lag):
    """
    Compute area under cross-correlation function over positive and negative lags,
    where negative lag means x leads y, positive lag means y leads x.

    Parameters:
    - x, y: 1D arrays of returns
    - max_lag: max number of lags to consider (positive integer)

    Returns:
    - auc: a signed measure in [-1, 1] indicating who leads whom.
      Positive means x leads y; negative means y leads x.
    """
    # Lags from -max_lag to max_lag, excluding 0
    lags = list(range(-max_lag, max_lag + 1))

    ccf_vals = ccf(x, y, lags)

    # Sum of absolute correlations for negative lags (x leads y)
    I_xy = sum(abs(ccf_vals[i]) for i, lag in enumerate(lags) if lag < 0)
    # Sum of absolute correlations for positive lags (y leads x)
    I_yx = sum(abs(ccf_vals[i]) for i, lag in enumerate(lags) if lag > 0)

    max_I = max(I_xy, I_yx)
    auc = np.sign(I_xy - I_yx) * (max_I / (I_xy + I_yx))
    return auc


def compute_lead_lag_matrix(assets, lag):
    """
    Compute the skew-sym lead-lag matrix for a set of assets
    """
    n = len(assets)
    lead_lag_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                lead_lag_matrix[i, j] = ccf_auc(assets[i], assets[j], lag)
                #print(ccf_auc(assets[i], assets[j], lag))
    
    return lead_lag_matrix

