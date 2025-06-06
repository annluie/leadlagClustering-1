import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, csc_matrix
from signet.cluster import Cluster

def compute_daily_return_matrix(df, window_dates):
    """
    Computes a matrix of daily log returns within a given window.

    Returns:
    - DataFrame with tickers as rows and dates as columns
    """
    df = df.copy()
    df = df[df['date'].isin(window_dates)]
    df = df[df['prevAdjClose'] > 0]

    df = df.sort_values(['ticker', 'date'])
    df['log_price'] = np.log(df['prevAdjClose'])

    # Compute daily returns 
    df['log_return'] = df.groupby('ticker')['log_price'].diff()

    df = df.dropna(subset=['log_return']) ##CHECK THIS -- Would first day be 0 because of the way this is done? might need an extra day in windowdates array (note: i think this is fixed)

    #I think(?) this solves the problem of the first day having an NaN value
    actual_window_dates = sorted(window_dates)[1:]
    df = df[df['date'].isin(actual_window_dates)]

    ##TODO: substract off the R_SPY term

    return df.pivot(index='ticker', columns='date', values='log_return')

def build_signed_graph(R, threshold=0.3):
    corr = R.T.corr().fillna(0)
    np.fill_diagonal(corr.values, 0)
    Ap = csc_matrix(corr.values * (corr.values >= threshold))
    An = csc_matrix(-corr.values * (corr.values <= -threshold))
    return Ap, An

def sponge_clustering(Ap, An, k=5, tau_p=1, tau_n=1):
    clusterer = Cluster((Ap, An))
    labels = clusterer.SPONGE(k=k, tau_p=tau_p, tau_n=tau_n)
    return labels

## Step 2 of the pipeline -- to be called during run the rolling sponge pipleine

def compute_synthetic_etfs(R, clustering_result):
    """
    Create synthetic ETFs for each cluster on a given date.
    
    Parameters:
    - R: pandas DataFrame (tickers × dates) of returns for the current window
    - clustering_result: dict with 'date' and 'clusters' from run_rolling_sponge_clustering
    
    Returns:
    - dict mapping cluster ID ([k]) to synthetic ETF time series
    """
    etfs = {}
    date = clustering_result['date']
    clusters = clustering_result['clusters']

    for label, tickers in clusters.items():
       # if len(tickers) < 2: ##might not be necessary
       #     continue  # Skip singleton clusters (ETF is trivial or meaningless)

        # Submatrix of returns for the cluster
        cluster_returns = R.loc[tickers]

        # Each row is an asset, columns are time. compute centroid as numpy array for broadcasting
        centroid = cluster_returns.mean(axis=0, skipna=True)
        centroid = centroid.values

        # Euclidean distance of each asset to the centroid
       # distances = np.linalg.norm(cluster_returns.values - centroid.values, axis=1)
        #The diffs line is to treat the NANs as 0s just for the sake of distance calculation. This could use work because data we dont have is being treated as "important" i.e. close to center 
        #TODO Fix the way this is handled -- I don't like that an asset with a lot of missing data will have a higher weight than other assets
        diffs = np.nan_to_num(cluster_returns.values - centroid)
        distances = np.linalg.norm(diffs, axis=1)       

        # Convert to weights: inverse distance, avoid div by 0. less distance to center implies stronger weights.
        epsilon = 1e-8
        inv_distances = 1 / (distances + epsilon)
        weights = inv_distances / inv_distances.sum() #normalize

        # Weighted average across assets → synthetic ETF
        #synthetic_etf = np.average(cluster_returns.values, axis=0, weights=weights)
        ##converts NaN values to 0 so they won't impact the overall value of the average return
        synthetic_etf = np.average(np.nan_to_num(cluster_returns.values), axis=0, weights=weights)

        etfs[label] = pd.Series(synthetic_etf, index=cluster_returns.columns)


    return {
        'date': date,
        'synthetic_etfs': etfs
    }

def run_sponge_clustering_window(df, start_date, end_date, threshold=0.3, k=5):
    # Ensure datetime format
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Drop duplicates and sort unique dates
    unique_dates = df['date'].drop_duplicates().sort_values().reset_index(drop=True)

    # Find the index of start_date and go one day earlier
    start_idx = unique_dates[unique_dates >= pd.to_datetime(start_date)].index[0]
    adjusted_start_date = unique_dates[max(0, start_idx - 1)]

    # Filter data between the two dates
    #mask = (df['date'] >= adjusted_start_date) & (df['date'] <= pd.to_datetime(end_date))
    #window_df = df[mask]

    # Extract unique dates in the window (sorted)
    window_dates = unique_dates[(unique_dates >= adjusted_start_date) & (unique_dates <= pd.to_datetime(end_date))].reset_index(drop=True)


    # Compute the return matrix for the entire window
    R = compute_daily_return_matrix(df, window_dates)
    tickers = R.index.tolist()

    if R.shape[1] < 2:
        raise ValueError("Not enough time points to compute return correlations.")

    # Build signed graph and run clustering
    Ap, An = build_signed_graph(R, threshold=threshold)
    labels = sponge_clustering(Ap, An, k=k)

    # Build cluster dictionary
    cluster_dict = {}
    for ticker, label in zip(tickers, labels):
        cluster_dict.setdefault(label, []).append(ticker)

    clustering_result = {
        'date': pd.to_datetime(end_date),  # Or just a placeholder
        'clusters': cluster_dict
    }

    synthetic_result = compute_synthetic_etfs(R, clustering_result)
    etfs = synthetic_result['synthetic_etfs']  # dict: cluster_id -> pd.Series

    # Build and return k × T DataFrame
    etf_df = pd.DataFrame.from_dict(etfs, orient='index')  # clusters as rows
    return etf_df
