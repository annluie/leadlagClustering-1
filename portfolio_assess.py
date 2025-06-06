
# # Portfolio assessment 

# %%
################
# Import dependencies
################
import numpy as np
import pandas as pd

# %% [markdown]
# # Performance metrics

# %% [markdown]
# ## Future Returns and Market Excess Returns
# 

# %%
def get_price(instrument, time, df):
    """
    Get the historical prices of an instrument at a specified time
    
    Parameters:
    instrument (str): The name of the financial instrument.
    time (str): The current time or date index.
    df (dataframe) : Dataframe containing the historical prices of all instruments.
    
    Returns:
    float: The price of the instrument at the specified time and horizon.
    """
    asset = df.loc((df['ticker'] == instrument) & (df['date'] == time))
    return asset['prevAdjclose'] if not asset.empty else None


def get_future_time(time, horizon):
    """
    Get the future time based on the current time and a specified horizon.
    
    Parameters:
    time (str): The current time or date index.
    horizon (int): The number of days into the future to look.
    
    Returns:
    str: The future time as a string.
    """
    return (pd.to_datetime(time) + pd.tseries.offsets.BDay(horizon)).date()


#################### Future returns ####################
def fret_log(instrument, time, horizon, df):
    """
    Calculate the future returns of an instrument over a specified time horizon.
    
    Parameters:
    instrument (str): The name of the financial instrument.
    time (str): The current time or date index.
    future_time (str): The future time or date index.
    df (Dataframe) : Dataframe containing the historical prices of all instruments.
    Returns:
    float: The calculated log raw return for the instrument at the specified time and horizon.
    """
    p_current = get_price(instrument, time, df)
    future_time = get_future_time(time, horizon)
    p_future = get_price(instrument, future_time, df)
    return np.log(p_future / p_current) if p_current and p_future else None


def fret_log_df(time, horizon, df):
    """
    Calculate the future returns of all instruments over a specified time horizon.
    
    Parameters:
    time (str): The current time or date index.
    future_time (str): The future time or date index.
    df (Dataframe) : Dataframe containing the historical prices of all instruments.
    
    Returns:
    Dataframe: The calculated raw return for all instruments at the specified time and horizon, where columns are tickers and dates are indices.
    """
    future_time = get_future_time(time, horizon)
    returns = {}
    for instrument in df['ticker'].unique():
        returns[instrument] = fret_log(instrument, time, future_time, df)
    return pd.DataFrame.from_dict(returns, index=pd.to_datetime([time]))


def fret_log_m(instrument, time, horizon, df):
    """
    Calculate the future market excess returns of an instruments over a specified time horizon.
    Parameters:
    instrument (str): The name of the financial instrument.
    time (str): The current time or date index.
    future_time (str): The future time or date index.
    df (Dataframe) : Dataframe containing the historical prices of all instruments.
    Returns:
    float: The calculated market excess return for the instrument at the specified time and horizon.
    """
    future_time = get_future_time(time, horizon)
    p_current = get_price(instrument, time, df)
    p_future = get_price(instrument, future_time, df)
    if p_current and p_future:
        return np.log(p_future / p_current) - np.log(get_price('SPY', future_time, df) / get_price('SPY', time, df))
    return None

def fret_log_m_df(time, horizon, df):
    """
    Calculate the future market excess returns of all instruments over a specified time horizon.
    
    Parameters:
    time (str): The current time or date index.
    future_time (str): The future time or date index.
    df (Dataframe) : Dataframe containing the historical prices of all instruments.
    
    Returns:
    Dataframe: The calculated market excess return for all instruments at the specified time and horizon.
    """
    future_time = get_future_time(time, horizon)
    returns = {}
    for instrument in df['ticker'].unique():
        returns[instrument] = fret_log_m(instrument, time, future_time, df)
    return pd.DataFrame.from_dict(returns, index=pd.to_datetime([time]))


# %% [markdown]
# ## PnL and Sharpe Ratio
# 


# %%
### PnL returns
def pnl(s,time, horizon, df):
    """
    Calculate the profit and loss (PnL) over a specified time horizon (horizon = rebalance freq).
    Parameters:
    s (Dataframe): Dataframe of weighted signals for all instruments 
    time (str): The current time or date index.
    future_time (str): The future time or date index.
    df (Dataframe) : Dataframe containing the historical prices of all instruments.
    
    Returns:
    float: The calculated PnL return for the instrument at the specified time and horizon.
    """
    #future_time = get_future_time(time, horizon)
    prices = fret_log_df(time, horizon, df)  # Get current prices
    pnl_return = (s.loc[time] * prices.loc[time]).sum()
    return pnl_return

def pnl_timeseries(s,horizon, df):
    """
    Calculate the profit and loss (PnL) over a specified time horizon for all instruments.
    
    Parameters:
    s (Dataframe): Dataframe of weighted signals for all instruments 
    horizon (int): The number of days into the future to look.
    df (Dataframe) : Dataframe containing the historical prices of all instruments.
    
    Returns:
    Series: The calculated PnL return for all instruments at each time index.
    """
    times = df['date'].unique()
    pnl_series = pd.Series(index=times, dtype=float)

    for time in times:
        pnl_series[time] = pnl(s, time, horizon, df)
    
    return pnl_series

def pnl_m(s, time, horizon, df):
    """
    Calculate the profit and loss (PnL) over a specified time horizon with market excess returns.
    
    Parameters:
    s (Dataframe): Dataframe of weighted signals for all instruments 
    time (str): The current time or date index.
    future_time (str): The future time or date index.
    df (Dataframe) : Dataframe containing the historical prices of all instruments.
    
    Returns:
    float: The calculated PnL return for the instrument at the specified time and horizon with market excess returns.
    """
    #future_time = get_future_time(time, horizon)
    prices = fret_log_m_df(time, horizon, df)  # Get current prices
    pnl_return = (s.loc[time] * prices.loc[time]).sum()
    return pnl_return

def pnl_m_timeseries(s, horizon, df):
    """
    Calculate the profit and loss (PnL) over a specified time horizon with market excess returns for all instruments.
    
    Parameters:
    s (Dataframe): Dataframe of weighted signals for all instruments 
    horizon (int): The number of days into the future to look.
    df (Dataframe) : Dataframe containing the historical prices of all instruments.
    
    Returns:
    Series: The calculated PnL return for all instruments at each time index with market excess returns.
    """
    times = df['date'].unique()
    pnl_series = pd.Series(index=times, dtype=float)

    for time in times:
        pnl_series[time] = pnl_m(s, time, horizon, df)
    
    return pnl_series

## Sharpe Ratio
def sharpe_ratio_annual(pnl_series, risk_free_rate=0.0):
    """
    Calculate the Sharpe Ratio of a series of PnL returns.
    
    Parameters:
    pnl_series (Series): A series of PnL returns.
    risk_free_rate (float): The risk-free rate to subtract from the returns.
    
    Returns:
    float: The calculated Sharpe Ratio.
    """
    excess_returns = pnl_series - risk_free_rate
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else np.nan


## Cumulative Returns
#def cumulative_returns()

