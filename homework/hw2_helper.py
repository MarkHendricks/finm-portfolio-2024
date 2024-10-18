import pandas as pd
import numpy as np

def stats(data, portfolio = None, portfolio_name = 'Portfolio', annualize = True):
    
    if portfolio is None:
        returns = data
    else:
        returns = data @ portfolio
    
    output = returns.agg(['mean','std'])
    output.loc['sharpe'] = output.loc['mean'] / output.loc['std']
    
    if annualize == True:
        output.loc['mean'] *= 12
        output.loc['std'] *= np.sqrt(12)
        output.loc['sharpe'] *= np.sqrt(12)
    
    if portfolio is None:
        pass
    else:
        output.columns = [portfolio_name]
    
    return output

# Create function to calculate max drawdown and associated dates

def max_drawdown(data, portfolio = None, portfolio_name = 'Portfolio'):
    
    if portfolio is None:
        returns = data
        output = pd.DataFrame(columns=returns.columns)
    else:
        returns = data @ portfolio
        output = pd.DataFrame(columns=[portfolio_name])
    
    cumulative = (returns + 1).cumprod()
    maximum = cumulative.expanding().max()
    drawdown = cumulative / maximum - 1
    
    for col in output.columns:
        
        output.loc['MDD',col] = drawdown[col].min()
        output.loc['Max Date',col] = cumulative[cumulative.index < drawdown[col].idxmin()][col]\
                                             .idxmax()\
                                             .date()
        output.loc['Min Date',col] = drawdown[col].idxmin().date()
        recovery_date = drawdown.loc[drawdown[col].idxmin():,col]\
                                             .apply(lambda x: 0 if x == 0 else np.nan)\
                                             .idxmax()
        
        if pd.isna(recovery_date):
            output.loc['Recovery Date',col] = recovery_date
            output.loc['Recovery Period',col] = np.nan
        else:
            output.loc['Recovery Date',col] = recovery_date.date()
            output.loc['Recovery Period',col] = (output.loc['Recovery Date',col]\
                                             - output.loc['Min Date',col])\
                                             .days
        
    return output

# Create function to retrieve other statistics

def stats_tail_risk(data, portfolio = None, portfolio_name = 'Portfolio', VaR = 0.05):
    
    if portfolio is None:
        returns = data
    else:
        returns = data @ portfolio
    
    output = returns.agg(['skew',
                          'kurt'])
    output.loc['VaR'] = returns.quantile(q = 0.05)
    output.loc['CVaR'] = returns[returns <= output.loc['VaR']].mean()
    output = pd.concat([output, max_drawdown(returns,portfolio,portfolio_name)])
    
    if portfolio is None:
        pass
    else:
        output.columns = portfolio_name
    
    return output

# Create function to display regression stats

def stats_OLS(model,y,x):
    
    output = model.params.to_frame(name = y.columns[0])
    
    return output