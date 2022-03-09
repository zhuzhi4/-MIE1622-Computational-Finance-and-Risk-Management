# Optional Starter Code for Final Exam Project

# not mandatory to be used, you can start with your own codes

# Import libraries
import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt

# Import other libraries
# Insert your code here #

# Read Daily Prices

# CSV file with price data
input_file_prices  = 'Daily_closing_prices.csv'
if path.exists(input_file_prices):
    print('\nReading daily prices datafile - {}\n'.format(input_file_prices))
    fid = pd.read_csv(input_file_prices)
    # instrument tickers
    tickers = list(fid.columns)[1:]
    # time periods
    dates = fid['Date']
    data_prices = fid.values[:,1:]
else:
    print("No such file '{}'".format(input_file_prices), file=sys.stderr)
    
# Convert dates into array [year month day]
def convert_date_to_array(datestr):
    temp = [int(x) for x in datestr.split('/')]
    return [temp[-1], temp[0], temp[1]]

dates_array = np.array(list(fid['Date'].apply(convert_date_to_array)))

# Question 1

# Specify quantile level for VaR/CVaR
alf = 0.95

# Number of assets in universe
Na = data_prices.shape[1]

# Number of historical scenarios
Ns = data_prices.shape[0]

# Positions in the portfolio
positions = np.array([100, 0, 0, 0, 0, 0, 0, 0, 200, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(Na,1)

################################ Insert your code here ################################

#print('Historical 1-day VaR %4.1f%% = $%6.2f,   Historical 1-day CVaR %4.1f%% = $%6.2f\n'% (100*alf, VaR1, 100*alf, CVaR1))
#print('    Normal 1-day VaR %4.1f%% = $%6.2f,       Normal 1-day CVaR %4.1f%% = $%6.2f\n'% (100*alf, VaR1n, 100*alf, CVaR1n))
#print('Historical 10-day VaR %4.1f%% = $%6.2f,   Historical 10-day CVaR %4.1f%% = $%6.2f\n'% (100*alf, VaR10, 100*alf, CVaR10))
#print('    Normal 10-day VaR %4.1f%% = $%6.2f,       Normal 10-day CVaR %4.1f%% = $%6.2f\n'% (100*alf, VaR10n, 100*alf, CVaR10n))


# Question 2

# Annual risk-free rate for years 2015-2016 is 2.5%
r_rf = 0.025

# Compute means and covariances for Question 2 (2019 and 2020)
cur_returns = data_prices[1:,:] / data_prices[:Ns-1,:] - 1
cur_returns = cur_returns

# Expected returns for Question 2
mu = np.mean(cur_returns, axis=0).reshape(cur_returns.shape[1],1)
# Covariances for Question 2
Q = np.cov(cur_returns.astype(float).T)


################################ Insert your code here ################################


# Question 3

# Import FF models data from 2019-01-03 to 2020-12-31
input_file_factors  = 'Daily_FF_factors.csv'
if path.exists(input_file_factors):
    print('\nReading daily FF factors datafile - {}\n'.format(input_file_factors))
    ff_data = pd.read_csv(input_file_factors)
    factors_name = list(ff_data.columns)[1:-1]
    rf      = ff_data['RF'][1:]
    factors = ff_data.values[1:,1:4]
else:
    print("No such file '{}'".format(input_file_factors), file=sys.stderr)

# Indicides helpful to seperate data
# dates[day_ind_start_2019] = '01/02/2019',
# dates[day_ind_end_2019] = '12/31/2019',
# dates[day_ind_start_2019] = '01/02/2020',
# dates[day_ind_end_2019] = '12/31/2019'.
day_ind_start_2019 = 0
day_ind_end_2019   = 251
day_ind_start_2020 = 252
day_ind_end_2020   = 504

data_prices_2019   = data_prices[day_ind_start_2019:day_ind_end_2019+1]
data_prices_2020   = data_prices[day_ind_start_2020:day_ind_end_2020+1]


################################ Insert your code here ################################





