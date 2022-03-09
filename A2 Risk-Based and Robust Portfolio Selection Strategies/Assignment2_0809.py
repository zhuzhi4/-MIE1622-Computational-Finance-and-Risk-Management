#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import math
import cplex 
import ipopt
import matplotlib.pyplot as plt


# In[2]:


# Complete the following functions
def strat_buy_and_hold(x_init, cash_init, mu, Q, cur_prices,returned):
    x_optimal = x_init
    cash_optimal = cash_init
    new_val = np.dot(cur_prices, x_optimal)
    w_optimal = (x_optimal*cur_prices)/ new_val
    return x_optimal, cash_optimal, w_optimal, 0 


# In[3]:


def strat_equally_weighted(x_init, cash_init, mu, Q, cur_prices, returned):
    curr_value = np.sum(x_init*cur_prices) + cash_init
    equal_wgt = (curr_value)/(len(x_init))
    x_optimal = np.floor(equal_wgt/ cur_prices)
    x_diff = abs(np.subtract(x_init, x_optimal))
    new_val = np.dot(cur_prices, x_optimal)
    w_optimal = (x_optimal*cur_prices)/ new_val
    cash_optimal = curr_value - np.dot(cur_prices, x_optimal) - (0.005*np.dot(cur_prices,x_diff))
    return x_optimal, cash_optimal, w_optimal,0


# In[4]:


def strat_min_variance(x_init, cash_init, mu, Q, cur_prices, returned):
    # obj = w.T * Q * W
    #s.t sum(wi)= 1, wi >=0 for all i 
    
    curr_value = np.sum(x_init*cur_prices) + cash_init
    n = len(x_init)
    #initialize Cplex object
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    
    #Define linear part of objective function and bounds on variables, no linear part
    c = [0.0]*n 
    lb = [0.0]*n
    cols = []
    for i in range(n):
        cols.append([[0],[1]])
    var_names = ["w_%s" % i for i in range(1,n+1)]
    #add linear obj, constr, lb to model
    cpx.linear_constraints.add(rhs=[1.0], senses="E")
    cpx.variables.add(obj=c, lb=lb, columns=cols, names = var_names)

    #define part of obj 
    qmat =[[list(range(n)),list(2*Q[i,:])] for i in range(n)]
    cpx.objective.set_quadratic(qmat)
    
    #set paramters
    cpx.parameters.threads.set(4)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    
    cpx.solve()
    
    #retrieve optimal weight
    w_min = cpx.solution.get_values()
    w_min = np.asarray(w_min)
    w_value = w_min*curr_value
    
    #transform into money amount and get new balance of cash 
    x_optimal = np.floor(w_value/ cur_prices)
    x_diff = abs(np.subtract(x_init, x_optimal)) 
    new_val = np.dot(cur_prices, x_optimal)
    w_optimal = (x_optimal*cur_prices)/ new_val
    cash_optimal = curr_value - np.dot(cur_prices, x_optimal) - (0.005*np.dot(cur_prices,x_diff))
    return x_optimal, cash_optimal, w_optimal, 0


# In[5]:


def strat_max_Sharpe(x_init, cash_init, mu, Q, cur_prices, returned):
    curr_value = np.sum(x_init*cur_prices) + cash_init
    n = len(x_init)+1
    # daily risk free rate 
    daily_rf = 1.045**(1.0/252)-1
    r_diff = mu - daily_rf
    contains_negative = any(r_diff<0)
    if contains_negative == True:
        x_optimal = x_init
        cash_optimal = cash_init
        w_optimal = (x_optimal*cur_prices)/curr_value 
    else: 
        #initialize Cplex object
        cpx = cplex.Cplex()
        cpx.objective.set_sense(cpx.objective.sense.minimize)

        #Define linear part of objective function and bounds on variables, no linear part
        c = [0.0]*n 
        lb = [0.0]*n
        cols = []

        for i in range(n-1):
            cols.append([[0,1],[r_diff[i],1]])

        cols.append([[0,1],[0,-1]])
        var_names = ["y_%s" % i for i in range(1,n)]
        var_names.append("k")

        #add linear obj, constr, lb to model
        cpx.linear_constraints.add(rhs=[1.0, 0], senses="EE")
        cpx.variables.add(obj=c, lb=lb, columns=cols, names = var_names)

        #define quad part of obj 
        c_k = np.zeros((20,1))        # add a col of 0 coeff for k 
        Q = np.hstack((Q,c_k))
        c_k = np.zeros((1,21))        # add a row of 0 coeff for k 
        Q = np.vstack((Q,c_k))

        qmat =[[list(range(n)),list(2*Q[i,:])] for i in range(n)]
        cpx.objective.set_quadratic(qmat)

        #set parameters
        cpx.parameters.threads.set(4)
        cpx.set_results_stream(None)
        cpx.set_warning_stream(None)

        #solve
        cpx.solve()

        #retrieve optimal sol
        y_max = cpx.solution.get_values()
        y_max = np.asarray(y_max)
        kappa = y_max[-1]
        y_max = y_max[0:-1]

        # calculate weight using y 
        w_max = y_max/kappa
        w_value = w_max*curr_value
        x_optimal = np.floor(w_value/ cur_prices)
        x_diff = abs(np.subtract(x_init, x_optimal))  
        new_val = np.dot(cur_prices, x_optimal)
        w_optimal = (x_optimal*cur_prices)/ new_val
        cash_optimal = curr_value - np.dot(cur_prices, x_optimal) - (0.005*np.dot(cur_prices,x_diff))

    return x_optimal, cash_optimal, w_optimal, 0 


# In[6]:


def strat_equal_risk_contr(x_init, cash_init, mu, Q, cur_prices, returned):
    class erc(object):
        def __init__(self):
            pass

        def objective(self, x):
            # The callback for calculating the objective
            y = x * np.dot(Q, x)
            fval = 0
            for i in range(n):
                for j in range(i,n):
                    xij = y[i] - y[j]
                    fval = fval + xij*xij
            fval = 2*fval
            return fval

        def gradient(self, x):
            # The callback for calculating the gradient
            grad = np.zeros(n)
            y = x * np.dot(Q, x)
            #  use finite differences to check the gradient
            for i in range(n):
                for j in range(n):
                    diff1 = np.dot(Q[i],x) + np.dot(Q[i][i],x[i])
                    diff2 = np.dot(Q[i][j], x[i])
                    delta_g = (y[i]-y[j]) * (diff1 - diff2)
                    grad[i] = grad[i] + delta_g
                grad[i] = 2 * 2 * grad[i]
            return grad

        def constraints(self, x):
        # The callback for calculating the constraints
            return [1.0] * n
    
        def jacobian(self, x):
        # The callback for calculating the Jacobian
            return np.array([[1.0] * n])
    
    curr_value = np.sum(x_init* cur_prices)
    n = len(x_init)
    w0 = (x_init*cur_prices)/ curr_value
    lb = [0.0] * n  # lower bounds on variables
    ub = [1.0] * n  # upper bounds on variables
    cl = [1]        # lower bounds on constraints
    cu = [1]        # upper bounds on constraints
        
    # Define IPOPT problem
    nlp = ipopt.problem(n=len(w0), m=len(cl), problem_obj=erc(), lb=lb, ub=ub, cl=cl, cu=cu)
    
    # Set the IPOPT options
    nlp.addOption('jac_c_constant'.encode('utf-8'), 'yes'.encode('utf-8'))
    nlp.addOption('hessian_approximation'.encode('utf-8'), 'limited-memory'.encode('utf-8'))
    nlp.addOption('mu_strategy'.encode('utf-8'), 'adaptive'.encode('utf-8'))
    nlp.addOption('tol'.encode('utf-8'), 1e-10)
    w_erc, info = nlp.solve(w0)
    w_erc = np.asarray(w_erc)
    w_erc = w_erc*(1/w_erc.sum())
    
    w_value = w_erc*curr_value
    x_optimal = np.floor(w_value/cur_prices)
    x_diff = abs(np.subtract(x_init, x_optimal)) 
    new_val = np.dot(cur_prices, x_optimal)
    w_optimal = (x_optimal*cur_prices)/ new_val
    cash_optimal = curr_value - np.dot(cur_prices, x_optimal) - (0.005*np.dot(cur_prices,x_diff))
    
    return x_optimal, cash_optimal, w_optimal, 0


# In[7]:


def strat_lever_equal_risk_contr(x_init, cash_init, mu, Q, cur_prices, returned ):
    class erc(object):
        def __init__(self):
            pass

        def objective(self, x):
            # The callback for calculating the objective
            y = x * np.dot(Q, x)
            fval = 0
            for i in range(n):
                for j in range(i,n):
                    xij = y[i] - y[j]
                    fval = fval + xij*xij
            fval = 2*fval
            return fval

        def gradient(self, x):
            # The callback for calculating the gradient
            grad = np.zeros(n)
            y = x * np.dot(Q, x)
            #  use finite differences to check the gradient
            for i in range(n):
                for j in range(n):
                    diff1 = np.dot(Q[i],x) + np.dot(Q[i][i],x[i])
                    diff2 = np.dot(Q[i][j], x[i])
                    delta_g = (y[i]-y[j]) * (diff1 - diff2)
                    grad[i] = grad[i] + delta_g
                grad[i] = 2 * 2 * grad[i]
            return grad

        def constraints(self, x):
        # The callback for calculating the constraints
            return [1.0] * n
    
        def jacobian(self, x):
        # The callback for calculating the Jacobian
            return np.array([[1.0] * n])
    
    pfval = np.sum(x_init*cur_prices) +cash_init - returned*(1+0.025/6)
    borrow = pfval
    curr_value = pfval + borrow
    n = len(x_init)
    w0 = (x_init*cur_prices)/ curr_value
    lb = [0.0] * n  # lower bounds on variables
    ub = [1.0] * n  # upper bounds on variables
    cl = [1]        # lower bounds on constraints
    cu = [1]        # upper bounds on constraints
        
    # Define IPOPT problem
    nlp = ipopt.problem(n=len(w0), m=len(cl), problem_obj=erc(), lb=lb, ub=ub, cl=cl, cu=cu)
    
    # Set the IPOPT options
    nlp.addOption('jac_c_constant'.encode('utf-8'), 'yes'.encode('utf-8'))
    nlp.addOption('hessian_approximation'.encode('utf-8'), 'limited-memory'.encode('utf-8'))
    nlp.addOption('mu_strategy'.encode('utf-8'), 'adaptive'.encode('utf-8'))
    nlp.addOption('tol'.encode('utf-8'), 1e-10)
    w_lerc, info = nlp.solve(w0)
    w_lerc = np.asarray(w_lerc)
    w_lerc = w_lerc*(1/w_lerc.sum())
    
    w_value = w_lerc*curr_value
    x_optimal = np.floor(w_value/cur_prices)
    x_diff = abs(np.subtract(x_init, x_optimal)) 
    new_val = np.dot(cur_prices, x_optimal)
    w_optimal = (x_optimal*cur_prices)/ new_val
    cash_optimal = curr_value - np.dot(cur_prices, x_optimal) - (0.005*np.dot(cur_prices,x_diff))
    
    return x_optimal, cash_optimal, w_optimal, borrow


# In[8]:


def strat_robust_optim(x_init, cash_init, mu, Q, cur_prices, returned):
    
    curr_value = np.sum(x_init*cur_prices) + cash_init
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    n = len(x_init)
    
    #############
        
    #required robustness
    w0 = [1.0/n] * n
    var_matr = np.diag(np.diag(Q))
    # Target portfolio return estimation error is return estimation error of 1/n portfolio
    rob_init = np.dot(w0, np.dot(var_matr, w0)) # return estimation error of initial portfolio
    rob_bnd  = rob_init
    

    #Define linear part of objective function and bounds on variables, no linear part
    c = [0.0]*n 
    lb = [0.0]*n
    cols = []
    for i in range(n):
        cols.append([[0],[1]])
    var_names = ["w_%s" % i for i in range(1,n+1)]
    #add linear obj, constr, lb to model
    cpx.linear_constraints.add(rhs=[1.0], senses="E")
    cpx.variables.add(obj=c, lb=lb, columns=cols, names = var_names)

    #define part of obj 
    qmat =[[list(range(n)),list(2*Q[i,:])] for i in range(n)]
    cpx.objective.set_quadratic(qmat)
    
    #set paramters
    cpx.parameters.threads.set(4)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    
    cpx.solve()
    w1 = np.array(cpx.solution.get_values())
    w_minVar = w1
    var_minVar = np.dot(w_minVar, np.dot(Q, w_minVar))
    ret_minVar = np.dot(mu, w_minVar)
    Portf_Retn = ret_minVar

    ##############
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    c = [0.0] * n
    lb = [0.0] * n
    ub = [1.0] * n 
    A = []
    for k in range(n):
        A.append([[0,1],[1.0,mu[k]]])
    var_names = ["w_%s" % i for i in range(1,n+1)]
    cpx.linear_constraints.add(rhs=[1.0,Portf_Retn], senses="EG")
    cpx.variables.add(obj=c, lb=lb, ub=ub, columns=A, names=var_names)
    Qmat = [[list(range(n)), list(2*Q[k,:])] for k in range(n)]
    cpx.objective.set_quadratic(Qmat)
    Qcon = cplex.SparseTriple(ind1=var_names, ind2=range(n), val=np.diag(var_matr))
    cpx.quadratic_constraints.add(rhs=rob_bnd, quad_expr=Qcon, name="Qc")
    
    cpx.parameters.threads.set(4)
    cpx.parameters.timelimit.set(60)
    cpx.parameters.barrier.qcpconvergetol.set(1e-12)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.solve()
    
    w_rMV = cpx.solution.get_values()
    # Round near-zero portfolio weights
    w_rMV = np.array(w_rMV)
    w_rMV[w_rMV<1e-6] = 0
    w_rMV = w_rMV / np.sum(w_rMV)
    
    w_value = w_rMV *curr_value
    x_optimal = np.floor(w_value/cur_prices)
    x_diff = abs(np.subtract(x_init, x_optimal))  
    new_val = np.dot(cur_prices, x_optimal)
    w_optimal = (x_optimal*cur_prices)/ new_val
    cash_optimal = curr_value - np.dot(cur_prices, x_optimal) - (0.005*np.dot(cur_prices,x_diff))
    
    return x_optimal, cash_optimal, w_optimal, 0 


# In[9]:


# Input file
input_file_prices = 'Daily_closing_prices20082009.csv'

# Read data into a dataframe
df = pd.read_csv(input_file_prices)

# Convert dates into array [year month day]
def convert_date_to_array(datestr):
    temp = [int(x) for x in datestr.split('/')]
    return [temp[-1], temp[0], temp[1]]

dates_array = np.array(list(df['Date'].apply(convert_date_to_array)))
data_prices = df.iloc[:, 1:].to_numpy()
dates = np.array(df['Date'])
# Find the number of trading days in Nov-Dec 2014 and
# compute expected return and covariance matrix for period 1
day_ind_start0 = 0
day_ind_end0 = len(np.where(dates_array[:,0]==2007)[0])
cur_returns0 = data_prices[day_ind_start0+1:day_ind_end0,:] / data_prices[day_ind_start0:day_ind_end0-1,:] - 1
mu = np.mean(cur_returns0, axis = 0)
Q = np.cov(cur_returns0.T)

# Remove datapoints for year 2007
data_prices = data_prices[day_ind_end0:,:]
dates_array = dates_array[day_ind_end0:,:]
dates = dates[day_ind_end0:]

# Initial positions in the portfolio
init_positions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 980, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20000])

# Initial value of the portfolio
init_value = np.dot(data_prices[0,:], init_positions)
print('\nInitial portfolio value = $ {}\n'.format(round(init_value, 2)))

# Initial portfolio weights
w_init = (data_prices[0,:] * init_positions) / init_value

# Number of periods, assets, trading days
N_periods = 6*len(np.unique(dates_array[:,0])) # 6 periods per year
N = len(df.columns)-1
N_days = len(dates)

# Annual risk-free rate for years 2019-2020 is 2.5%
r_rf = 0.045
# Number of strategies
strategy_functions = ['strat_buy_and_hold', 'strat_equally_weighted', 'strat_min_variance', 'strat_max_Sharpe', 'strat_equal_risk_contr', 'strat_lever_equal_risk_contr', 'strat_robust_optim']
strategy_names     = ['Buy and Hold', 'Equally Weighted Portfolio', 'Mininum Variance Portfolio', 'Maximum Sharpe Ratio Portfolio', 'Equal Risk Contributions Portfolio', 'Leveraged Equal Risk Contributions Portfolio', 'Robust Optimization Portfolio']
N_strat = 7  # comment this in your code
#N_strat = len(strategy_functions)  # uncomment this in your code
fh_array = [strat_buy_and_hold, strat_equally_weighted, strat_min_variance, strat_max_Sharpe, strat_equal_risk_contr, strat_lever_equal_risk_contr, strat_robust_optim]


# In[10]:


portf_value = [0] * N_strat
x = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
cash = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
weight = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
borrowed = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
start_list = np.zeros(12)
end_list = np.zeros(12)
for period in range(1, N_periods+1):
   # Compute current year and month, first and last day of the period
    if dates_array[0, 0] == 8:
        cur_year  = 8 + math.floor(period/7)
    else:
        cur_year  = 2008 + math.floor(period/7)

    cur_month = 2*((period-1)%6) + 1
    day_ind_start = min([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month)) if val])
    start_list[period-1] = day_ind_start
    day_ind_end = max([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month+1)) if val])
    end_list[period-1] = day_ind_end
    print('\nPeriod {0}: start date {1}, end date {2}'.format(period, dates[day_ind_start], dates[day_ind_end]))
   
   # Prices for the current day
    cur_prices = data_prices[day_ind_start,:]

   # Execute portfolio selection strategies
    for strategy in range(N_strat):

      # Get current portfolio positions
        if period == 1:
            curr_positions = init_positions
            curr_cash = 0
            portf_value[strategy] = np.zeros((N_days, 1))
            brw = 0
        else:
            curr_positions = x[strategy, period-2]
            curr_cash = cash[strategy, period-2]
            brw = borrowed[strategy, period-2]
      # Compute strategy, how to call function?? 
       
        x[strategy, period-1], cash[strategy, period-1],weight[strategy, period-1], borrowed[strategy, period-1] = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices, brw)
    

      # Verify that strategy is feasible (you have enough budget to re-balance portfolio)
      # Check that cash account is >= 0
      # Check that we can buy new portfolio subject to transaction costs
      # If cash < 0, reduce amount of stock purchase according to re-balance weight 

        if cash[strategy][period-1]<0:
            
            curr_value = np.dot(curr_positions,cur_prices) + curr_cash - brw*(1 + 0.025/6)  #calculate total amount of money we have 
            if strategy == 5:  
                curr_value = curr_value*2
            ratio = x[strategy][period-1]/np.sum(x[strategy][period-1])           # determine weight of rebalance port 
            dollar_excess = abs(cash[strategy][period-1])* ratio                    #excess in dollar amount wrt weight 
            pos_excess = np.ceil(dollar_excess/cur_prices)                     #excess num of stock for each stock
            x[strategy][period-1] = x[strategy][period-1] - pos_excess       #new rebalance position
            transaction_cost = np.dot(cur_prices, abs(x[strategy][period-1]-curr_positions))*0.005
            cash[strategy][period-1] = curr_value - np.sum(cur_prices*x[strategy][period-1]) - transaction_cost
            
      # Compute portfolio value
        p_values = np.dot(data_prices[day_ind_start:day_ind_end+1,:], x[strategy, period-1]) + cash[strategy, period-1]
        if strategy ==5: 
            p_values -= borrowed[strategy, period-1]
        portf_value[strategy][day_ind_start:day_ind_end+1] = np.reshape(p_values, (p_values.size,1))
        print('  Strategy "{0}", value begin = $ {1:.2f}, value end = $ {2:.2f}'.format( strategy_names[strategy], 
             portf_value[strategy][day_ind_start][0], portf_value[strategy][day_ind_end][0]))

      
   # Compute expected returns and covariances for the next period
    cur_returns = data_prices[day_ind_start+1:day_ind_end+1,:] / data_prices[day_ind_start:day_ind_end,:] - 1
    mu = np.mean(cur_returns, axis = 0)
    Q = np.cov(cur_returns.T)
    


# In[11]:


# Plot results
# 1. daily value of portfolio 
plt.rcParams['figure.figsize'] = [15, 10]
plt.plot(portf_value[0], label = 'Buy and Hold')
plt.plot(portf_value[1], label = 'Equally Weighted Portfolio')
plt.plot(portf_value[2], label = 'Minimum Variance Portfolio')
plt.plot(portf_value[3], label = 'Maximum Sharpe Ratio Portfolio')
plt.plot(portf_value[4], label = 'Equal Risk Contribution')
plt.plot(portf_value[5], label = 'Leveraged Equal Risk Contribution')
plt.plot(portf_value[6], label = 'Robust Optimization')
plt.legend()
plt.ylabel('Portfolio Value $', fontsize=15)
plt.xlabel('Trading days', fontsize=15)
plt.title('Daily Portfolio values between 2019 and 2020', fontsize= 20)
plt.savefig('PVal_2008-2009')
plt.show()


# In[31]:


# plot mmd 
start_list = np.int_(start_list)
end_list = np.int_(end_list)
portf_values_list = np.array(portf_value)
def get_max_dd(portf_values):
    mdd = 0 
    peak = -np.inf
    dd = np.zeros(portf_values.shape)
    n = portf_values.shape[0]
    for i in range(n):
        if portf_values[i] > peak:
            peak = portf_values[i]
        dd[i] = 100*(peak-portf_values[i]) / peak
        if dd[i] > mdd:
            mdd = dd[i]
    return mdd

dd_list = np.zeros((7,12))
for i in range(7):
    for j in range(12):
        start, end = start_list[j], end_list[j]
        portf_values_curr_period = portf_values_list[i].flatten()[start:end+1]
        dd_list[i,j] = get_max_dd(portf_values_curr_period)
        
plt.figure(figsize = (20,10))
plt.plot(dd_list[0], label = 'Buy and hold')
plt.plot(dd_list[1], label = 'Equal Weight')
plt.plot(dd_list[2], label = 'Minimum Variance')
plt.plot(dd_list[3], label = 'Maximum Sharpes Ratio')
plt.plot(dd_list[4], label = 'Equal Risk Contribution')
plt.plot(dd_list[5], label = 'Leverage Equal Risk Contribution')
plt.plot(dd_list[6], label = 'Robust Optimization')
plt.legend()
plt.xlabel('Dropdown Percentage', fontsize = 15)
plt.ylabel('Period', fontsize =15)
plt.title('Maximum Dropdown per Period for 2008-2009', fontsize =25)


# In[22]:


## Dynamic Change Strat 3 
df_col = list(df.columns.values)
asset_name = df_col[1:]
init_positions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 980, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20000])
init_val = init_positions* data_prices[0, :] #initial $value of each stock 
init_pval = sum(init_val)
w_init = init_val/init_pval
period = np.arange(13)
for i in range(20):
    asset_w = [w_init[i]]
    for j in range(12):
        asset_w.append(weight[2][j][i])        #strategy 3 index 2
    plt.plot(period, asset_w, label = asset_name[i])
plt.legend()
plt.xlim([0,12])
plt.ylim([0,1])
plt.xlabel('Trading Periods', fontsize=20)
plt.ylabel('Portfolio Weight Proportion',fontsize=20)
plt.title('Dynamic Change of Value Weight for Minimum Variance Strategy',fontsize=25)


# In[21]:


## Dynamic Change Strat 4
df_col = list(df.columns.values)
asset_name = df_col[1:]
init_positions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 980, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20000])
uw_init = init_positions/np.sum(init_positions)
period = np.arange(13)
utotal_list = []
for i in range(12):
    utotal_list.append(np.sum(x[2][i]))

for i in range(20):
    asset_u = [uw_init[i]]
    for j in range(12):
        u =x[2][j][i]#strategy 3 index 2
        u_total = utotal_list[j]
        uw = u/u_total
        asset_u.append(uw)
    plt.plot(period, asset_u, label = asset_name[i])
plt.legend()
plt.xlim([0,12])
plt.ylim([0,1])
plt.xlabel('Trading Periods', fontsize=20)
plt.ylabel('Portfolio Weight Proportion',fontsize=20)
plt.title('Dynamic Change of Share Unit Weight for Minimum Variance Strategy',fontsize=25)


# In[23]:


for i in range(20):
    asset_w = [w_init[i]]
    for j in range(12):
        asset_w.append(weight[3][j][i])        #strategy 4 index 3
    plt.plot(period, asset_w, label = asset_name[i])
plt.legend()
plt.xlim([0,12])
plt.ylim([0,1])
plt.xlabel('Trading Periods', fontsize=20)
plt.ylabel('Portfolio Weight Proportion',fontsize=20)
plt.title('Dynamic Change of Value Weight for Maximum Sharpe Ratio',fontsize=25)


# In[25]:


## Dynamic Change Strat 7
df_col = list(df.columns.values)
asset_name = df_col[1:]
init_positions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 980, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20000])
uw_init = init_positions/np.sum(init_positions)
period = np.arange(13)
utotal_list = []
for i in range(12):
    utotal_list.append(np.sum(x[3][i]))

for i in range(20):
    asset_u = [uw_init[i]]
    for j in range(12):
        u =x[3][j][i]#strategy 4 index 3
        u_total = utotal_list[j]
        uw = u/u_total
        asset_u.append(uw)
    plt.plot(period, asset_u, label = asset_name[i])
plt.legend()
plt.xlim([0,12])
plt.ylim([0,1])
plt.xlabel('Trading Periods', fontsize=20)
plt.ylabel('Portfolio Weight Proportion',fontsize=20)
plt.title('Dynamic Change of Share Unit Weight for Maximum Sharpes Ratio Strategy',fontsize=23)


# In[28]:


for i in range(20):
    asset_w = [w_init[i]]
    for j in range(12):
        asset_w.append(weight[6][j][i])
    plt.plot(period, asset_w, label = asset_name[i])

plt.legend()
plt.xlim([0,12])
plt.ylim([0,1])
plt.xlabel('Trading Periods', fontsize=20)
plt.ylabel('Portfolio Weight Proportion',fontsize=20)
plt.title('Dynamic Change of Value Weight for Robust Optimization',fontsize=25)    


# In[30]:


## Dynamic Change Strat 7
df_col = list(df.columns.values)
asset_name = df_col[1:]
init_positions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 980, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20000])
uw_init = init_positions/np.sum(init_positions)
period = np.arange(13)
utotal_list = []
for i in range(12):
    utotal_list.append(np.sum(x[6][i]))

for i in range(20):
    asset_u = [uw_init[i]]
    for j in range(12):
        u =x[6][j][i]#strategy 7 index 6
        u_total = utotal_list[j]
        uw = u/u_total
        asset_u.append(uw)
    plt.plot(period, asset_u, label = asset_name[i])
plt.legend()
plt.xlim([0,12])
plt.ylim([0,1])
plt.xlabel('Trading Periods', fontsize=20)
plt.ylabel('Portfolio Weight Proportion',fontsize=20)
plt.title('Dynamic Change of Share Unit Weight for Robust Optimization',fontsize=23)


# In[ ]:




