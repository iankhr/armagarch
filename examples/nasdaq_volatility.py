# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 08:53:44 2020

@author: iankh
"""

import pandas as pd
import armagarch as ag

# read data
data = pd.read_csv('NASDAQ.csv', parse_dates = True)
# set dates as the main index
data.set_index('Date', inplace = True)
# get returns
data = 100*data['Adj Close'].pct_change().dropna()

# define the model
meanMdl = ag.ARMA(order = {'AR':1, 'MA':0})
volMdl = ag.garch(order = {'p':1,'q':1})
dist = ag.tStudent()

# set-up the model
mdl = ag.empModel(data.to_frame(), meanMdl, volMdl, dist)
mdl.fit()

"""
Compare the output with Matlab
ARIMA(1,0,0) Model (t Distribution):
 
                  Value      StandardError    TStatistic      PValue  
                _________    _____________    __________    __________

    Constant    0.0009524     0.00017051        5.5855      2.3308e-08
    AR{1}         0.13987       0.019051         7.342      2.1037e-13
    DoF            8.3525         1.0273        8.1308       4.266e-16

 
 
    GARCH(1,1) Conditional Variance Model (t Distribution):
 
                  Value       StandardError    TStatistic      PValue  
                __________    _____________    __________    __________

    Constant    1.6076e-06     6.1538e-07        2.6123       0.0089925
    GARCH{1}       0.89701       0.011191        80.153               0
    ARCH{1}       0.095254       0.010975         8.679      3.9935e-18
    DoF             8.3525         1.0273        8.1308       4.266e-16
    
Differences in standard errors are due to nonrobust variance-covariance matrix
used in Matlab by default.
"""