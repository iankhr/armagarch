# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 08:53:44 2020

@author: iankh
"""

import pandas as pd
import armagarch as ag

# read data
data = pd.read_csv('FTSE_1984_2012.csv', parse_dates = True)
# set dates as the main index
data.set_index('Date', inplace = True)
# flip from oldest to newest
data = data[::-1]
# get returns
data = 100*data['Adj Close'].pct_change().dropna()

# define the model
meanMdl = ag.ARMA(order = {'AR':1, 'MA':1})
volMdl = ag.gjr(order = {'p':1,'o':1,'q':1})
dist = ag.normalDist()

# set-up the model
mdl = ag.empModel(data.to_frame(), meanMdl, volMdl, dist)
mdl.fit()

"""
Compare the output with Kevin Sheppard
Parameter   Estimate       Std. Err.      T-stat
mu          0.032146        0.010084     3.18795
omega       0.017610        0.003330     5.28813
alpha       0.030658        0.006730     4.55564
gamma       0.091709        0.012944     7.08484
beta        0.906327        0.009784     92.62951
"""