# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:37:51 2020

This is an example of how to use armagarch package to use ARMA-GARCH module.
The module uses the data from Kenneth French's Data library and estimates
conditional volatility for excess market returns.

@author: Ian Khrashchevskyi
"""

import armagarch as ag
import pandas_datareader as web
import matplotlib.pyplot as plt
import numpy as np

# load data from KennethFrench library
ff = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench')
ff = ff[0]

# define mean, vol and distribution
meanMdl = ag.ARMA(order = {'AR':1,'MA':0})
volMdl = ag.garch(order = {'p':1,'o':1,'q':1})
distMdl = ag.normalDist()

# create a model
model = ag.empModel(ff['Mkt-RF'].to_frame(), meanMdl, volMdl, distMdl)
# fit model
model.fit()

# get the conditional mean
Ey = model.Ey

# get conditional variance
ht = model.ht
cvol = np.sqrt(ht)

# get standardized residuals
stres = model.stres

# plot in three subplots
fig, ax = plt.subplots(3,1, figsize=(13,10))
Ey.plot(ax=ax[0], color = 'black', title = 'Expected returns',\
        legend = False)
cvol.plot(ax=ax[1], color = 'black', title = 'Conditional volatility',\
          legend = False)
stres.plot(ax=ax[2], color = 'black', title = 'Standardized residuals',\
           legend = False)
plt.subplots_adjust(hspace=0.5)
plt.show()
