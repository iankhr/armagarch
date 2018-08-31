# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:18:57 2018

@author: Ian
"""
from basicFun import getLag
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from garch import garch
import statsmodels.tsa as smt
yf.pdr_override()

data = yf.download("^GSPC", start="2010-01-01", end="2018-01-01")
rt = np.log(data['Adj Close']).diff().dropna()

gm = garch(rt*100, PQ=(1,0), poq = (1,0,1))
gm.fit()

# testing the goodness of fit
et = gm.et
var = np.sqrt(gm.ht)*1.96 # 5%
prop = np.sum(np.abs(et)>var)/len(var)
print(str(prop*100)+'%')
var = np.sqrt(gm.ht)*2.58 # 1%
prop = np.sum(np.abs(et)>var)/len(var)
print(str(prop*100)+'%')
# it means that the model underestimates the variance. if the values are bigger
# than 5% and 1%. In particular, it manages really bad the tails
