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

gm = garch(rt*100, PQ=(1,0), poq = (2,0,1))
gm.fit()
