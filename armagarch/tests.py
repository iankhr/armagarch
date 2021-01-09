# -*- coding: utf-8 -*-
"""
Created on Sat Jan 09 10:10:01 2021

This is the file that contains different statistical tests used for modelling of ARMA-GARCH processes

@author: Ian Khrashchevskyi
"""
import pandas as pd
import numpy as np
import scipy.stats as stats

def sampleCrossCovariance(x,l):
    # get sample length
    T = x.shape[0]
    means = x.mean()
    # get sample
    sample = x.shift(-l).dropna()
    nolag = x.loc[sample.index,:]
    # get cross covariance
    crossCov = (nolag - means).T@(sample - means)/T
    return crossCov


def sampleCrossCorrelation(x,l):
    T = x.shape[0]
    # get cross cvoariance matrix
    crossCov = sampleCrossCovariance(x,l)
    # get sample standard dev matrix
    D = np.diag(x.std())
    # get cross correlations
    crossCorr = np.linalg.inv(D)@crossCov@np.linalg.inv(D)
    # transpose crossCorr to be alongside Tsay's notation
    crossCorr = crossCorr.T
    crossCorr.columns = x.columns
    crossCorr.index = x.columns
    # simplified cross correlations of Tiao and Box (1981)
    # I change their notation from .,- and + to 0,-1 and 1
    simpleCrossCorr = crossCorr.copy()
    simpleCrossCorr[simpleCrossCorr>=2/np.sqrt(T)] = 1
    simpleCrossCorr[simpleCrossCorr<=-2/np.sqrt(T)] = -1
    simpleCrossCorr[(simpleCrossCorr<2/np.sqrt(T))\
                    & (simpleCrossCorr>-2/np.sqrt(T))] = 0
    return crossCorr, simpleCrossCorr

def multLjungBox(x, m):
    # estimate sample cross covariance matrix
    G0 = sampleCrossCovariance(x, 0)
    G0 = G0.values
    # get sample size
    T = x.shape[0]
    # LjungBox statistic
    Q = 0
    for i in range(m):
        Gl = sampleCrossCovariance(x, i+1)
        Gl = Gl.values
        Q = Q + 1/(T-i-1)*np.trace(Gl.T@np.linalg.inv(G0)@Gl@np.linalg.inv(G0))
    Q = Q*T**2
    # estimate p values for Q
    df = x.shape[1]**2*m
    pvalue = 1-stats.chi2.cdf(Q, df)
    return Q, pvalue

def makeLags(data, l):
    # for each of column create lags
    lags = [data.iloc[:,i].shift(j+1).rename('{}lag{}'.format(data.columns[i],j+1))\
                            for i in range(data.shape[1]) for j in range(l)]
    lags = pd.concat(lags, axis=1).dropna()
    return lags