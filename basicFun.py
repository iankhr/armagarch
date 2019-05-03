# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:02:21 2018

@author: Ian
"""

"""
These are the helper functions, which are used in most of the optimization
like two-sided hessian and so on
"""
import numpy as np

def getLag(data, lag):
    # yt = data[lag:]
    Lags = np.zeros((len(data),lag))
    for i in range(lag):
        Lags[i+1:,i] = data[i:i-lag]
    
    return Lags


def _normLik(self, data, mu, ht, out=False):
        if np.any(ht<=0):
            nlogLik = np.Inf
        else:
            lls = np.log(2*np.pi) + np.log(ht) + (data-mu)**2/ht
            nlogLik = 0.5*np.sum(lls)
    
        if np.isnan(nlogLik):
            nlogLik = np.Inf
            
        if out == False:
            return nlogLik
        else:
            return nlogLik, lls


def hessian_2sided(fun,theta, args):
    """
    Taken from Kevin's Sheppard "Python for Econometrics"
    """
    f = fun(theta, *args)
    h = 1e-5*np.abs(theta)
    thetah = theta + h
    h = thetah - theta
    K = np.size(theta,0)
    h = np.diag(h)
    fp = np.zeros(K)
    fm = np.zeros(K)  
    for i in range(K):
        fp[i] = fun(theta+h[i], *args)
        fm[i] = fun(theta-h[i], *args)
    
    
    fpp = np.zeros((K,K))
    fmm = np.zeros((K,K))
    for i in range(K):
        for j in range(i,K):
            fpp[i,j] = fun(theta + h[i] + h[j], *args)
            fpp[j,i] = fpp[i,j]
            fmm[i,j] = fun(theta - h[i] - h[j], *args)
            fmm[j,i] = fmm[i,j]
    
    
    hh = (np.diag(h))
    hh = hh.reshape((K,1))
    hh = hh @ hh.T
    H = np.zeros((K,K))
    for i in range(K):
        for j in range(i,K):
            H[i,j] = (fpp[i,j] - fp[i] - fp[j] + f
            + f - fm[i] - fm[j] + fmm[i,j])/hh[i,j]/2
            H[j,i] = H[i,j]
    return H
