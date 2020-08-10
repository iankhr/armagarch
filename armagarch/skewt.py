# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:47:57 2020

This class defines skewt distribution for ARMA-GARCH models

@author: Ian Khrashchevskyi
"""
from .distModel import DistModel
import numpy as np
from scipy.special import gamma

class skewt(DistModel):
    """
    INPUT:
        data - innovations
        params -dict with mean and Var and other as a parameter nu
    """
    def _giveName(self):
        if self._params is None:
            self._params = {'Mean':0,'Var':1, 'other':[3,2]}
        
        self._name = 'Student'
        self._startingValues = [3,0.5]
        self._varnames = ['nu','l']
    
    
    def _setConstraints(self, data=None):
        self._constraints = {'Mean':[(-np.Inf, np.inf),], 'Var':[(0,np.inf),],\
                             'other':[(3,np.Inf),(-0.9999,0.9999)]}
    
    
    def lls(self, data = None, params = None):
        if data is None:
            data = self._data
            
        if params is None:
            params = self._params
        
        mu = params['Mean']
        var = params['Var']
        ht = var.values
        data = data.values-mu
        nu = params['other']
        l = nu[1]
        nu = nu[0]
        a = gamma((nu+1)/2)*np.sqrt(nu-2)*(l-1/l)
        a = a/(np.sqrt(np.pi)*gamma(nu/2))
        b = np.sqrt((l**2+1/l**2-1)-a**2)
        tVar = -a/b
        IndicF = -1*(data/np.sqrt(ht)<tVar)
        IndicF = 2*IndicF+1
        IndicF = 2*IndicF
        ells = np.log(gamma((nu+1)/2)/(np.sqrt(np.pi*(nu-2))*gamma(nu/2)))\
                + np.log(b) + np.log(2/(l+1/l))\
                - 0.5*np.log(ht)\
                - 0.5*(nu+1)*np.log(1+(b*data/np.sqrt(ht)+a)**2/(nu-2)*l**IndicF)  
        return -ells