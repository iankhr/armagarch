# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:46:36 2020
This class defines t-Student distribution for ARMA-GARCH models

@author: Ian Khrashchevskyi
"""

from .distModel import DistModel
import numpy as np
import scipy.stats as stats
from scipy.special import gamma

class tStudent(DistModel):
    """
    INPUT:
        data - innovations
        params -dict with mean and Var and other as a parameter nu
    """
    def _giveName(self):
        if self._params is None:
            self._params = {'Mean':0,'Var':1, 'other':3}
        
        self._name = 'Student'
        self._startingValues = 3
        self._varnames = ['nu']
    
    
    def _setConstraints(self, data=None):
        self._constraints = {'Mean':[(-np.Inf, np.inf),], 'Var':[(0,np.inf),],\
                             'other':[(3,np.Inf),]}
    
    
    def lls(self, data =None, params = None):
        if data is None:
            data = self._data
            
        if params is None:
            params = self._params
        
        mu = params['Mean']
        var = params['Var']
        nu = params['other']
        ells = np.log(gamma((nu+1)/2)/(np.sqrt(np.pi*(nu-2))*gamma(nu/2)))\
                - 0.5*np.log(var.values) \
                - (nu+1)/2*np.log(1+(data.values-mu)**2/(var.values*(nu-2))) 
        return -ells


    def simulate(self, nobs= 1):
        """
        Use built in simulator for now
        """
        return stats.t.rvs(df = self._params['other'],\
                                loc = self._params['Mean'],\
                                scale = self._params['Var'],\
                                size = nobs)