# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:24:58 2020

This class defines normal distribution for ARMA-GARCH models

@author: Ian Khrashchevskyi
"""
from .distModel import DistModel
import numpy as np
import pandas as pd
import scipy.stats as stats

class normalDist(DistModel):
    """
    Works as it should!
    INPUT:
        data - innovations
        params -dict with mean and Var
    """
    def _giveName(self):
        if self._params is None:
            self._params = {'Mean':0,'Var':1}
            
        self._name = 'Gaussian'
        self._startingValues = None
        
    
    #@profile
    def lls(self, data=None, params = None):       
        if data is None:
            data = self._data
        
        if params is None:
            params = self._params
            
        mu = params['Mean']
        var = params['Var']
        if (type(var) != pd.Series) & (type(var) != pd.DataFrame):
            ells = np.log(2*np.pi) + np.log(var) + (data.values-mu)**2/var
        else:
            ells = np.log(2*np.pi) + np.log(var.values) + (data.values-mu)**2/var.values
        ells = 0.5*ells
        return ells

    
    def simulate(self, nobs= 1):
        """
        Use built in simulator for now
        """
        return stats.norm.rvs(loc = self._params['Mean'],\
                                scale = self._params['Var'],\
                                size = nobs)
        