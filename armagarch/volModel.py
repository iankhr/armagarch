# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:19:25 2020

This file contains a template class for future volatility models 

@author: Ian Khrashchevskyi
"""

import pandas as pd
import numpy as np

class VolModel(object):
    def __init__(self, innovations = None, params=None, order = None):
        if (innovations is not None) & (type(innovations) != pd.DataFrame):
                raise TypeError('Data must be in DataFrame!')
            
        self._data = innovations
        self._order = order
        self._params = params
        self.type = 'VolModel'
        self._constraints = None
        self._giveName()
        self._setConstraints(innovations)
        pass
    
    
    def _giveName(self):
        self._name = 'Constant'
        self._pnum = 1
        self._varnames = ['Constant',]
        self._startingValues = 0
    
    
    def ht(self, params = None):
        if params is None:
            params = self._params
        
        # constant volatility implies that params equal to ht
        ht = params
        return ht
    
    
    def reconstruct(self, et, params = None, other=None):
        if params is None:
            params = self._params
        pass
    
    
    def stres(self, params=None):
        stres= self._data.values/np.sqrt(self.ht(params).values)
        return pd.DataFrame(stres, columns = [self._data.columns[0]+'stres'],\
                            index = self._data.index)
    
    
    def predict(self, nsteps, params = None, data = None, other = None):
        """
        Makes the preditiction of the variance
        """
        if params is None:
            params = self._params
            
        if data is None:
            data = self._data
            
        return self._params
    
    
    def func_constr(self, params):
        # defines functional constraints on some parameters
        return None
    

    def _setConstraints(self, data=None):
        # redefing constraints with new data
        self._constraints = None
    
    def expectedVariance(self, params):
        return np.nan
        
    @property
    def name(self):
        return self._name

    
    @property
    def params(self):
        return self._params

    
    @property
    def order(self):
        return self._order

    
    @property
    def data(self):
        return self._data
    
    
    @data.setter
    def data(self, newData):
        if (type(newData) != pd.DataFrame):
            raise TypeError('Data must be in DataFrame!')
        self._data = newData
        self._setConstraints(newData)
        
    
    @property
    def pnum(self):
        return self._pnum
    
    
    @property
    def constr(self):
        return self._constraints
    
    @property
    def varNames(self):
        return self._varnames
    
    @property
    def startVals(self):
        return self._startingValues