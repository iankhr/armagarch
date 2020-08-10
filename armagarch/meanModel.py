# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:03:59 2020

MeanModel class is a template class for any mean models

Full documentation is coming...

@author: Ian Khrashchevskyi
"""
import pandas as pd


class MeanModel(object):
    def __init__(self, data = None, params = None, order = None, other = None):
        """
        Creates the class
        """
        if params is None:
            # define simple zero constant model
            params = 0
                    
        if data is not None:
            if (type(data) != pd.DataFrame):
                raise TypeError('Data must be in DataFrame!')
            
        self._data = data
        self._params = params
        self._other = other
        self._order = order
        self.type = 'MeanModel'
        self._constraints = None
        self._giveName()
        self._setConstraints()
        self._getStartingVals()
        pass
    
    
    def reconstruct(self, et):
        pass
    
    
    def _getStartingVals(self):
        pass
    
    
    def _giveName(self):
        self._name = 'Constant'
        self._varnames = ['Constant',]
        self._pnum = 1
        self._startingValues = 0
        
    
    def condMean(self, params = None, other = None, data = None):
        """
        Estimates conditional mean of the model
        """
        if params is None:
            return self._params
        else:
            return params
    
    
    def et(self, params = None, other = None, data = None):
        """
        Estimates residuals from the model
        """
        if params is None:
            return self._data - self._params
        else:
            return self._data - params    
       
        
    def predict(self, nsteps, params = None, data = None, other = None):
        """
        Makes the preditiction of the mean
        """
        if params is None:
            params = self._params
            
        if data is None:
            data = self._data
            
        return self._params
    
    
    def func_constr(self, params):
        # defines functional constraints on some parameters
        return None
    
    
    def _setConstraints(self):
        # redefing constraints with new data
        self._constraints = None
    
    
    @property
    def params(self):
        return self._params
    
    
    @property
    def data(self):
        return self._data
    
    
    @data.setter
    def data(self, newData):
        if (type(newData) != pd.DataFrame):
            raise TypeError('Data must be in DataFrame!')
        self._data = newData
        self._giveName()
        self._setConstraints()
        self._getStartingVals()
        
    
    @property
    def name(self):
        return self._name
    
    
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
