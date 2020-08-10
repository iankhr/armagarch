# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:23:59 2020

This class is a template for future distribution models

@author: Ian Khrashchevskyi
"""

import numpy as np

class DistModel(object):
    """

        Parameters
        ----------
        params : dict, optional
            params is a dict that contains all of the . The default is None.
        data : DataFrame, optional
            data is a DataFrame with the values to be used in . The default is None.

        Returns
        -------
        None.

    """
    def __init__(self, params=None, data = None):
        
        self._data = data
        self._params = params
        self.type = 'DistModel'
        self._giveName()
        self._setConstraints()
        pass
    
    
    def _setConstraints(self, data=None):
        self._constraints = {'Mean':(-np.Inf, np.inf), 'Var':(0,np.inf),\
                             'other':None}
    
    
    def _giveName(self):
        self._name = 'Dist'
        self._pnum = 1
        self._varnames = ['Mean','Var']
        self._startingValues = 0
        pass
    
    
    def func_constr(self, params):
        """
        """
        return None
    
    
    def lls(self, data=None, params=None):
        pass
    
    
    def ll(self, data = None, params = None):
        return np.sum(self.lls(data, params))
    
    
    def cdf(self):
        pass
    
    
    def pdf(self):
        pass
    
    
    def invcdf(self):
        pass
    
    
    def invpdf(self):
        pass
    
    
    def simulate(self, nobs = 1):
        pass
    
    @property
    def reqpars(self):
        return self._reqPars
    
    
    @property
    def name(self):
        return self._name
    
    
    @property
    def params(self):
        return self._params
    
    
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