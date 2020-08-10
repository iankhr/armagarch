# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:17:36 2020

ARMA model with additional regressors

Full documentation is coming

@author: Ian Khrashchevskyi
"""

from .meanModel import MeanModel
import numpy as np
from .errors import InputError
from .ARMA import ARMA

class regARMA(MeanModel):
    """
    INPUT:
        dict of the following form:
            AR - number of AR lags
            MA - number of MA lags
            regCols - list of the columns in data to be used as regressors
            y - column to be used as y in the formula.
    """
    def _giveName(self):
        self._name = 'ARMA with Regressors'
        # creating lags matrix            
        if self._order is None:
            raise ValueError('Order must be specified')
        else:
            # getting regressors values
            self._regCols = self._other['regdata'].columns
            self._regs = self._other['regdata']
            # adding to order columns
            for item in self._regCols:
                self._order[item] = 1
            # getting additional ARMA model
            if self._data is not None:
                # define data and regressors
                self._coreARMA = ARMA(order = self._order, data = self._data, other = self._other)
            else:
                self._coreARMA = ARMA(order = self._order, other=self._other)
            self._pnum = 1+self._order['AR']+self._order['MA']+len(self._regCols)
            self._varnames = ['Constant','AR','MA']+list(self._regCols)
            self._setConstraints()


    def _getStartingVals(self):
        if self._data is not None:
            # get starting values from ARMA model
            self._startingValues = self._coreARMA._startingValues
            # add starting values for the regressors from OLS
            X = np.array([np.ones(len(self._regs)), np.reshape(self._regs.values,(len(self._regs),))]).T
            Y = self._data.values
            beta = np.linalg.inv(X.T@X)@X.T@Y
            self._startingValues = np.concatenate((self._startingValues,np.reshape(beta[1:],(len(beta[1:]),))))
        else:
            self._startingValues = np.zeros((self._pnum,))+0.5
    
    
    def _setConstraints(self):
        # redefing constraints with new data
        if self._data is None:
            self._constraints = [(-np.inf,np.inf) for i in range(self._pnum)]
        else:
            self._coreARMA.data = self._data
            self._constraints = self._coreARMA.constr
            # getting our own constraints
            self._constraints = self._constraints + \
                                [(-np.Inf, np.Inf) for i in range(len(self._regCols))]
            
    #@profile        
    def condMean(self, params = None, data = None, other = None):
        if data is None:
            data = self._data
        else:
            data = data

        if other is None:
            regs = self._regs.values
        else:
            regs = other['regdata'].values
            
        if params is None:
            params = self._params
            
        # extract parameters for ARMA
        parARMA = params[:1+self._order['AR']+self._order['MA']]
        parreg = params[1+self._order['AR']+self._order['MA']:]
        Ey = self._coreARMA.condMean(params = parARMA, data = data)
        # adding expectations from regression
        Ey = Ey + regs@parreg
        return Ey
    
    def reconstruct(self, et, params=None, other = None):
        """
        Reconstrcuts data based on regressors and innovations
        """
        if params is None:
            params = self._params
        
        if other is None:
            regdata = self._regs.values
        else:
            regdata = other['regdata'].values
            
        # extract parameters for ARMA
        parARMA = params[:1+self._order['AR']+self._order['MA']]
        #parARMA = parARMA.reshape((len(parARMA),1))
        parreg = params[1+self._order['AR']+self._order['MA']:]
        #parreg = parreg.reshape((len(parreg),1))
        regs = et+regdata@parreg
        return self._coreARMA.reconstruct(params = parARMA, et = regs)
        
    
    #@profile
    def et(self, params = None, data = None, other = None):
        if data is None:
            data = self._data    
        return data.subtract(self.condMean(params, data, other = other), axis=0)
    
    
    #@profile
    def func_constr(self, params):
        parARMA = params[:1+self._order['AR']+self._order['MA']]
        # just using ARMA contraint here
        constr = self._coreARMA.func_constr(parARMA)        
        return constr
    
    def predict(self, nsteps, other, params = None, data = None):
        """
        Makes the preditiction of the mean
        """
        if params is None:
            params = self._params
            
        if data is None:
            data = self._data
            
        if other is None:
            raise InputError('Regressors are not specified.')
        else:
            regressors = other['regdata']
        
        if len(regressors) != nsteps:
            raise InputError('Length of regressors do not coincide with number of steps')
            
        # extract parameters for ARMA
        parARMA = params[:1+self._order['AR']+self._order['MA']]
        parreg = params[1+self._order['AR']+self._order['MA']:]
        Ey = self._coreARMA.predict(nsteps = nsteps, params = parARMA, data = data)
        #Ey = self._coreARMA.predict(nsteps = nsteps, params = parARMA, data = (data-np.reshape(regs@parreg, (len(regs@parreg),1))))
        # adding expectations from regression
        Ey = Ey + regressors@parreg
        return Ey