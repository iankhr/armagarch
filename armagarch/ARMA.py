# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:14:57 2020

ARMA class model.

Full documentation is on the way

@author: Ian Khrashchevskyi
"""

from .meanModel import MeanModel
import pandas as pd
import statsmodels.tsa.api as sm
import numpy as np
from .errors import InputError

#### Drefining different mean models here ####
class ARMA(MeanModel):
    """
    Works as it should!
    Input:
        data - Pandas dataFrame
        order - dict with AR and MA specified.
    """
    def _giveName(self):
        self._name = 'ARMA'
        # creating lags matrix            
        if self._order is None:
            raise ValueError('Order must be specified')
        else:
            # define AR
            if self._data is not None:
                if self._order['AR']>0:
                    lags = pd.concat([self._data.shift(i+1) \
                                  for i in range(self._order['AR'])], axis=1)
                    lags.columns = ['Lag'+str(i+1) for i in range(self._order['AR'])]
                    self._lags = lags.fillna(0).values
            
            if self._other is None:
                self._include_constant = True
            elif 'constant' in self._other:
                self._include_constant = self._other['constant']
            else:
                self._include_constant = True
            
            self._pnum = self._order['AR']+self._order['MA']
            self._varnames = ['AR','MA']
            if self._include_constant:
                self._pnum = self._pnum+1
                self._varnames = ['Constant'] + self._varnames
            self._setConstraints()
     
    
    def _getStartingVals(self):
        if self._data is not None:
            if self._include_constant:
                c = 'c'
            else:
                c = 'nc'
            try:
                model = sm.ARMA(self._data.values, (self._order['AR'],self._order['MA'])).fit(trend = c)
                self._startingValues = model.params
            except ValueError:
                self._startingValues = None            
        else:
            self._startingValues = np.zeros((self._pnum,))+0.5
    
    
    def _setConstraints(self):
        # redefing constraints with new data
        if self._data is None:
            self._constraints = [(-np.inf,np.inf) for i in range(self._pnum)]
        else:
            self._constraints = \
                                [(-0.99999,0.99999) for i in range(self._order['AR'])]+\
                                [(-0.99999,0.99999) for i in range(self._order['MA'])]
            if self._include_constant:
                self._constraints = [((-10*np.abs(np.mean(self._data.values)), \
                                   10*np.abs(np.mean(self._data.values)))),]\
                  + self._constraints
            
    #@profile        
    def condMean(self, params = None, data = None, other = None):
        if data is None:
            data = self._data.values
            lags = self._lags
        else:
            if self._order['AR']>0:
                lags = pd.concat([data.shift(i+1) \
                                      for i in range(self._order['AR'])], axis=1)
                lags.columns = ['Lag'+str(i+1) for i in range(self._order['AR'])]
                lags = lags.fillna(0).values
            data = data.values
            
        if params is None:
            params = self._params
            
        if self._include_constant:
            Ey = np.ones((len(data),))*params[0]
            startPointer = 1
        else:
            Ey = np.zeros((len(data),))
            startPointer = 0
            
        if self._order['AR']>0:
            # We need to calculate AR lags
            ar = params[startPointer:self._order['AR']+startPointer]
            Ey = Ey+lags@ar
        
        if self._order['MA']>0:
            ma = params[self._order['AR']+startPointer:]
            ylags = np.zeros((self._order['MA']))
            eylags = np.zeros((self._order['MA']))
            lagind = np.arange(0,len(ylags))
            lagind = np.roll(lagind,1)
            for t in range(len(data)):
                etlag = ylags - eylags
                maIncr = etlag@ma
                Ey[t] = Ey[t]+maIncr                    
                ylags = ylags[lagind]
                ylags[0] = data[t]
                eylags = eylags[lagind]
                eylags[0] = Ey[t]
        return Ey
    
    def reconstruct(self, et, params=None, other=None):
        """
        The function gets innovations and generates the process with specified
        model
        
        et must be DataFrame
        """
        if params is None:
            params = self._params
        
        # get alpha and beta
        if self._order['AR']>0:
            alpha = params[1:1+self._order['AR']]
        else:
            alpha = 0
            
        if self._order['MA']>0:
            beta = params[1+self._order['AR']:]
        else:
            beta = 0
        
        # add mean immediately
        data = np.ones(np.shape(et))*params[0]+et
        
        # create lag variables
        ytlag = np.zeros(self._order['AR'])
        etlag = np.zeros(self._order['MA'])
        lagindyt = np.arange(0,len(ytlag))
        lagindyt = np.roll(lagindyt,1)
        lagindet = np.arange(0,len(etlag))
        lagindet = np.roll(lagindet,1)
        for t in range(len(et)):
            if t>0:
                if self._order['AR']>0:
                    data[t] = data[t] + ytlag@alpha
                    ytlag = ytlag[lagindyt]
                    ytlag[0] = data[t]
            
                if self._order['MA']>0:
                    data[t] = data[t] +  etlag@beta
                    etlag = etlag[lagindet]
                    etlag[0] = et[t]
        
        return data
        
    #@profile
    def et(self, params = None, data = None, other = None):
        if data is None:
            data = self._data
            
        return data.subtract(self.condMean(params, data), axis=0)
    
    
    def predict(self, nsteps, params = None, data = None, other = None):
        """
        Makes the preditiction of the mean
        """
        if params is None:
            params = self._params
            
        if data is None:
            data = self._data
        
        if nsteps<=0:
            raise InputError('Number of forecasting steps must be positive.')
        
        if self._include_constant:
            prediction = np.ones((nsteps,))*params[0]
            startPointer = 1
        else:
            prediction = np.ones((nsteps,))*params[0]
            startPointer = 0
            
        # create AR and MA parts
        if self._order['AR']>0:
            # get the latest data
            datalags = data[-self._order['AR']:].values
            # create ar matrix
            ar = np.ones((nsteps+self._order['AR'],1))
            ar[:len(datalags)] = datalags
            arp = params[startPointer:self._order['AR']+startPointer]
        else:
            ar = 0
            
        if self._order['MA']>0:
            ey = self.condMean(params, data, other)
            et = data.subtract(ey, axis=0)
            # get the latest data
            etlags = et[-self._order['MA']:].values
            # create ar matrix
            ma = np.zeros((nsteps+self._order['MA'],1))
            ma[:len(etlags)] = etlags
            mapars = params[self._order['AR']+startPointer:]
        else:
            ma = 0
        
        # iterate and get predictions
        for t in range(nsteps):
            # add MA and AR components
            if self._order['MA']>0:
                prediction[t] = prediction[t] + ma[t:t+self._order['MA']][::-1].T@mapars
                
            if self._order['AR']>0:
                try:
                    prediction[t] = prediction[t] + ar[t:t+self._order['AR']][::-1].T@arp
                except:
                    print(arp)
                    print(ar[t:t+self._order['AR']][::-1])
                    print(ar[t:t+self._order['AR']][::-1]@arp)
                    raise ValueError
                    
                if nsteps>1:
                    ar[t+self._order['AR']] = prediction[t]
                        
        return prediction
    
    #@profile
    def func_constr(self, params):
        # testing for unit circle
        constr = []
        if self._order['AR']>0:
            arparams = params[1:self._order['AR']+1]
            ar = np.r_[1, -arparams]
        else:
            ar = np.r_[1, 0]
            
        if self._order['MA']>0:
            maparams = params[self._order['AR']+1:]
            ma = np.r_[1, maparams]
        else:
            ma = np.r_[1,0]
        
        testP = sm.ArmaProcess(ar,ma)
        testInvertible = testP.isinvertible
        testStationary = testP.isstationary
        constr = [(testInvertible*testStationary)*2-1,]
        
        return constr