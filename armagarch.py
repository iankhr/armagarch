# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 18:24:05 2018

@author: Ian
"""

import numpy as np
import pandas as pd
import scipy
import shutil
import datetime
import statsmodels.tsa.api as sm
from scipy.special import gamma

"""
Declaring custom errors to be used in the object below
"""

class Error(Exception):
    pass


class InputError(Error):
    def __init__(self, message):
        self.message = message

class ProcedureError(Error):
    def __init__(self, message):
        self.message = message
        
class HessianError(Error):
    def __init__(self, message):
        self.message = message


"""
MeanModel specificies the model of the mean and has to return innovations. 

data is DataFrame 
"""
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
            
            self._pnum = 1+self._order['AR']+self._order['MA']
            self._varnames = ['Constant','AR','MA']
            self._setConstraints()
     
    
    def _getStartingVals(self):
        if self._data is not None:
            model = sm.ARMA(self._data.values, (self._order['AR'],self._order['MA'])).fit()
            self._startingValues = model.params
        else:
            self._startingValues = np.zeros((self._pnum,))+0.5
    
    
    def _setConstraints(self):
        # redefing constraints with new data
        if self._data is None:
            self._constraints = [(-np.inf,np.inf) for i in range(self._pnum)]
        else:
            self._constraints = [((-10*np.abs(np.mean(self._data.values)), \
                                   10*np.abs(np.mean(self._data.values)))),]+\
                                [(-0.99999,0.99999) for i in range(self._order['AR'])]+\
                                [(-0.99999,0.99999) for i in range(self._order['MA'])]
            
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
            
        Ey = np.ones((len(data),))*params[0]
        if self._order['AR']>0:
            # We need to calculate AR lags
            ar = params[1:self._order['AR']+1]
            Ey = Ey+lags@ar
        
        if self._order['MA']>0:
            ma = params[self._order['AR']+1:]
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
            
        # take the msot of the latest part of the data we need to make prediction
        ey = self.condMean(params, data, other)
        et = data.subtract(ey, axis=0)
        ey = ey.values[-self._order['AR']:][::-1]
        et = et.values[-self._order['MA']:][::-1]
        # making the prediction (Works similar as CondMean, but with a twist)
        prediction = np.ones((nsteps,))*params[0]
        laginde = np.roll(np.arange(0,self._order['MA']),1)
        lagindy = np.roll(np.arange(0,self._order['AR']),1)
        ar = params[1:self._order['AR']+1]
        ma = params[self._order['AR']+1:]
        for t in range(nsteps):
            # add AR components
            if self._order['AR']>0:
                prediction[t] = prediction[t] + ey@ar
                ey = ey[lagindy]
            
            if self._order['MA']>0:
                prediction[t] = prediction[t] + et@ma
                et = et[laginde]
            
            ey[0] = prediction[t]
            et[0] = 0
        
        """
        ATTENTION: IT WILL NOT WORK FOR AR>1 or MA>1 !!!! NEEDS TO BE PATCHED!
        """
        
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
    
    def predict(self, nsteps, params = None, data = None):
        """
        Makes the preditiction of the mean
        """
        if params is None:
            params = self._params
            
        if data is None:
            data = self._data
            
        return self._params
        
"""
VolModel specifies volatility model and has tpo return standardized innovations
By default it's constant
"""
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
    
    
    def stres(self, params=None):
        stres= self._data.values/np.sqrt(self.ht(params).values)
        return pd.DataFrame(stres, columns = [self._data.columns[0]+'stres'],\
                            index = self._data.index)
    
    
    def predict(self, nsteps, params = None, data = None):
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

# specifying GARCH (1,1)
class garch(VolModel):
    """
    Works as it should!
    Input:
        data- pandas
        order - dict
    """
    def _etLags(self, et):
        if self._order['p']>0:
            lags = pd.concat([et.shift(i+1) \
                                  for i in range(self._order['p'])], axis=1)
            lags.columns = ['Lag'+str(i+1) for i in range(self._order['p'])]
            etlags = lags.fillna(0).values
            return etlags
    
    
    def _giveName(self):
        self._name = 'GARCH'
        if self._order is None:
            raise ValueError('Order must be specified!')
        else:
            self._pnum = 1 + self._order['p']+self._order['q']
            self._varnames = ['omega','alpha','beta']
            # regardless of the model assume that it's GARCH (1,1)
            self._startingValues = np.hstack((0.001,0.1,\
                                              np.zeros((np.max((0,(self._order['p']-1)))),),\
                                              0.8, 
                                              np.zeros((np.max((0,(self._order['q']-1)))),)))
            # get et lags
            if self._data is not None:
                self._etlags = self._etLags(self._data)
        
        
    #@profile
    def _setConstraints(self, data=None):
        # defining constraints
        finfo = np.finfo(np.float64)
        if data is None:
            self._constraints = [(finfo.eps, np.inf),]+\
                            [(finfo.eps, 0.999999999) for i in range(self._order['p'])] +\
                            [(finfo.eps, 0.999999999) for i in range(self._order['q'])]
        else:
            self._constraints = [(finfo.eps, 2*np.var(self._data.values)),]+\
                            [(finfo.eps, 0.999999999) for i in range(self._order['p'])] +\
                            [(finfo.eps, 0.999999999) for i in range(self._order['q'])]
            self._etlags = self._etLags(self._data)

    
    #@profile
    def ht(self, params = None, data = None):
        if params is None:
            params = self._params
            
        if data is None:
            data = self._data
            etlags = self._etlags
        else:
            etlags = self._etLags(data)
        
        # unpack the parameters
        omega = params[0]
        # unpack alpha
        if self._order['p']>0:
            alpha = params[1:1+self._order['p']]
        else:
            alpha = 0
        
        # unpack beta
        if self._order['q']>0:
            beta = params[1+self._order['p']:]
        else:
            beta = 0
        
        # specify how you estimate it
        ht = np.ones(len(data))*omega
        ht[0] = np.mean(data**2)
        if self._order['p']>0:
            ht = ht+(etlags**2)@alpha
            
        # add the lagged volatility
        if self._order['q']>0:
            htlag = np.zeros((self._order['q']))
            lagind = np.arange(0,len(htlag))
            lagind = np.roll(lagind,1)
            for t in range(len(ht)):
                betaC = htlag@beta
                ht[t] = ht[t] + betaC
                htlag = htlag[lagind]
                htlag[0] = ht[t]
        
        ht = pd.DataFrame(ht, columns = [str(data.columns[0])+'Vol',], \
                          index = data.index)
        # embedding non negativity contraints
        if np.any(ht<0):
            ht = 0
        
        if np.any(params<0):
            ht = 0
        
        return ht
 
    
    def predict(self, nsteps, params = None, data = None):
        """
        Makes the preditiction of the variance
        """
        if params is None:
            params = self._params
            
        if data is None:
            data = self._data
            etlags = self._etlags
        else:
            etlags = self._etLags(data)
            
        # take the msot of the latest part of the data we need to make prediction
        ht = self.ht(params, data)
        
        # unpack the parameters
        omega = params[0]
        # unpack alpha
        if self._order['p']>0:
            alpha = params[1:1+self._order['p']]
        else:
            alpha = 0
        
        # unpack beta
        if self._order['q']>0:
            beta = params[1+self._order['p']:]
        else:
            beta = 0

        # making the prediction (Works similar as CondMean, but with a twist)
        prediction = np.ones((nsteps,))*omega
        laginde = np.roll(np.arange(0,self._order['p']),1)
        lagindht = np.roll(np.arange(0,self._order['q']),1)
        et = etlags[-self._order['p']:,0][::-1]
        htlags = ht[-self._order['q']:][::-1]
        for t in range(nsteps):
            if self._order['p']>0:
                prediction[t] = prediction[t] + (et**2)@alpha
                et = et[laginde]
            
            if self._order['q']>0:
                prediction[t] = prediction[t] + htlags@beta
                htlags = et[lagindht]
            
            ht[0] = prediction[t]
            et[0] = np.sqrt(prediction[t])
        
        return prediction
    
    #@profile
    def func_constr(self, params):
        # unpack alpha
        if self._order['p']>0:
            alpha = params[1:1+self._order['p']]
        else:
            alpha = 0
        
        # unpack beta
        if self._order['q']>0:
            beta = params[1+self._order['p']:]
        else:
            beta = 0
            
        return [0.999-np.sum(alpha)-np.sum(beta)]

    
class gjr(VolModel):
    """
    Works as it should!
    Input:
        data- pandas
        order - dict
    """
    def _etLags(self, et):
        maxLag = np.max((self._order['p'],self._order['o']))
        if maxLag>0:
            lags = pd.concat([et.shift(i+1) \
                                  for i in range(maxLag)], axis=1)
            lags.columns = ['Lag'+str(i+1) for i in range(maxLag)]
            etlags = lags.fillna(0).values
            return etlags
    
    
    def _giveName(self):
        self._name = 'GJR-GARCH'
        if self._order is None:
            raise ValueError('Order must be specified!')
        else:
            try:
                self._pnum = 1 + self._order['p']+self._order['o']+self._order['q']
            except:
                raise ValueError('Order was not specified correctly')
            self._varnames = ['omega','alpha','gamma','beta']
            # regardless of the model assume that it's GARCH (1,1)
            self._startingValues = np.hstack((0.001,0.1,\
                                              np.zeros((np.max((0,(self._order['p']-1)))),),\
                                              0.01,
                                              np.zeros((np.max((0,(self._order['o']-1)))),),\
                                              0.8, 
                                              np.zeros((np.max((0,(self._order['q']-1)))),)))
            # get et lags
            if self._data is not None:
                self._etlags = self._etLags(self._data)
        
        
    #@profile
    def _setConstraints(self, data=None):
        # defining constraints
        finfo = np.finfo(np.float64)
        if data is None:
            self._constraints = [(finfo.eps, np.inf),]+\
                            [(finfo.eps, 0.999999999) for i in range(self._order['p'])] +\
                            [(-0.999999999, 0.999999999) for i in range(self._order['o'])] +\
                            [(finfo.eps, 0.999999999) for i in range(self._order['q'])]
        else:
            self._constraints = [(finfo.eps, 2*np.var(self._data.values)),]+\
                            [(finfo.eps, 0.999999999) for i in range(self._order['p'])] +\
                            [(-0.999999999, 0.999999999) for i in range(self._order['o'])] +\
                            [(finfo.eps, 0.999999999) for i in range(self._order['q'])]
            self._etlags = self._etLags(self._data)
    
    #@profile
    def ht(self, params = None, data = None):
        if params is None:
            params = self._params
            
        if data is None:
            data = self._data
            etlags = self._etlags
        else:
            etlags = self._etLags(data)
        
        # unpack the parameters
        omega = params[0]
        # unpack alpha
        if self._order['p']>0:
            alpha = params[1:1+self._order['p']]
        else:
            alpha = 0
        
        # unpack gamma
        if self._order['o']>0:
            gamma = params[1+self._order['p']:1+self._order['p']+self._order['o']]
        else:
            gamma = 0
        
        # unpack beta
        if self._order['q']>0:
            beta = params[1+self._order['p']+self._order['o']:]
        else:
            beta = 0
        
        # specify how you estimate it
        ht = np.ones(len(data))*omega
        ht[0] = np.mean(data**2)
        etlagsp = etlags[:,:self._order['p']]**2
        if self._order['p']>0:
            ht = ht+etlagsp@alpha
        
        # adjusting for gamma
        if self._order['o']>0:
            etlagso = etlags[:,:self._order['o']]
            etlagso[etlagso>0] = 0
            etlagso = etlagso**2
            ht = ht+etlagso@gamma
        
        # add the lagged volatility
        if self._order['q']>0:
            htlag = np.zeros((self._order['q']))
            lagind = np.arange(0,len(htlag))
            lagind = np.roll(lagind,1)
            for t in range(len(ht)):
                betaC = htlag@beta
                ht[t] = ht[t] + betaC
                htlag = htlag[lagind]
                htlag[0] = ht[t]
        
        ht = pd.DataFrame(ht, columns = [str(data.columns[0])+'Vol',], \
                          index = data.index)
        # embedding non negativity contraints
        ht[ht<0] = 0.000000001        
        return ht


    def predict(self, nsteps, params = None, data = None):
        """
        Makes the preditiction of the variance
        """
        if params is None:
            params = self._params
            
        if data is None:
            data = self._data
            
        return self._params
    
    
    #@profile
    def func_constr(self, params):
        # unpack alpha
        if self._order['p']>0:
            alpha = params[1:1+self._order['p']]
        else:
            alpha = 0
        
        # unpack gamma
        if self._order['o']>0:
            gamma = params[1+self._order['p']:1+self._order['p']+self._order['o']]
        else:
            gamma = 0
        
        # unpack beta
        if self._order['q']>0:
            beta = params[1+self._order['p']+self._order['o']:]
        else:
            beta = 0
            
        return [0.999-np.sum(alpha)-np.sum(beta)-0.5*np.sum(gamma)]
"""
Class Dist specifies distribution and hence log-likelihood function to use
""" 
class DistModel(object):
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
        # defines functional constraints on some parameters
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
    
    
    def simulate(self):
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

"""
Define Normal univariate distribution
"""
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
        ells = np.log(2*np.pi) + np.log(var.values) + (data.values-mu)**2/var.values
        ells = 0.5*ells
        return ells
        
    
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
"""
Class empModel combines previous classes in one class
"""
class empModel(object):
    def __init__(self, Data, Mean, Vol, Dist, startingVals = None):
        # Data
        self._data = Data           
        # Models as defined in classes above
        self._mean = Mean
        self._vol = Vol
        # gives ll and also adds extra variables to estimate
        self._dist = Dist
        self._startingVals = startingVals
        self._params = None
        # only Maximum log-likelihood is implemented for now
        self._method = 'ML'
        
    #@profile
    def _parUnpack(self, params):
        # getting total amounts of parameters to estimate
        nmpars = self._mean.pnum
        nvpars = self._vol.pnum
        # now unpacl all of the parameters using the mask
        mpars = params[:nmpars]
        vpars = params[nmpars:nmpars+nvpars]
        # everything rest must be distibution parameters
        dpars = params[nmpars+nvpars:]
        return [mpars, vpars, dpars]
    
    #@profile
    def apply(self, params = None, data = None, other = None):
        """
        Applies model on new or existing data
        """
        if data is None:
            data = self._data
        if params is None:
            if self._params is None:
                raise ValueError('The model parameters are not specified')
            else:
                params = self._params
        
        # unpack the parameters
        pars = self._parUnpack(params)
        # apply the mean model and get the residuals
        et = self._mean.et(params = pars[0], data = data, other = other)
        Ey = data-et
        # now estimate conditional volatility
        ht = self._vol.ht(pars[1], data = et)
        stres = et.values/np.sqrt(ht.values)
        return {'Ey':Ey, 'et':et, 'ht':ht, 'stres':stres, 'pars':pars}
    
    #@profile
    def _optApply(self, params, out = False):
        # get the parameters from the model
        vals = self.apply(params)
        # feed it into the ll function
        distpars = {'Mean':0, 'Var':vals['ht'], 'other':vals['pars'][2]}
        lls = self._dist.lls(data = vals['et'], params = distpars)
        logLik = np.sum(lls)
        # stability precautions
        if np.isnan(logLik):
            logLik = np.Inf
        if np.any(vals['ht']<0):
            logLik = np.Inf
        #print(list(params) + [-np.sum(lls),])
        if out == True:
            return logLik, lls
        else:
            return logLik
    
    #@profile
    def _optConst(self, params):
        vals = self._parUnpack(params)
        # get the functional constraints from mean
        constrMean = self._mean.func_constr(vals[0])
        # get the functional constraints from Vol
        constrVol = self._vol.func_constr(vals[1])
        # get the functional constraints from Dist
        constrDist = self._dist.func_constr(vals[2])
        # creating constraints from the results
        constr = []
        if constrMean is not None:
            constr = constr + constrMean
        if constrVol is not None:
            constr = constr + constrVol
        if constrDist is not None:
            constr = constr + constrDist
        
        if len(constr)>0:
            return np.array(constr)
        else: 
            return 1
    
    #@profile
    def _optBounds(self):
        # just to get bounds going on
        self._mean.data = self._data
        self._vol.data = self._data
        self._dist.data = self._data
        # get bounds from mean model
        boundsMean = self._mean.constr
        # get bounds from vol model
        boundsVol = self._vol.constr
        # get bounds from dist model
        boundsDist = self._dist.constr
        # creating bounds from the results
        bounds = []
        # parsing bounds from mean
        if boundsMean is not None:
            bounds = bounds + boundsMean
        # parsing bounds from vol
        if boundsVol is not None:
            bounds = bounds + boundsVol
        # parsing bounds from dist
        if boundsDist['other'] is not None:
            bounds = bounds + boundsDist['other']
        
        return bounds
        
    #@profile        
    def fit(self, startingVals = None, epsilon = 1e-6, acc = 1e-7, iterations = 100, iprint =0,\
            printTable = True, estimatesOnly = False):
        # setting starting values for optimizer
        if startingVals is None:
            if self._startingVals is None:
                startingVals = self._startVals()         
            else:
                startingVals = self._startingVals
        
        #print(startingVals)
        # define bounds for parameters (model dependent)
        bounds = self._optBounds()
        # running the optimizer
        self._params, self._finalLL, self._optIts, \
        self._imode, self._smode = scipy.optimize.fmin_slsqp(self._optApply,
                                            startingVals,
                                            f_ieqcons = self._optConst,
                                            bounds = bounds,
                                            epsilon=epsilon, 
                                            acc = acc, iter=iterations,
                                            iprint = iprint, 
                                            full_output = True)
        # run apply model with estimated parameters and safe the results
        self._mdlRes = self.apply()
        # get information criteria and r-squared
        self._ICs()
        if estimatesOnly == False:
            # get vcv matrix
            self._vcv = self._getvcv()      
            # print table
            if printTable == True:
                self.summary()
    
    #@profile
    def summary(self):
        """
        The model assumes that you have constant included all the time.
        You need to adjust it in future. To adjust you need to take away
        [1,] from reps when it's feeded into TableOutput function
        """
        params = self._params
        vcv = self._vcv
        pars = self._mdlRes['pars']
        output = np.vstack((params,np.sqrt(np.diag(vcv)),params/np.sqrt(np.diag(vcv)))).T
        meanParams = output[:len(pars[0]),:]
        volParams = output[len(pars[0]):len(pars[0])+len(pars[1]),:]
        distParams = output[len(pars[0])+len(pars[1]):,:]                
        tab = 4
        columns = shutil.get_terminal_size().columns
        # printing the upper body
        title = self._mean.name +'-'+self._vol.name+ " estimation results"
        print(title.center(columns))
        print('='*columns)
        smallCol = columns/2-tab
        sts = self._smallStats()
        for i in range(8):
            item1 = sts[i]
            item2 = sts[i+8]
            print(self._cellStr(item1[0], item1[1], smallCol) + tab*' '
                  + self._cellStr(item2[0], item2[1], smallCol))
        
        # printing the mean model
        ocl = (columns)/4-tab
        if len(meanParams)>0:
            print(' '*columns)
            print('Mean Model'.center(columns))
            print('='*columns)        
            self._tableOutput(meanParams, self._mean.varNames, 
                                  [1,]+list(self._mean._order.values()), tab, ocl)
            
        # printing the volatility model
        if len(volParams)>0:
            print(' '*columns)
            print('Volatility Model'.center(columns))
            print('='*columns)
            self._tableOutput(volParams, self._vol.varNames, 
                              [1,]+list(self._vol._order.values()), tab, ocl)
                    
        if self._dist.constr['other'] is not None:
            print(' '*columns)
            print(('Distribution: '+self._dist.name).center(columns))
            print('='*columns)
            self._tableOutput(np.atleast_2d(distParams), self._dist.varNames, np.ones(len(pars[2])), tab, ocl)
        
        print('='*columns)
        print('Covariance estimator: robust')
        if self._imode !=0:
            print('Warning! '+self._smode)
    
    
    def predict(self, nsteps=1):
        """
        Makes prediction based on the fitted model
        """
        if nsteps<=0:
            raise ValueError('Number of steps must be a positive number!')
        
        # get prediction of the mean
        mPred = self._mean.predict(nsteps)
        # get prediction for volatility
        vPred = self._vol.predict(nsteps)
        return [mPred, vPred]
    
    def simulate(self, nobs, params = None):
        """
        Needs to be done
        """
        if params is None:
            params = self._params
            if params is None:
                raise ValueError('Parameters of the model must be specified!')
        
        if nobs<=0:
            raise ValueError('Number of observations must be positive number!')
        pass
    
    def _startVals(self):
        """
        Needs to be adjusted
        """
        self._mean.data = self._data
        meanStart = self._mean.startVals
        volStart = self._vol.startVals
        distStart = self._dist.startVals
        startVals = np.hstack((meanStart, volStart, distStart))
        startVals = startVals[startVals != np.array(None)]
        return startVals
    
    #@profile
    def _ICs(self):
        # estimate Information Criteria
        k = len(self._params)
        n = len(self._data)
        L = -self._finalLL
        self._AIC = 2*k - 2*L
        if n/k<40:
            self._AIC = self._AIC + 2*k*(k+1)/(n-k-1)
        
        self._BIC = np.log(n)*k - 2*L
        self._HQIC = -2*L + (2*k*np.log(np.log(n)))
        self._SIC = -2*L + np.log(n+2*k)
        # estimate adjusted R^2
        pars = self._mdlRes['pars']
        Ey = self._mdlRes['Ey']
        self._rsq = np.sum((Ey-np.mean(self._data))**2)/np.sum((self._data-np.mean(self._data))**2)
        self._rsq = self._rsq.values[0]
        self._adjrsq =1-(1-self._rsq)*(len(self._data)-1)/(len(self._data)-(len(pars[0])-1)-1)
    
    #@profile
    def _getvcv(self):
        parameters =self._params
        data = self._data      
        T = len(data)
        step = 1e-5 * parameters
        scores = np.zeros((T,len(parameters)))
        for i in range(len(parameters)):
            h = step[i]
            delta = np.zeros(len(parameters))
            delta[i] = h
            
            loglik, logliksplus = self._optApply(parameters + delta, out=True)
            
            loglik, logliksminus = self._optApply(parameters - delta,  out=True)
            scores[:,i] = (logliksplus[:,0] - logliksminus[:,0])/(2*h)
            
        I = (scores.T @ scores)/T
        args = ()
        J = self._hessian_2sided(self._optApply, parameters, args)
        J = J/T
        vcv = np.eye(len(parameters))
        try:
            Jinv = np.mat(np.linalg.inv(J))
            vcv = Jinv*np.mat(I)*Jinv/T
            vcv = np.asarray(vcv)
            return vcv
        except:
            raise HessianError('Hessian is singular! St.errors are not calculated')
    
    #@profile
    def _tableOutput(self, output, rowNames, reps, tab, ocl):
        columns = shutil.get_terminal_size().columns
        poq = np.cumsum(reps)
        pointer = 0
        counter = 0
        print(int(ocl)*' '+tab*' '
              + ' '*int(ocl-len('Estimate'))+'Estimate' +tab*' '
              + ' '*int(ocl-len('Std. Error'))+'Std. Error'+tab*' '
              + ' '*int(ocl-len('t-stat'))+'t-stat'
              )
        print('-'*columns)
        # remove names with zero reps       
        # build the table
        if np.shape(output)[1]>1:
            for i in range(len(output)):
                item = np.round(output[i], decimals = 3)
                # creating name
                if i>= poq[pointer]: 
                    pointer = pointer+1
                    while reps[pointer] == 0:
                        pointer = pointer+1
                    
                    if reps[pointer]>1:
                        counter = counter+1
                    else:
                        counter = 0
                
                elif counter >0:
                    counter = counter+1
                
                if counter == 0:
                    rowName = rowNames[pointer]
                else:
                    rowName = rowNames[pointer]+'['+str(counter)+']'
                
                tabLenName = ' '*int(ocl-len(str(rowName)))
                # putting the values
                est = str(item[0])
                se = str(item[1])
                tstat = str(item[2])
                print(str(rowName)+tabLenName+tab*' '
                      +' '*int(ocl-len(est)) + est+ tab*' '
                      +' '*int(ocl-len(se)) + se+tab*' '
                      +' '*int(ocl-len(tstat)) + tstat)
        else:
            tabLenName = ' '*int(ocl-len(str(rowName)))
            # putting the values
            est = str(output[0])
            se = str(output[1])
            tstat = str(output[2])
            print(str(rowName)+tabLenName+tab*' '
                  +' '*int(ocl-len(est)) + est+ tab*' '
                  +' '*int(ocl-len(se)) + se+tab*' '
                  +' '*int(ocl-len(tstat)) + tstat)

    #@profile    
    def _smallStats(self):
        data = self._data
        ll = -self._finalLL
        sts = []
                            
        sts.append(['Dep Variable', str(self._data.columns[0])])
        sts.append(['Mean Model', self._mean.name])
        sts.append(['Vol Model', self._vol.name])
        sts.append(['Distribution',self._dist.name])
        sts.append(['Method', self._method])
        sts.append(['', ''])
        now = datetime.datetime.now()
        sts.append(['Date', now.strftime("%a, %b %d %Y")])
        sts.append(['Time', now.strftime("%H:%M:%S")])        
        sts.append(['R-squared', str(np.round(self._rsq, decimals=2))])
        sts.append(['Adj. R-squared', str(np.round(self._adjrsq, decimals=2))])
        sts.append(['Log Likelihood', str(np.round(ll, decimals=2))])
        sts.append(['AIC', str(np.round(self._AIC, decimals=2))])
        sts.append(['BIC', str(np.round(self._BIC, decimals=2))])
        sts.append(['Num obs', str(len(data))])
        sts.append(['Df Residuals', str(len(data)-len(self._params))])
        sts.append(['Df Model', str(len(self._params))])
        return sts    
        
    #@profile
    def _cellStr(self, cellName, cellContent, length):
        resLen = int(length - len(cellName) - len(cellContent))
        if cellName !='':
            cellName = cellName+':'     
            if resLen<0:
                return cellName+' '+cellContent
            else:
                return cellName+' '*resLen + cellContent
        
        else:
            return ' '*int(length)+' '
    
    #@profile
    def _hessian_2sided(self,fun,theta, args):
        f = fun(theta, *args)
        h = 1e-5*np.abs(theta)
        thetah = theta + h
        h = thetah - theta
        K = np.size(theta,0)
        h = np.diag(h)
        fp = np.zeros(K)
        fm = np.zeros(K)  
        hh = (np.diag(h))
        hh = hh.reshape((K,1))
        hh = hh @ hh.T
        H = np.zeros((K,K))
        fpp = np.zeros((K,K))
        fmm = np.zeros((K,K))
        for i in range(K):
            fp[i] = fun(theta+h[i], *args)
            fm[i] = fun(theta-h[i], *args)
            for j in range(i,K):
                fpp[i,j] = fun(theta + h[i] + h[j], *args)
                fpp[j,i] = fpp[i,j]
                fmm[i,j] = fun(theta - h[i] - h[j], *args)
                fmm[j,i] = fmm[i,j]

        for i in range(K):
            for j in range(i,K):
                H[i,j] = (fpp[i,j] - fp[i] - fp[j] + f
                + f - fm[i] - fm[j] + fmm[i,j])/hh[i,j]/2
                H[j,i] = H[i,j]
        return H

    ################## PROPERTY LIST ################ 
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, newData):
        if (type(newData) != pd.DataFrame):
            raise TypeError('Data must be in DataFrame!')
        self._data = newData
    
    @property
    def params(self):
        return self._params
    
    @property
    def ht(self):
        return self._mdlRes['ht']
    
    @property
    def et(self):
        return self._mdlRes['et']
    
    @property
    def Ey(self):
        return self._mdlRes['Ey']
    
    @property
    def stres(self):
        return self._mdlRes['stres']
    
    @property
    def mdlRes(self):
        return self._mdlRes
    
    @property
    def ICs(self):
        return [self._AIC, self._BIC, self._HQIC]