# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:21:20 2020

This class creates GARCH(P,Q) model

@author: Ian Khrashchevskyi
"""

from .volModel import VolModel
import pandas as pd
import numpy as np

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
 
    
    def predict(self, nsteps, params = None, data = None, other = None):
        """
        Makes prediction of variance
        """
        if params is None:
            params = self._params
            
        if data is None:
            data = self._data
            
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
        # create lags of et and ht
        if self._order['p']>0:
            # create lags
            et = data[-self._order['p']:]
            laget = np.zeros((nsteps+len(et),1))
            laget[:len(et)] = et.values
            lagetsq = laget**2
        
        if self._order['q']>0:
            # create lags of ht
            httemp = ht[-self._order['q']:]
            laght = np.zeros((nsteps+len(httemp),1))
            laght[:len(httemp)] = httemp.values
        
        # make predictions
        for t in range(nsteps):
            if self._order['q']>0:
                prediction[t] = prediction[t] + laght[t:t+self._order['q']][::-1]@beta
            
            if self._order['p']>0:
                prediction[t] = prediction[t] + lagetsq[t:t+self._order['p']][::-1]@alpha                
                lagetsq[t+1] = prediction[t]
            
            if self._order['q']>0:
                laght[t+1] = prediction[t]
                
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
    
    
    def expectedVariance(self, params):
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
        return omega/(1-np.sum(alpha) - np.sum(beta))
    
    
    def reconstruct(self, et, h0=None, params=None, other=None):
        """
        Scale up the innovations by variance
        """
        #start_time = time.perf_counter()
        if params is None:
            params = self._params
            
        if type(et) == pd.DataFrame:
            et = et.values

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
        ht = np.ones(np.shape(et))*omega
        if len(np.shape(ht))>1:
            # we are applying for more than one et series our model
            dim = np.shape(ht)[1]
        else:
            dim = 1
        
        if h0 is not None:
            ht[0] = h0
                
        et[0] = et[0]*np.sqrt(ht[0])
        #end_time = time.perf_counter()
        #print('Benchmark 1 {} seconds'.format(end_time-start_time))
        # create lag variables
        htlag = np.zeros((self._order['q'],dim))
        etlag = np.zeros((self._order['p'],dim))
        lagindht = np.arange(0,len(htlag))
        lagindht = np.roll(lagindht,1)
        lagindet = np.arange(0,len(etlag))
        lagindet = np.roll(lagindet,1)
        #end_time = time.perf_counter()
        #print('Benchmark 2 {} seconds'.format(end_time-start_time))
        for t in range(len(et)):
            if t>0:
                #ht[t] = ht[t] + (etlag**2)@alpha + htlag@beta
                ht[t] = ht[t] + alpha@(etlag**2) + beta@htlag
                et[t] = et[t]*np.sqrt(ht[t])
            
            htlag = htlag[lagindht]
            htlag[0] = ht[t]
            etlag = etlag[lagindet]
            etlag[0] = et[t]
        
        #end_time = time.perf_counter()
        #print('Benchmark 3 {} seconds'.format(end_time-start_time))
        # embedding non negativity contraints
        if np.any(ht<0):
            ht = 0
        
        if np.any(params<0):
            ht = 0
        
        return [et, ht]