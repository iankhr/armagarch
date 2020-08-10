# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:22:50 2020

This class creates GJR - GARCH(P,O,Q) model

@author: Ian Khrashchevskyi
"""

from .volModel import VolModel
import pandas as pd
import numpy as np

class gjr(VolModel):
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
                            [(finfo.eps, 0.999999999) for i in range(self._order['o'])] +\
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


    def reconstruct(self, et, h0=None, params=None, other=None):
        """
        Scale up the innovations by variance
        """
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
        ht = np.ones(np.shape(et))*omega
        if len(np.shape(ht))>1:
            # we are applying for more than one et series our model
            dim = np.shape(ht)[1]
        else:
            dim = 1
            
        if h0 is not None:
            ht[0] = h0
        
        et[0] = et[0]*np.sqrt(ht[0])
        
        # create lag variables
        htlag = np.zeros((self._order['q'],dim))
        etlag = np.zeros((self._order['p'],dim))
        lagindht = np.arange(0,len(htlag))
        lagindht = np.roll(lagindht,1)
        lagindet = np.arange(0,len(etlag))
        lagindet = np.roll(lagindet,1)
        for t in range(len(et)):
            if t>0:
                etlago = etlag.copy()
                etlago[etlago>0] = 0
                #ht[t] = ht[t] + (etlag**2)@alpha + htlag@beta + (etlago**2)@gamma
                ht[t] = ht[t] + alpha@(etlag**2) + beta@htlag + gamma@(etlago**2)
                et[t] = et[t]*np.sqrt(ht[t])
            
            htlag = htlag[lagindht]
            htlag[0] = ht[t]
            etlag = etlag[lagindet]
            etlag[0] = et[t]
            
        # embedding non negativity contraints
        if np.any(ht<0):
            ht = 0
        
        if np.any(params<0):
            ht = 0
        
        return [et, ht]


    def predict(self, nsteps, params = None, data = None, other = None):
        """
        Makes the preditiction of the variance
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
        

        # making the prediction (Works similar as CondMean, but with a twist)
        prediction = np.ones((nsteps,))*omega
        # create lags of et and ht
        if (self._order['p']>0) | (self._order['o']>0):
            # create lags
            et = data[-np.max((self._order['p'], self._order['o'])):]
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
                if self._order['q']>0:
                    indicator = (laget[t:t+self._order['p']][::-1]<0)*1
                    prediction[t] = prediction[t] + lagetsq[t:t+self._order['o']][::-1]@gamma*indicator
                
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
    
    def expectedVariance(self, params):
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
        return omega/(1-np.sum(alpha) - 0.5*np.sum(gamma) - np.sum(beta))