# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:49:01 2020

This is the main class that ties up Mean, Volatility and Distribution classes
and performs maximum likelihood estimation

@author: Ian Khrashchevskyi
"""

import pandas as pd
import numpy as np
from .errors import InputError, HessianError
import datetime
import scipy
import shutil

class empModel(object):
    def __init__(self, Data, Mean, Vol, Dist, startingVals = None, params = None):
        # Data
        self._data = Data           
        # Models as defined in classes above
        self._mean = Mean
        self._vol = Vol
        # gives ll and also adds extra variables to estimate
        self._dist = Dist
        self._startingVals = startingVals
        self._params = params
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
        stres = pd.DataFrame(stres, index = et.index, columns = et.columns)
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
        """
        The function fits in the model
        """
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
    
    
    def predict(self, nsteps=1, params = None, data = None, other = None):
        """
        Makes prediction based on the fitted model
        """
        if nsteps<=0:
            raise ValueError('Number of steps must be a positive number!')
        
        if params is None:
            params = self._params
            
        if data is None:
            data = self._data
        
        try:
            otherOld = {}
            otherOld['regdata']= other['olddata']
        except:
            otherOld = None
        
        pars = self._parUnpack(params)
        # apply model first
        mdlres = self.apply(data = data, params = params, other = otherOld)
        # get prediction of the mean
        mPred = self._mean.predict(nsteps, params = pars[0], data = data, other = other)
        # get prediction for volatility
        vPred = self._vol.predict(nsteps, params = pars[1], data = mdlres['et'], other = other)
        return [mPred, vPred]

    
    def simulate(self, nobs=1, params = None, h0=None, other=None):
        """
        Makes simulations. 
            nobs- number of obsevations
            h0 - starting variance. if not specified average e^2 is used
            other - dictionary with regressors and other model information
        
        OUTPUT:
            List with 
                data - simulated data
                Errs - list with errors and ht
                stErrs - list with standardized errors
        """
        if params is None:
            params = self._params
            if params is None:
                raise ValueError('Parameters of the model must be specified!')
        
        if nobs<=0:
            raise ValueError('Number of observations must be positive number!')
        
        if h0 is None:
            h0 = np.mean(self.et.values**2)
        
        pars = self._parUnpack(params)
        # simulate random variables
        stErrs = self._dist.simulate(nobs)
        # scale them up by variance
        Errs = self._vol.reconstruct(stErrs, h0=h0, params = pars[1], other = other)
        # add mean
        data = self._mean.reconstruct(Errs[0], params = pars[0], other = other)       
        return [data, Errs, stErrs]

    
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
        # degrees of freedom
        self._mdl_df = k - 1      
        # AIC
        self._AIC = 2*k - 2*L
        if n/k<40:
            self._AIC = self._AIC + 2*k*(k+1)/(n-k-1)
        
        self._BIC = np.log(n)*k - 2*L
        self._HQIC = -2*L + (2*k*np.log(np.log(n)))
        self._SIC = -2*L + np.log(n+2*k)
        # estimate adjusted R^2
        # estimate adjusted R^2
        Ey = self._mdlRes['Ey']
        et = self._mdlRes['et']
        Y = Ey + et
        # estimate r-rquared and adjusted r-squared
        SSR = np.sum(et**2)
        SST = np.sum((Y - np.mean(Y))**2)
        self._rsq = 1 - SSR/SST
        try:
            self._rsq = self._rsq.values[0]
        except:
            pass
        self._adjrsq =1-(1-self._rsq)*(len(self._data)-1)/(len(self._data)\
                                                           -self._mdl_df-1)
    
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
        sts.append(['Df Residuals', str(len(data)-self._mdl_df)])
        sts.append(['Df Model', str(self._mdl_df)])
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
    
    @property
    def expectedVariance(self):
        if self._params is None:
            raise InputError('Parameters are not specified')
        else:
            pars = self._parUnpack(self._params)
            return self._vol.expectedVariance(pars[1])