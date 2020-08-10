# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 20:08:14 2018

@author: Ian
"""

from .basicFun import hessian_2sided
from .basicFun import getLag
import numpy as np
import scipy
import shutil
import datetime
from scipy.optimize import brute
from scipy.special import gamma
from statsmodels.tsa.arima_model import ARMA

class legacygarch(object):
    def __init__(self, data, PQ = (0,0), poq = (1,0,1), startingVals = None, \
                 constant = True, debug = False, printRes = True, fast = False,\
                 extraRegressors = None, dist = 'Normal'):
        self._data = data
        self._extraRegressors = extraRegressors
        if self._extraRegressors is not None:
            try:
                self._numregs = np.shape(extraRegressors)[1]
            except:
                self._numregs = 1
        else:
            self._numregs = 0
        
        self._poq = poq
        self._PQ = PQ
        if self._poq[1] == 0:
            self._gtype = 'GARCH'
        else:
            self._gtype = 'GJR'
            
        self._startVals = startingVals
        self._dist = dist
        self._method = 'MLE'
        self._model = constant
        self._debug = debug
        self._printRes = printRes
        self._fast = fast

    #@profile
    def _meanProcess(self, yt, parameters, PQ = (1,0)):
        et = np.zeros((len(yt),))
        c = parameters[0]
        lags = getLag(yt,PQ[0])
        if (PQ[0]>0) or (PQ[1]>0):
            phi = parameters[1:PQ[0]+1]
            theta = parameters[PQ[0]+1:PQ[0]+PQ[1]+1]
        
        if self._numregs>0:
            beta = parameters[PQ[0]+PQ[1]+1:]
            
        for i in range(len(yt)):
            ey = c
            if i>0:
                if PQ[0] > 0:
                    ytLag = lags[i]
                    ey = ey +ytLag@phi
                if PQ[1]>0:
                    etLag = np.array(et[i-PQ[1]:i])[::-1]
                    ey = ey + etLag@theta[:PQ[1]]
                           
            et[i]=float(ey)
        
        et = yt - et
        if self._extraRegressors is not None:
            if len(beta)>1:
                et = et - self._extraRegressors.values@beta
            else:
                et = et - self._extraRegressors*beta
                
            
        return np.asarray(et)
    
    #@profile
    def _normLik(self, data, ht, out=False):
        """
        Likelihood function for Normal distribution
        """
        if np.any(ht<=0):
            nlogLik = np.Inf
        else:
            lls = np.log(2*np.pi) + np.log(ht) + (data)**2/ht
            nlogLik = 0.5*np.sum(lls)
    
        if np.isnan(nlogLik):
            nlogLik = np.Inf
            
        if out == False:
            return nlogLik
        else:
            return nlogLik, lls

        
    def _tLik(self, data, ht, nu, out=False):
        """
        Likelihood function for t-Student distribution
        """
        if np.any(ht<=0) or nu<=2:
            nlogLik = np.Inf
        else:             
            lls = np.log(gamma((nu+1)/2)/(np.sqrt(np.pi*(nu-2))*gamma(nu/2)))\
                - 0.5*np.log(ht) - (nu+1)/2*np.log(1+data**2/(ht*(nu-2)))           
            nlogLik = np.sum(-lls)
        
        if np.isnan(nlogLik):
            nlogLik = np.Inf
        
        if out == False:
            return nlogLik
        else:
            return nlogLik, lls

    
    def _skewtLik(self, data, ht, nu, l, out=False):
        """
        Likelihood function for skew-t Distribution
        """
        if np.any(ht<=0) or nu<=2 or np.abs(l)>=1:
            nlogLik = np.Inf
        else:
            a = gamma((nu+1)/2)*np.sqrt(nu-2)*(l-1/l)
            a = a/(np.sqrt(np.pi)*gamma(nu/2))
            b = np.sqrt((l**2+1/l**2-1)-a**2)
            tVar = -a/b
            IndicF = -1*(data/np.sqrt(ht)<tVar)
            IndicF = 2*IndicF+1
            IndicF = 2*IndicF
            lls = np.log(gamma((nu+1)/2)/(np.sqrt(np.pi*(nu-2))*gamma(nu/2)))\
                + np.log(b) + np.log(2/(l+1/l))\
                - 0.5*np.log(ht)\
                - 0.5*(nu+1)*np.log(1+(b*data/np.sqrt(ht)+a)**2/(nu-2)*l**IndicF)  
            nlogLik = np.sum(-lls)
        
        if np.isnan(nlogLik):
            nlogLik = np.Inf
        
        if out == False:
            return nlogLik
        else:
            return nlogLik, lls
        

    #@profile         
    def _garchParUnwrap(self, parameters, poq):
        omega = parameters[0]
        pointer = 1
        # getting alphas
        if poq[0] == 0:
            alpha = 0
        else:
            pointer = pointer+poq[0]
            alpha = parameters[1:pointer]
                
        # getting gammas
        if poq[1] == 0:
            gamma = 0
        else:
            gamma = parameters[pointer: pointer+poq[1]]
            pointer = pointer+poq[1]
        
        # getting betas
        if poq[2] == 0:
            beta = 0
        else:
            beta = parameters[pointer:pointer+poq[2]]
            
        return omega, alpha, gamma, beta

    #@profile    
    def _garchht(self, parameters, et, gtype = 'GARCH', poq = (1,0,1)):
        # it is common to set  ho as unconditional varince
        ht = []
        #print(parameters)
        omega, alpha, gamma, beta = self._garchParUnwrap(parameters, poq)
        h0 = np.mean(et**2)
        ht.append(h0)
        for i in np.arange(1,len(et),1):
                etLag = np.array(et[i-poq[0]:i])[::-1]
                htLag = np.array(ht[i-poq[2]:i])[::-1]
                if gtype == 'GARCH':
                    tempRes = omega
                    try:
                        tempRes += alpha[:len(etLag)]@(etLag**2)
                    except:
                        pass
                    
                    try:
                        tempRes += beta[:len(htLag)]@htLag
                    except:
                        pass
                    ht.append(tempRes)
                elif gtype=='GJR':
                    etLagGamma = np.array(et[i-poq[1]:i])[::-1]
                    indFun = 1*(etLagGamma<0)
                    ht.append(omega+np.sum(alpha*etLag**2)
                    +np.sum(gamma*indFun*etLagGamma**2)
                    +np.sum(beta*np.asarray(htLag)))
        
        # embeding the inequality for stability reasons
        if (np.any(np.asarray(parameters)<0)):
            ht = 0
        
        if (np.any(np.isnan(parameters))):
            ht = 0
        
        if (1-np.sum(alpha)-np.sum(beta)-0.5*np.sum(gamma))<0:
            ht = 0
           
        return np.asarray(ht)

    #@profile   
    def _garchll(self, parameters, data, gtype, poq, out=False, brute = False):
        if (self._model == False) or (brute==True):
            et = data - np.mean(data)
            if self._dist =='Student':
                if brute == True:
                    nu=3
                else:
                    nu = parameters[-1]
            elif self._dist == 'skewt':
                if brute == True:
                    nu=3
                    l = 0.5
                else:
                    nu = parameters[-2]
                    l = parameters[-1]
                    
            ht = self._garchht(parameters[:-1],et, gtype, poq)
        else:
            Mparams = parameters[:1+np.sum(self._PQ)+self._numregs]
            if self._dist == 'Student':
                Gparams = parameters[1+np.sum(self._PQ)+self._numregs:-1]
                nu = parameters[-1]
            elif self._dist == 'skewt':
                nu = parameters[-2]
                l = parameters[-1]
                Gparams = parameters[1+np.sum(self._PQ)+self._numregs:-2]
            else:
                Gparams = parameters[1+np.sum(self._PQ)+self._numregs:]
                
            et = self._meanProcess(data, Mparams, self._PQ)
            ht = self._garchht(Gparams, et, gtype, poq)
         
        if out == False:
            if self._dist=='Normal':
                nlogLik = self._normLik(et, ht, out)
            elif self._dist == 'Student':
                nlogLik = self._tLik(et, ht, nu, out)
            elif self._dist == 'skewt':
                nlogLik = self._skewtLik(et, ht, nu, l, out)
            else:
                raise TypeError('The distribution nopt implemented')
            if self._debug == True:
                print(parameters, nlogLik)
                
            return nlogLik
        else:
            nlogLik, lls = self._normLik(et, ht, out)
            return nlogLik, lls

    #@profile        
    def _garchConst(self, parameters, data, gtype, poq, out=None):
        if self._model == False:
            if self._dist == 'Student':
                Gparams = parameters[:-1]
                nu = parameters[-1]
            elif self._dist == 'skewt':
                Gparams = parameters[:-2]
                nu = parameters[-2]
                l = parameters[-1]
            else:
                Gparams = parameters
                
            omega, alpha, gamma, beta = self._garchParUnwrap(Gparams, poq)
            return np.array([0.999-np.sum(alpha)-np.sum(beta)-0.5*np.sum(gamma)])
        else:
            Mparams = parameters[:1+np.sum(self._PQ)]
            if self._dist == 'Student':
                Gparams = parameters[1+np.sum(self._PQ)+self._numregs:-1]
                nu = parameters[-1]
            elif self._dist == 'skewt':
                Gparams = parameters[1+np.sum(self._PQ)+self._numregs:-2]
                nu = parameters[-2]
                l = parameters[-1]
            else:
                Gparams = parameters[1+np.sum(self._PQ)+self._numregs:]
            omega, alpha, gamma, beta = self._garchParUnwrap(Gparams, poq)
            const = np.array([0.999-np.sum(alpha)-np.sum(beta)-0.5*np.sum(gamma)])
            if np.sum(self._PQ)>0:
                if self._PQ[0]>0:
                    ars = -Mparams[0:self._PQ[0]+1]
                    ars[0] = 1
                    rootsAR = 1-np.abs(np.roots(ars))
                    np.append(const, np.all(rootsAR>0)*2-1)
                if self._PQ[1]>0:
                    mas = -Mparams[self._PQ[0]:]
                    mas[0] = 1
                    rootsMA = 1-np.abs(np.roots(mas))
                    np.append(const,np.all(rootsMA>0)*2-1)
                
            return const

    
    def _valsConstructor(self,z,*params):
        data, gtype, poq = params
        return self._garchll(z, data, gtype, poq, brute= True)
    
    #@profile
    def _getStartVals(self):
        if self._startVals is None:
            # perform a grid search
            if self._model == True:
                if (self._PQ[0]>0) | (self._PQ[1]>0):
                    # fitting ARMA model
                    tMdl = ARMA(self._data,self._PQ).fit()
                    startingVals = list(tMdl.params.values)
                    et = tMdl.resid.values
                else:       
                    startingVals=[np.mean(self._data)]
                    et = self._data-startingVals[0]
                if self._numregs>0:
                    startingVals = startingVals + list(np.zeros((self._numregs,)))
            else:
                et = self._data
                startingVals = []
                
            # getting the starting vals for garch
            params = (et, 'GARCH', (self._poq[0], 0, self._poq[2]))
            # bound for omega
            rranges = [slice(0.001, 0.1, 0.1)]
            # range for p 
            for i in range(self._poq[0]):
                rranges.append(slice(0.1, 0.15, 0.01))
            # range for q
            for i in range(self._poq[2]):
                rranges.append(slice(0.6, 0.8, 0.1))
            
            gridRes = brute(self._valsConstructor, tuple(rranges), \
                            args = params, full_output=True, finish=None)
            vals= gridRes[0]
            if self._poq[1] == 0:
                startingVals = startingVals+list(vals)
            else:
                startingVals = startingVals+list(vals[:1+self._poq[0]])\
                             + list(np.zeros((self._poq[1],)))\
                             + list(vals[1+self._poq[0]:])
            
            if self._dist == 'Student':
                startingVals = startingVals+[3,]
            elif self._dist == 'skewt':
                startingVals = startingVals+[3,0.5]
        else:
            startingVals = self._startVals
        
        return startingVals

    #@profile
    def fit(self):
        finfo = np.finfo(np.float64)
        # getting starting values
        startingVals = self._getStartVals()
        # creating bounds for an optimizer
        bounds = []
        if self._model == True:
            bounds.append((-10*np.abs(np.mean(self._data)), 10*np.abs(np.mean(self._data))))
            if np.sum(self._PQ)>0:
                for i in range(np.sum(self._PQ)):
                    bounds.append((-0.999, 0.999))
            
            if self._numregs>0:
                for i in range(self._numregs):
                    bounds.append((-np.inf, np.inf))
        
        # GARCH bounds
        bounds.append((finfo.eps, 2*np.var(self._data)))#omega
        for i in range(np.sum(self._poq)):
            bounds.append((finfo.eps,0.9999))
        
        # Distribution bounds
        if self._dist == 'Student':
            # above 12 degrees of freedom it converges to Normal
            bounds.append((3,np.inf))
        elif self._dist == 'skewt':
            bounds.append((3,np.inf))
            bounds.append((-0.999,0.999))
        
        args = (np.asarray(self._data), self._gtype, self._poq)    
        
        if self._printRes ==  True:
            optimOutput = 1
        else:
            optimOutput = 0
            
        # estimating standard GARCH model
        #print(startingVals)
        self._estimates = scipy.optimize.fmin_slsqp(self._garchll, startingVals, 
                                            args = args, 
                                            f_ieqcons = self._garchConst,
                                            bounds = bounds,
                                            epsilon=1e-6, acc = 1e-7, iter=100,\
                                            iprint = optimOutput)
        # Once the model is estimated I can get the standard errors 
        if self._fast == False:
            self._vcv = self._getvcv()
            self._log_likelihood = -self._garchll(self._estimates,\
                                                             self._data,\
                                                             self._gtype,\
                                                             self._poq)
            self._getICs()
        
        self._rsquared()
        self._stres = self._et/np.sqrt(self._ht)
        if self._printRes == True:
            self._printResults()

    #@profile           
    def _rsquared(self):
        if self._model == True:
            Mparams = self._estimates[:1+np.sum(self._PQ)+self._numregs]
            Gparams = self._estimates[1+np.sum(self._PQ)+self._numregs:]
            omega, alpha, gamma, beta = self._garchParUnwrap(Gparams, self._poq)
            self._uncondVar = omega/(1-np.sum(alpha)-np.sum(beta)-0.5*np.sum(gamma))
            et = self._meanProcess(self._data, Mparams, self._PQ)
            self._ht = self._garchht(Gparams, et, self._gtype, self._poq)
            self._et = et
            Ey = self._data - et
            self._rsq = np.sum((Ey-np.mean(self._data))**2)/np.sum((self._data-np.mean(self._data))**2)
            self._adjrsq =1-(1-self._rsq)*(len(self._data)-1)/(len(self._data)-np.sum(self._PQ)-1)
        else:
            self._et = self.data
            self._ht = self._garchht(self._estimates,self._et, self._gtype, self._poq)
            self._rsq = np.nan
            self._adjrsq = np.nan

    #@profile
    def _normLf(self):
        l = 1/np.sqrt(2*np.pi*self._ht)*np.exp(-(self._data
                     -np.mean(self._data))**2/(2*self._ht))
        return np.prod(l)

    #@profile
    def _getICs(self):
        k = len(self._estimates)
        n = len(self._data)
        L = self._log_likelihood
        self._AIC = 2*k - 2*L
        if n/k<40:
            self._AIC = self._AIC + 2*k*(k+1)/(n-k-1)
        
        self._BIC = np.log(n)*k - 2*L
        self._HQIC = -2*L + (2*k*np.log(np.log(n)))
        self._SIC = -2*L + np.log(n+2*k)

    #@profile
    def _getvcv(self):
        parameters =self._estimates
        data = self._data
        gtype = self._gtype
        poq = self._poq       
        T = len(data)
        step = 1e-5 * parameters
        scores = np.zeros((T,len(parameters)))
        for i in range(len(parameters)):
            h = step[i]
            delta = np.zeros(len(parameters))
            delta[i] = h
            
            loglik, logliksplus = self._garchll(parameters + delta, \
            data, gtype, poq, out=True)
            
            loglik, logliksminus = self._garchll(parameters - delta, \
            data, gtype, poq, out=True)
            scores[:,i] = (logliksplus - logliksminus)/(2*h)
            
        I = (scores.T @ scores)/T
        args = (data, gtype, poq)
        J = hessian_2sided(self._garchll, parameters, args)
        J = J/T
        try:
            Jinv = np.mat(np.linalg.inv(J))
            vcv = Jinv*np.mat(I)*Jinv/T
            vcv = np.asarray(vcv)
        except:
            print('WARNING: Hessian is singular! St.errors are not calcualted')
            vcv = np.eye(len(parameters))

        return vcv
    
    #@profile
    def _printResults(self):
        params = self._estimates
        vcv = self._vcv
        gtype = self._gtype
        output = np.vstack((params,np.sqrt(np.diag(vcv)),params/np.sqrt(np.diag(vcv)))).T
        if self._model== True:
            meanParams = output[:1+np.sum(self._PQ)+self._numregs,:]
            if self._dist == 'Student':
                goutput = output[1+np.sum(self._PQ)+self._numregs:-1,:]
            elif self._dist == 'skewt':
                goutput = output[1+np.sum(self._PQ)+self._numregs:-2,:]
            else:
                goutput = output[1+np.sum(self._PQ)+self._numregs:,:]
        else:
            goutput = output
                
        tab = 4
        columns = shutil.get_terminal_size().columns
        # printing the upper body
        title = gtype+ " estimation results"
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
        if self._model == True:
            print('Mean Model'.center(columns))
            print('='*columns)
            if self._numregs==0:             
                self._tableOutput(meanParams, ['Constant','AR','MA'], 
                                  (1, self._PQ[0], self._PQ[1]), tab, ocl)
            else:
                self._tableOutput(meanParams, ['Constant','AR','MA','reg'], 
                                  (1, self._PQ[0], self._PQ[1], self._numregs),\
                                  tab, ocl)
                    
        # print the results of mean model here
        
        # printing the volatility model
        print('Volatility Model'.center(columns))
        print('='*columns)
        self._tableOutput(goutput, ['omega','alpha','gamma','beta'], 
                          (1, self._poq[0],self._poq[1],self._poq[2]), tab, ocl)
                
        if self._dist == 'Student':
            print('Distribution: Student'.center(columns))
            print('='*columns)
            self._tableOutput(np.atleast_2d(output[-1,:]), ['nu', ], (1, ), tab, ocl)
        elif self._dist == 'skewt':
            print('Distribution: Student'.center(columns))
            print('='*columns)
            self._tableOutput(output[-2:,:], ['nu','lambda'], (1,1), tab, ocl)
        
        print('='*columns)
        print('Covariance estimator: robust')

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
                
                tabLenName = ' '*int(ocl-len(rowName))
                # putting the values
                est = str(item[0])
                se = str(item[1])
                tstat = str(item[2])
                print(rowName+tabLenName+tab*' '
                      +' '*int(ocl-len(est)) + est+ tab*' '
                      +' '*int(ocl-len(se)) + se+tab*' '
                      +' '*int(ocl-len(tstat)) + tstat)
        else:
            tabLenName = ' '*int(ocl-len(rowName))
            # putting the values
            est = str(output[0])
            se = str(output[1])
            tstat = str(output[2])
            print(rowName+tabLenName+tab*' '
                  +' '*int(ocl-len(est)) + est+ tab*' '
                  +' '*int(ocl-len(se)) + se+tab*' '
                  +' '*int(ocl-len(tstat)) + tstat)

    #@profile    
    def _smallStats(self):
        data = self._data
        gtype = self._gtype
        ll = self._log_likelihood
        sts = []
        if self._model == False:
            model = 'Zero-Constant'
        elif np.sum(self._PQ) == 0:
            model = 'Constant'
        else:
            model = ''
            if self._PQ[0] > 0:
                model = model + 'AR'
            if self._PQ[1] > 0:
                model = model + 'MA'
                            
        sts.append(['Dep Variable', 'y'])
        sts.append(['Mean Model', model])
        sts.append(['Vol Model', gtype])
        sts.append(['Distribution',self._dist])
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
        sts.append(['Df Residuals', str(len(data)-len(self._estimates))])
        sts.append(['Df Model', str(len(self._estimates))])
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

        
    def summary(self):
        self._printResults()
        
    
    def applyModel(self, newData, reconstruct = False, y0 = 0, h0=1):
        if reconstruct == False:
            if self._model == True:
                Mparams = self._estimates[:1+np.sum(self._PQ)]
                Gparams = self._estimates[1+np.sum(self._PQ):]
                et = self._meanProcess(newData, Mparams, PQ = (1,0))
                ht = self._garchht(Gparams, et, gtype = self._gtype, poq = self._poq)
            else:
                et = newData
                ht = self._garchht(self._estimates, et, self._gtype, self._poq)
            return [et, ht]
        else:
            # it means that we got innovations and need to return yts and hts
            if self._model == True:
                Mparams = self._estimates[:1+np.sum(self._PQ)]
                Gparams = self._estimates[1+np.sum(self._PQ):]
                # generate ht from the model
                ht = np.zeros(np.size(newData))
                #print(parameters)
                omega, alpha, gamma, beta = self._garchParUnwrap(Gparams, self._poq)
                ht[0] = h0
                et = newData
                et[0] *= np.sqrt(ht[0])
                # generating variance        
                for i in np.arange(1,len(et),1):
                    etLag = np.array(et[i-self._poq[0]:i])[::-1]
                    htLag = np.array(ht[i-self._poq[2]:i])[::-1]
                    if self._gtype == 'GARCH':
                        tempRes = omega
                        try:
                            tempRes += alpha[:len(etLag)]@(etLag**2)
                        except:
                            pass
                        
                        try:
                            tempRes += beta[:len(htLag)]@htLag
                        except:
                            pass                    
                    elif self._gtype=='GJR':
                        etLagGamma = np.array(et[i-self._poq[1]:i])[::-1]
                        indFun = 1*(etLagGamma<0)
                        tempRes = omega
                        try:
                            tempRes += alpha[:len(etLag)]@(etLag**2)
                        except:
                            pass
                        
                        try:
                            tempRes += beta[:len(htLag)]@htLag
                        except:
                            pass
                        
                        try:
                            tempRes += indFun*(gamma[:len(etLagGamma)]@etLagGamma)
                        except:
                            pass
                    
                    if tempRes>=0:
                        ht[i] = tempRes
                    else:
                        ht[i] = h0
                    
                    et[i] *= np.sqrt(ht[i])
                
                # reconstructing ys
                eY = np.zeros(np.size(ht))
                # add constant
                eY = eY+Mparams[0]
                if (self._PQ[0]>0) or (self._PQ[1]>0):
                    phi = Mparams[1:self._PQ[0]+1]
                    theta = Mparams[self._PQ[0]+1:]
                # check if MA is there
                if self._PQ[1]>0:
                    etLag = getLag(et, self._PQ[1])
                    maComp = etLag*theta[:self._PQ[1]]
                    eY = eY + maComp
                # check if ARMA is there
                if self._PQ[0]>0:
                    for i in range(len(eY)):
                        if i == 0:
                            if type(phi) == float:
                                eY[0] = eY[0]+phi*y0
                            else:
                                eY[0] = eY[0]+phi[0]*y0
                        else:
                            ytLag = np.array(eY[i-self._PQ[0]:i])[::-1]
                            eY[i] = eY[i] + ytLag@phi
                
                return [eY+et, ht]
            else:
                """
                This part is WRONG! You need to generate ht first and then
                multiply et by sqrt(ht) the result will be yt
                """
                yt = newData
                ht = self._garchht(self._estimates, yt, self._gtype, self._poq)
            return [yt, ht]

    
    def predict(self, step = 1):
        """
        The function does one-step ahead forecast. Simulation is not stable
        This part needs some work to be done on
        """
        poq = self._poq
        PQ = self._PQ
        htF = []
        ey = []
        if self._model == False:
            Gparams = self._estimates
            omega, alpha, gamma, beta = self._garchParUnwrap(Gparams, poq)
        else:
            Mparams = self._estimates[:1+np.sum(self._PQ)]
            Gparams = self._estimates[1+np.sum(self._PQ):]
            omega, alpha, gamma, beta = self._garchParUnwrap(Gparams, poq)
        
        if step >1:
            """
            Here I use simulations to infer the values
            """
            if self._dist == 'Normal':
                ets = np.random.normal(0, 1, size=(1000+step-1, 1000))
                ets = ets[1000:,:]
        
        for i in range(step):
            if i == 0:
                etLag = np.array(self._et[-poq[0]:])[::-1]
                htLag = np.array(self._ht[-poq[2]:])[::-1] 
                if poq[1]>0:
                    etLagGamma = np.array(self._et[-poq[1]:])[::-1]
                    indFun = 1*(etLagGamma<0)
                else:
                    etLagGamma = 0
                    indFun = 0

                htFtemp = omega + np.sum(alpha*etLag**2)+ np.sum(beta*htLag)\
                + np.sum(gamma*indFun*etLagGamma**2)
                
                htF.append(htFtemp)
                if self._model == False:
                    ey.append(0)
                else:
                    eyt=Mparams[0]
                    if PQ[0] > 0:
                        phi = Mparams[1:PQ[0]+1]
                        ytLag = np.array(self._data[-PQ[0]:])[::-1]
                        eyt = eyt +np.sum(ytLag*phi)
                    if PQ[1]>0:
                        theta = Mparams[1+PQ[0]:]
                        etLag = np.array(self._et[-PQ[1]:])[::-1]
                        eyt = eyt + np.sum(etLag*theta)
                        
                    ey.append(eyt)
            else:
                htFtemp = []
                for j in range(1000):
                    # previous volatility
                    htLag = np.roll(htLag,1)
                    htLag[0] = htF[-1]
                    # alpha component
                    etLag = np.roll(etLag,1)
                    etLag[0] = ets[i-1,j]
                    etInov = np.sum(alpha*etLag**2)
                    # gamma component
                    if poq[1]>0:
                        etLagGamma = np.roll(etLagGamma)
                        etLagGamma[0] = ets[i-1,j]
                        indFun = 1*(etLagGamma<0)
                        gammaComp = np.sum(indFun*etLagGamma*gamma)
                    else:
                        gammaComp = 0
                    
                    htFtemp.append(omega + np.sum(beta*htLag)+ etInov + gammaComp)
                    if self._model == False:
                        eyt = 0
                    else:
                        eyt=Mparams[0]
                        
                    if PQ[0] > 0:
                        phi = Mparams[1:PQ[0]+1]
                        ytLag = np.roll(ytLag,1)
                        ytLag[0] = ey[-1]
                        eyt = eyt +np.sum(ytLag*phi)
                    if PQ[1]>0:
                        theta = Mparams[1+PQ[0]:]
                        eyt = eyt + np.sum(etLag*theta)   
                
                ey.append(np.sum(eyt)/1000)
                htF.append(np.sum(htFtemp)/1000)
            
        return np.array(ey), np.array(htF)

    
    @property
    def ht(self):
        return self._ht
    
    
    @property
    def params(self):
        return self._estimates
    
    
    @property
    def vcv(self):
        return self._vcv
    
    
    @property
    def data(self):
        return self._data
    
    
    @property
    def AIC(self):
        return self._AIC
    
    
    @property
    def BIC(self):
        return self._BIC
    
    
    @property
    def HQIC(self):
        return self._HQIC
    
    
    @property
    def SIC(self):
        return self._SIC    
   
    @property
    def ll(self):
        return self._log_likelihood
    
    
    @property
    def et(self):
        return self._et
    
    
    @property
    def stres(self):
        return self._stres
    
    
    @property
    def uncvar(self):
        return self._uncondVar