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
        self._name = 'NoConstant'
        self._pnum = 0
        self._varnames = []
        self._setConstraints()

    def _getStartingVals(self):
        self._startingValues = []

    def _setConstraints(self):
        # redefing constraints with new data
        self._constraints = []

    # @profile
    def condMean(self, params=None, data=None, other=None):
        if data is None:
            data = self._data

        return data

    def reconstruct(self, et, params=None, other=None):
        """
        The function gets innovations and generates the process with specified
        model

        et must be DataFrame
        """
        if params is None:
            params = self._params
        return et

    # @profile
    def et(self, params=None, data=None, other=None):
        if data is None:
            data = self._data
        return data

    # @profile
    def func_constr(self, params):
        constr = []
        return constr