# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 08:22:45 2020

This class fits multivariate ARMA-GARCH type models

@author: iankh
"""

import pandas as pd
import numpy as np
from .errors import InputError, HessianError
import datetime
import scipy
import shutil

class multiModel(object):
    def __init__(self, Data, Mean, Vol, Dist, startingVals = None, params = None):
        pass