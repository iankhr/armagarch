# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:03:12 2020

This files contains some list of custom errors

@author: Ian Khrashchevskyi
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
