#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:44:19 2020

@author: wanxiangfan
"""

from scipy.optimize import fsolve
def f(x,y):
    return np.sin(x)-y

starting_guess = 3
print( fsolve(f, starting_guess) )


starting_guess = -3
print( fsolve(f, starting_guess, 4) )