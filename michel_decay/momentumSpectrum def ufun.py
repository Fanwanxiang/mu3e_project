#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 19:32:37 2019

@author: wanxiangfan
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize    # to define a universal function for array

N = 10000                    # sample number
rho = 3/4                      # rho is constant and =3/4 in SM
momMax = 52.8                  # max monmentum of electrons
def g(y):
    return y*y
@vectorize                     # to define a universal function for array
def f(x):
    y = 4*x**2*(3*(1-x)+2*rho*(4/3*x-1))*g(3)
    return y 
def p(x):                      # rho = 3/4
    p = f(x)
    return p
 # sample number 
mom_e = np.random.uniform(0,momMax,N) # mon_e momentum of electron
x = mom_e/momMax
P_mom = p(x)/momMax# Pdf of electron
y = np.random.uniform(0,1,N)
mom_accept = mom_e[P_mom > y]
"""momentum_accept=[]
for i in range(N):
    if p(momentum[i]/P_emax)/P_emax > y[i]:
        momentum_accept.append(momentum[i])"""

count, bins, ignored = plt.hist(mom_accept, bins= 'auto', density=True) # draw the histagram of s_theta
#print(bins) 
plt.plot(bins, (6*(bins/52.8)**2-4*(bins/52.8)**3)/52.8, linewidth=2, color='r')
plt.show()

