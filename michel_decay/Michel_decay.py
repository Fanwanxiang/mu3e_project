#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:24:03 2019

@author: wanxiangfan
"""

import numpy as np
import matplotlib.pyplot as plt

'''============================================================================
   first use analytical method to get theta distribution'''
N = 500000

s = np.random.uniform(0,1,N) 
print(s)
''' generate uniform distribution sample, 
    s is a array which is the set of the data I generate in this project'''
theta = np.arccos(1-2*s)
#hist, bin_edges  = np.histogram(s, bins= 20)

np.all(theta >= 0)
True
np.all(theta <= np.pi)
True
count, bins, ignored = plt.hist(theta, bins= 'auto', normed=True) # draw the histagram of s_theta
'''count, bins, ignored = plt.hist(theta, bins= np.linspace(0,np.pi,20), normed=True)
bins is an array but can be set up as string like auto ''' 
#print(bins) 
plt.plot(bins, 0.5*np.sin(bins), linewidth=2, color='r')
plt.show()
'''============================================================================
second, generate phi as uniform distribution
then combine them together to get the splid angle distribution'''
phi = np.random.uniform(0,2*np.pi,N) 
omega = np.array([theta,phi]).T
print(np.array([theta,phi]))
print(omega)  # the solid angle is not need, since plt.hist2d is smart enough.

'''============================================================================
Third, Make a 2D histogram plot'''
plt.hist2d(theta, phi, bins=(40, 40), cmap=plt.cm.jet) # the graph is resonable 


