#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:39:07 2019

@author: wanxiangfan
"""
import numpy as np
import matplotlib.pyplot as plt
N = 10000000 # sample number
theta = np.random.uniform(0,np.pi,N) 
y = np.random.uniform(0,1,N)
P_theta = np.sin(theta) #probabilty density P(theta)
Theta_accept = theta[P_theta > y] 
#count, bins, ignored = plt.hist(Theta_accept, bins= 'auto', normed=True) # draw the histagram of s_theta
#print(bins) 
#plt.plot(bins, 0.5*np.sin(bins), linewidth=2, color='r')
#plt.show()
N_phi = len(Theta_accept)
phi = np.random.uniform(0,2*np.pi,N_phi) 

plt.hist2d(Theta_accept, phi, bins=(100, 100), cmap=plt.cm.jet) # the graph is resonable 
plt.colorbar()