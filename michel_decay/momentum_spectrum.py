#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:13:47 2019

@author: wanxiangfan
"""

import numpy as np
import matplotlib.pyplot as plt
N = 10**7 # sample number

rho = 3/4 # rho is constant and =3/4 in SM
momMax = 52.8 # max monmentum of electrons
def p(xp): # rho = 3/4
    p=4*xp**2*(3*(1-xp)+2*rho*(4/3*xp-1))
    return p
 # sample number 
mom_e = np.random.uniform(0,52.8,N) # mon_e momentum of electron
x = mom_e/momMax
P_mom = 4*x**2*(3*(1-x)+2*rho*(4/3*x-1))/momMax# Pdf of electron
y = np.random.uniform(0,1,N)
mom_accept = mom_e[P_mom > y]
np.savetxt("michel_decay_positron4momentum.txt", mom_accept)
"""momentum_accept=[]
for i in range(N):
    if p(momentum[i]/P_emax)/P_emax > y[i]:
        momentum_accept.append(momentum[i])"""

count, bins, ignored = plt.hist(mom_accept, bins= 'auto', density=True) # draw the histagram of s_theta
#print(bins) 
plt.plot(bins, (6*(bins/52.8)**2-4*(bins/52.8)**3)/52.8, linewidth=2, color='r')
plt.show()


'''============================================================================
oringinal code without universal function, which is slower than the current code
theta = np.random.uniform(0,np.pi,N) 
y = np.random.uniform(0,1,N)

Theta_accept=[]
for i in range(N):
    if np.sin(theta[i]) > y[i]:
        Theta_accept.append(theta[i])

count, bins, ignored = plt.hist(Theta_accept, bins= 'auto', normed=True) # draw the histagram of s_theta
#print(bins) 
plt.plot(bins, 0.5*np.sin(bins), linewidth=2, color='r')
plt.show()'''
#==============================================================================
'''rho = 3/4
def p(xp): # rho = 3/4
    p=4*xp**2*(3*(1-xp)+2*rho*(4/3*xp-1))
    return p
 # sample number
x = np.random.uniform(0,1,N) # x= p/p_max
y = np.random.uniform(0,1,N)

x_accept=[]
for i in range(N):
    if p(x[i]) > y[i]:
        x_accept.append(x[i])

count, bins, ignored = plt.hist(x_accept, bins= 'auto', normed=True) # draw the histagram of s_theta
#print(bins) 
plt.plot(bins, 6*bins**2-4*bins**3, linewidth=2, color='r')
plt.show()'''