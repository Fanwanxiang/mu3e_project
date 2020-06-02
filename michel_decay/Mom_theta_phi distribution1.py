#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:58:43 2019

@author: wanxiangfan
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize    # to define a universal function for array
#%%
N = 10**7# 10**7                    # sample number
rho = 3/4                      # rho is constant and =3/4 in SM
momMax = 52.8                  # max monmentum of electrons
electron_Mass = 0.5109989461   # MeV , rest mass of electrons
@vectorize                     # to define a universal function for array
def p(x):                      # rho = 3/4
    p=4*x**2*(3*(1-x)+2*rho*(4/3*x-1))
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

count, bins, ignored = plt.hist(mom_accept, bins= 100, density=True) # draw the histagram of s_theta
#print(bins) 
plt.plot(bins, (6*(bins/52.8)**2-4*(bins/52.8)**3)/52.8, linewidth=2, color='r')
plt.xlabel('Momentum of electrons (MeV)')
plt.ylabel('Pdf') 
plt.title('Momentum spectrum of Michel muon decay')
plt.show()

#===============================================================================
#%%
N_mom_accept = len(mom_accept) # number of momentum samples 
s = np.random.uniform(0,1,N_mom_accept) 
s_theta = np.arccos(1-2*s)
#hist, bin_edges  = np.histogram(s, bins= 20)

'''np.all(s >= 0)
True
np.all(s < 1)
True
count, bins, ignored = plt.hist(s, 15, normed=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()'''
np.all(s_theta >= 0)
True
np.all(s_theta <= np.pi)
True
count, bins, ignored = plt.hist(s_theta, bins= 100, normed=True) # draw the histagram of s_theta
#print(bins) 
plt.plot(bins, 0.5*np.sin(bins), linewidth=2, color='r')
plt.xlabel('Theta')
plt.ylabel('Pdf') 
plt.title('Theta Angle spectrum of Michel muon decay')

plt.show()

#===============================================================================
#%%
phi = np.random.uniform(0,2*np.pi,N_mom_accept) 
count, bins, ignored = plt.hist(phi, bins= 100, normed=True) # draw the histagram of s_theta
plt.xlabel('phi')
plt.ylabel('Pdf') 
plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
plt.title('Phi Angle spectrum of Michel muon decay')
plt.show()

#===============================================================================
#%%
'''Sphereical coordinates transform to the Cartisian coordiantes
'''
px = mom_accept*np.sin(s_theta)*np.cos(phi)
count, bins, ignored = plt.hist(px, bins= 100, normed=True)
plt.xlabel('momentum in x direction (Mev)')
plt.ylabel('Pdf') 
#plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
plt.title('Momentum spectrum of Michel muon decay in x-direction (Mev)')
plt.savefig('Momentum spectrum of Michel muon decay in x-direction (Mev).png',dpi=600)
plt.show()

#%%
py = mom_accept*np.sin(s_theta)*np.sin(phi)
count, bins, ignored = plt.hist(px, bins= 100, normed=True)
plt.xlabel('momentum in y direction (Mev)')
plt.ylabel('Pdf') 
#plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
plt.title('Momentum spectrum of Michel muon decay in y-direction (Mev)')
plt.savefig('Momentum spectrum of Michel muon decay in y-direction (Mev).png',dpi=600)
plt.show()

#%%
pz = mom_accept*np.cos(s_theta)
count, bins, ignored = plt.hist(pz, bins= 100, normed=True)
plt.xlabel('momentum in z direction (Mev)')
plt.ylabel('Pdf') 
#plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
plt.title('Momentum spectrum of Michel muon decay in z-direction (Mev)')
plt.savefig('Momentum spectrum of Michel muon decay in z-direction (Mev).png',dpi=600)
plt.show()

# calculate energy of the electron using relativistic dispersion relation
eletron_energy = np.sqrt(mom_accept**2 + electron_Mass**2)

list1= np.array([px,py,pz,eletron_energy])
mometum_energy_list = np.transpose(list1)
np.savetxt("michel_decay_momentum_energy.txt", mometum_energy_list)


plt.hist(px, bins=30, alpha=0.3,label='px', histtype = 'stepfilled')
plt.hist(py, bins=30, alpha=0.3, label='py', histtype = 'stepfilled')
#plt.hist(py, bins=40, alpha=0.3, label='pz', histtype = 'step')
plt.legend(loc='upper right')
plt.show()