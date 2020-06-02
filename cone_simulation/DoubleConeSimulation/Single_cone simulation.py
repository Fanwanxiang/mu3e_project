#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:29:54 2019

@author: wanxiangfan
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize    # to define a universal function for array
"""adjust the N to change the number of samples
   adjust lmax and theta to change the geometry of the cone
"""
N = 10**7                    # sample number
lmax = 1 # maximun length
theta = np.pi/6  #  angles
A = 2/lmax**2  # normalised factor
#%%
"""This cell is used to generate the distribution along slant height of the cone"""
x = np.random.uniform(0,1,N) # uniform disribution
l = lmax * np.sqrt(x)
count, bins, ignored = plt.hist(l, bins = "auto", normed=True) # draw the histagram of l
#print(bins) 
plt.plot(bins, A * bins, linewidth=2, color='r')
plt.xlabel('Slant height l')
plt.ylabel('Pdf') 
plt.title('Probility distribution along slant height' )
plt.show()
#%%
"""This cell is used to generate the distribution of phi angle of the cone"""
phi_max = 2*np.pi
phi = np.random.uniform(0,phi_max,N)
count, bins, ignored = plt.hist(phi, bins = "auto", normed=True) # draw the histagram of l
#print(bins) 
plt.plot(bins,1/phi_max * np.ones_like(bins), linewidth=2, color='r')
plt.xlabel('Phi')
plt.ylabel('Pdf') 
plt.title('Probility distribution on the angle phi')
plt.show()
#%%
"""This cell is used to 2D distrbution on l and phi"""
plt.hist2d(l, phi, bins=(40, 40), cmap=plt.cm.jet) # the graph is resonable 
plt.colorbar()
plt.xlabel('Slant height l')
plt.ylabel('azimuthal angle Phi') 
plt.title('2D distribution on (l,phi)')
plt.show()
#%%
"""use coordinates transformation to calculate the distribution on (x,y,z)
   In a 2D manifold(cone surface), there are 2 degree of freedom (l,phi)"""
"""x-direction"""
x = l*np.sin(theta)*np.cos(phi)
count, bins, ignored = plt.hist(x, bins= 'auto', normed=True)
plt.xlabel('x')
plt.ylabel('Pdf') 
#plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
plt.title('decay distribution in x-direction')
plt.show()
#%% 
"""y-direction"""
y = l*np.sin(theta)*np.sin(phi)
count, bins, ignored = plt.hist(y, bins= 'auto', normed=True)
plt.xlabel('y')
plt.ylabel('Pdf') 
#plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
plt.title('decay distribution in y-direction')
plt.show()
#%%
"""z-direction"""
z = l*np.cos(theta)
count, bins, ignored = plt.hist(z, bins= 'auto', normed=True)
plt.xlabel('z')
plt.ylabel('Pdf') 
#plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
plt.title('decay distribution in z-direction')
plt.show()
#%%
"""2d distribution on (x,y)"""
plt.hist2d(x, y, bins=(40, 40), cmap=plt.cm.jet) # the graph is resonable 
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y') 
plt.title('2D distribution on (x,y)')
plt.axis('equal')
plt.show()
"""2d distribution on (x,z)"""
plt.hist2d(x, z, bins=(40, 40), cmap=plt.cm.jet) # the graph is resonable 
plt.colorbar()
plt.xlabel('x')
plt.ylabel('z') 
plt.title('2D distribution on (x,z)')
plt.axis('equal')
plt.show()
"""2d distribution on (y,z)"""
plt.hist2d(y, z, bins=(40, 40), cmap=plt.cm.jet) # the graph is resonable 
plt.colorbar()
plt.xlabel('y')
plt.ylabel('z') 
plt.title('2D distribution on (y,z)')
plt.axis('equal')
plt.show()




















