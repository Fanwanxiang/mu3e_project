#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:11:05 2020

@author: wanxiangfan
"""
import numpy as np


l = 100 # mm total length of the target
l_h = l/2 # mm half length of the target
r = 10  # mm radius of the target
ls = ((l_h)**2 + r**2)**0.5 # slant length
angle = np.arctan(r/l_h)


    
"""def functions to plot distribution"""
def creat_events_on_a_cone(N, lmax, theta): # and return the original events #lmax is the slant length
    """used to generate the distribution along slant height of the cone"""
    x = np.random.uniform(-1,1,N)
    mask1 = x > 0
    mask2 = x <= 0
    x1 = x[mask1] # uniform disribution
    l1 = lmax * np.sqrt(abs(x1))
    x2 = x[mask2] # uniform disribution
    l2 = -lmax * np.sqrt(abs(x2))
    l = np.concatenate([l1,l2])  
    """This cell is used to generate the distribution of phi angle of the cone"""
    phi_max = 2 * np.pi
    phi = np.random.uniform(0, phi_max, N).astype('f')
    """x-direction"""
    x = l * np.sin(theta) * np.cos(phi)
    """y-direction"""
    y = l * np.sin(theta) * np.sin(phi)
    """z-direction"""
    z1 = (l1 - lmax) * np.cos(theta)  # shift the cone by lmax*cos(theta) in z-direction
    z2 = (l2 + lmax) * np.cos(theta)
    z = np.concatenate([z1, z2])
    return x,y,z,l,phi
x,y,z,_,_ = creat_events_on_a_cone(N = 10**5, lmax = ls, theta = angle)
position =np.transpose(np.array([x,y,z]))
np.savetxt("position.txt", position)
#xp = np.loadtxt("array.txt")