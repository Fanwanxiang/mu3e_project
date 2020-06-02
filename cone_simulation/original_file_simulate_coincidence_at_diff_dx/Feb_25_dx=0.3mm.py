#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:38:51 2020

@author: wanxiangfan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:43:13 2019

@author: wanxiangfan
"""
#import time
#start_time = time.time()
import numpy as np
#from matplotlib import cm
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#from scipy import special as spc
#from scipy import stats
#from itertools import combinations 
#from scipy.optimize import curve_fit
# from numba import vectorize    # to define a universal function for array
"""adjust the N to change the number of samples
   adjust lmax and theta to change the geometry of the cone
"""
N_bins = 30  # number of bins
R = 10**(7) # Ghz particles per unit time -- Particle Rate; upper limit of R that this laptop can simulate ~6*10^7
dt = 1  # ns  i.e. 1ns #time resolution in second
#dx, dy, dz = 0.05, 0.05, 0.05 #space resolution in meter 50 micorns = 0.05 mm
#dy = 2*50*10**(-1)
#dz = 2*50*10**(-1)
N_in_dt = R * dt #number of particles decays on the cone in dt
N_half = int(N_in_dt/2)
#N = N1 + N2  # sample number
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


def find_out_coincidences_3electrons_in_1_ns(N ,dx ,lmax ,theta): # and return number of coincidence, 3electtron events and the original events and so on.
    dy = dx
    dz = dx 
    x,y,z,l,phi = creat_events_on_a_cone(N, lmax, theta)
    x1,x2,x3 = np.array_split(x, 3)
    y1,y2,y3 = np.array_split(y, 3)
    z1,z2,z3 = np.array_split(z, 3)
    l1,l2,l3 = np.array_split(l, 3)
    phi1, phi2, phi3 = np.array_split(phi, 3)
    #distance of each points in x,y,z directions 
    length = len(x3) # lenth of the shortest array 
    x1,x2,x3 = x1[:length],x2[:length],x3[:length]
    y1,y2,y3 = y1[:length],y2[:length],y3[:length]
    z1,z2,z3 = z1[:length],z2[:length],z3[:length]
    l1,l2,l3 = l1[:length],l2[:length],l3[:length]
    phi1, phi2, phi3 = phi1[:length], phi2[:length], phi3[:length]
    x12 = np.absolute(x1-x2)
    x13 = np.absolute(x1-x3)
    x23 = np.absolute(x2-x3)
    y12 = np.absolute(y1-y2)
    y13 = np.absolute(y1-y3)
    y23 = np.absolute(y2-y3)
    z12 = np.absolute(z1-z2)
    z13 = np.absolute(z1-z3)
    z23 = np.absolute(z2-z3)
    # find 3 points which are too close to each other
    mask = (x12 < dx) & (x13 < dx) & (x23 < dx) & (y12 < dy) & (y13 < dy) & (y23 < dy) & (z12 < dz) & (z13 < dz) & (z23 < dz)
    x1_accept = x1[mask] 
    #x2_accept = x2[mask] 
    #x3_accept = x3[mask] 
    #y1_accept = y1[mask] 
    #y2_accept = y2[mask] 
    #y3_accept = y3[mask] 
    #z1_accept = z1[mask] 
    #z2_accept = z2[mask] 
    #z3_accept = z3[mask] 
    #l1_accept = l1[mask] 
    #l2_accept = l2[mask] 
    #l3_accept = l3[mask] 
    #phi1_accept = phi1[mask] 
    #phi2_accept = phi2[mask] 
    #phi3_accept = phi3[mask]     
    N_coincidence = len(x1_accept) # number of coincidence
    #x_all = np.concatenate([x1_accept, x2_accept, x3_accept])
    #y_all = np.concatenate([y1_accept, y2_accept, y3_accept])
    #z_all = np.concatenate([z1_accept, z2_accept, z3_accept])
    #l_all = np.concatenate([l1_accept, l2_accept, l3_accept])
    #phi_all = np.concatenate([phi1_accept, phi2_accept, phi3_accept])
    #allfalse = a = np.full(len(x1), False)
    #if mask.all() == false:
    #    print("there are no three points which are indistinguishable")
    #else:xyz = np.vstack([x,y,z])
    
    #xyz = np.vstack([x_all,y_all,z_all])
    #kde = stats.gaussian_kde(xyz)
    #density = kde(xyz)
    
    # Plot scatter with mayavi
    #mlab.figure(title)
    #mlab.points3d(x_all, y_all, z_all, density, scale_mode='none', scale_factor=0.07)
    #mlab.colorbar()
    #mlab.axes()
    #mlab.show()
    #x_3e, y_3e, z_3e, l_3e, phi_3e =x_all, y_all, z_all, l_all, phi_all
    return N_coincidence#, x_3e, y_3e, z_3e, l_3e, phi_3e, x, y, z, l, phi




#x,y,y_err,_,_ = creat_arrays_of_prob_and_dx(n1=10,n2=0,n3=0,N=1000)
    
with open("dx=0.3mm.txt",mode='a') as f:
    N_coincidence = find_out_coincidences_3electrons_in_1_ns(N = 3*R,dx = 0.3, lmax = ls, theta=angle)
    f.write("%d"%(N_coincidence)+"\n")
    #prob.to_csv("dx=0.05mm.txt",sep='',mode='a',header=False)
    #np.savetxt('dx=0.05mm.txt',prob)
#print("--- %s seconds ---" % (time.time() - start_time))


























