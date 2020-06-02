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
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy import special as spc
from scipy import stats
from mayavi import mlab
from itertools import combinations 
from numba import vectorize    # to define a universal function for array

"""adjust the N to change the number of samples
   adjust lmax and theta to change the geometry of the cone
"""

R = 100 # Ghz particles per unit time -- Particle Rate; upper limit of R that this laptop can simulate ~6*10^7
dt = 1  # ns  i.e. 1ns #time resolution in second
dx, dy, dz = 0.05, 0.05, 0.05 #space resolution in meter 50 micorns = 0.05 mm
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
@vetorize
def creat_events_on_a_cone(N, theta): # and return the original events
    """used to generate the distribution along slant height of the cone"""
    #N2 = N1
    #N = N1 + N2
    ##A = 2 / l ** 2  # normalised factor
    #x1 = np.random.uniform(0, 1, N1).astype('f')  # uniform disribution
    #l1 = l * np.sqrt(x1)
    #x2 = np.random.uniform(0, 1, N2).astype('f')  # uniform disribution
    #l2 = -l * np.sqrt(x2)
    #l = np.concatenate([l1, l2])   
    x = np.random.uniform(-1,1,N)
    mask1 = x > 0
    mask2 = x <= 0
    x1 = x[mask1] # uniform disribution
    l1 = l * np.sqrt(abs(x1))
    x2 = x[mask2] # uniform disribution
    l2 = -l * np.sqrt(abs(x2))
    l = np.concatenate([l1,l2])
    """This cell is used to generate the distribution of phi angle of the cone"""
    phi_max = 2 * np.pi
    phi = np.random.uniform(0, phi_max, N).astype('f')
    """x-direction"""
    x = l * np.sin(theta) * np.cos(phi)
    """y-direction"""
    y = l * np.sin(theta) * np.sin(phi)
    """z-direction"""
    z1 = (l1 - l) * np.cos(theta)  # shift the cone by l*cos(theta) in z-direction
    z2 = (l2 + l) * np.cos(theta)
    z = np.concatenate([z1, z2])
    #print('len(x)',len(x))
    return x,y,z,l,phi


def find_out_3_coincidences(lmax,theta):
    N = 3
    x,y,z,l,phi = creat_events_on_a_cone(N, lmax, theta)
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    z1 = z[0]
    z2 = z[1]
    z3 = z[2]
    x12 = np.absolute(x1-x2)
    x13 = np.absolute(x1-x3)
    x23 = np.absolute(x2-x3)
    y12 = np.absolute(y1-y2)
    y13 = np.absolute(y1-y3)
    y23 = np.absolute(y2-y3)
    z12 = np.absolute(z1-z2)
    z13 = np.absolute(z1-z3)
    z23 = np.absolute(z2-z3)
    mask = (x12 < dx) & (x13 < dx) & (x23 < dx) & (y12 < dy) & (y13 < dy) & (y23 < dy) & (z12 < dz) & (z13 < dz) & (z23 < dz)
    x1_accept = x1[mask] 
    N_coincidence = len(x1_accept) # number of coincidence
    return N_coincidence
def find_out_N_coincidences(N, lmax, theta): # return the number of coincidence only
    x,y,z,l,phi = creat_events_on_a_cone(N, lmax, theta)
    r = np.array([x,y,z])
    #print("r =", r)
    r_T = np.matrix.transpose(r) # the list of position of N points
    #print("r_T =",r_T)
    Nc3 =  np.array(list(combinations(r_T,3))) #find all possible combinations of points in list r_T --combination N choose 3 
    #print("Nc3 =",Nc3)
    #print(Nc3[0][0])
    #Nc3_T = np.matrix.transpose(Nc3)
    #print("Nc3_T =",Nc3_T)
    x1 = Nc3[:,0,0]
    y1 = Nc3[:,0,1]
    z1 = Nc3[:,0,2]
    x2 = Nc3[:,1,0]
    y2 = Nc3[:,1,1]
    z2 = Nc3[:,1,2]
    x3 = Nc3[:,2,0]
    y3 = Nc3[:,2,1]
    z3 = Nc3[:,2,2]
    #print("x1 =", x1,"y1 =", y1 ," z1= ", z1)
    '''x1,x2,x3 = np.array_split(x, 3)
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
    phi1, phi2, phi3 = phi1[:length], phi2[:length], phi3[:length]'''
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
    N_coincidence = len(x1_accept) # number of coincidence
    return N_coincidence



def simulation_3_events(lmax,theta):
    L = []
    L_3e = [] #select the event when conincidence happens.
    count = 0
    no_of_events = 10**(16)
    print('N_3e: |','count')
    for i in range(no_of_events):
        N_3e = find_out_3_coincidences(lmax,theta) #number of mu
        count += 1
        print(N_3e,"|", count)
        L.append([N_3e])
        if N_3e > 0:
            L_3e.append([N_3e])
    if len(L_3e) == 0:
        print("Probability <", 1/len(L))
    else:
        print("probability =", len(L_3e)/len(L))
simulation_3_events(lmax = l, theta = angle)
