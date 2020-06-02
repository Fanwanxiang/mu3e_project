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

from math import *
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy import special as spc
from scipy import stats
from mayavi import mlab
# from numba import vectorize    # to define a universal function for array
"""adjust the N to change the number of samples
   adjust lmax and theta to change the geometry of the cone
"""
N_bins = 30  # number of bins
R = 6*10000000 # Ghz particles per unit time -- Particle Rate; upper limit of R that this laptop can simulate ~6*10^7
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
def creat_events_on_a_cone(N1, lmax, theta): # and return the original events
    """used to generate the distribution along slant height of the cone"""
    N2 = N1
    N = N1 + N2
    #A = 2 / lmax ** 2  # normalised factor
    x1 = np.random.uniform(0, 1, N1).astype('f')  # uniform disribution
    l1 = lmax * np.sqrt(x1)
    x2 = np.random.uniform(0, 1, N2).astype('f')  # uniform disribution
    l2 = -lmax * np.sqrt(x2)
    l = np.concatenate([l1, l2])   
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

def find_out_N_coincidences(N1, lmax, theta): # return the number of coincidence only
    x,y,z,l,phi = creat_events_on_a_cone(N1, lmax, theta)
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
    N_coincidence = len(x1_accept) # number of coincidence
    return N_coincidence


def find_out_coincidences(N1, lmax, theta): # and return number of coincidence, 3electtron events and the original events and so on.
    x,y,z,l,phi = creat_events_on_a_cone(N1, lmax, theta)
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
    x2_accept = x2[mask] 
    x3_accept = x3[mask] 
    y1_accept = y1[mask] 
    y2_accept = y2[mask] 
    y3_accept = y3[mask] 
    z1_accept = z1[mask] 
    z2_accept = z2[mask] 
    z3_accept = z3[mask] 
    l1_accept = l1[mask] 
    l2_accept = l2[mask] 
    l3_accept = l3[mask] 
    phi1_accept = phi1[mask] 
    phi2_accept = phi2[mask] 
    phi3_accept = phi3[mask] 
    x_all = np.concatenate([x1_accept, x2_accept, x3_accept])
    y_all = np.concatenate([y1_accept, y2_accept, y3_accept])
    z_all = np.concatenate([z1_accept, z2_accept, z3_accept])
    l_all = np.concatenate([l1_accept, l2_accept, l3_accept])
    phi_all = np.concatenate([phi1_accept, phi2_accept, phi3_accept])
    
    N_coincidence = len(x1_accept) # number of coincidence
    
    #allfalse = a = np.full(len(x1), False)
    #if mask.all() == false:
    #    print("there are no three points which are indistinguishable")
    #else:xyz = np.vstack([x,y,z])
    '''
    xyz = np.vstack([x_all,y_all,z_all])
    kde = stats.gaussian_kde(xyz)
    density = kde(xyz)
    
    # Plot scatter with mayavi
    mlab.figure(title)
    mlab.points3d(x_all, y_all, z_all, density, scale_mode='none', scale_factor=0.07)
    mlab.colorbar()
    mlab.axes()
    mlab.show()'''
    x_3e, y_3e, z_3e, l_3e, phi_3e =x_all, y_all, z_all, l_all, phi_all
    return N_coincidence, x_3e, y_3e, z_3e, l_3e, phi_3e, x, y, z, l, phi
'''
figure = mlab.figure('DensityPlot')
pts = mlab.points3d(x_all,y_all,z_all,scale_mode='none', scale_factor=0.07)
mlab.axes()
mlab.show()'''
def plot_hist_1D(sample, xlabel, ylabel, title):
    count, bins, ignored = plt.hist(sample, bins="auto", density=True)  # draw the histagram of l
    # print(bins)
    # plt.plot(bins, A * bins, linewidth=2, color='r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_hist_2D(sample1, sample2, xlabel, ylabel, title, axis):
    plt.hist2d(sample1, sample2, bins=(N_bins, N_bins), cmap=plt.cm.jet)  # the graph is resonable
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis(axis)
    plt.show()

def plot_cone_surface_without_mask(X,Y,Z,title):
    xyz = np.vstack([X,Y,Z])
    kde = stats.gaussian_kde(xyz)
    density = kde(xyz)
    
    # Plot scatter with mayavi
    mlab.figure(title)
    mlab.points3d(X, Y, Z, density, scale_mode='none', scale_factor=0.07)
    mlab.colorbar()
    mlab.axes()
    mlab.show() 


def plot_cone_surface(X,Y,Z,title):
    # Some parameters (note size only really changes scale on axis)
    #sizeX,sizeY = 1,2
    #amplitude = 1
    #nPoints = 500

    # Calculate wavenumbers
    #kx = nX*pi/sizeX
    #ky = nY*pi/sizeY

    xyz = np.vstack([X,Y,Z])
    kde = stats.gaussian_kde(xyz)
    density = kde(xyz)
    
    # Plot scatter with mayavi
    figure = mlab.figure(title)
    figure.scene.disable_render = True
    
    pts = mlab.points3d(X, Y, Z, density, scale_mode='none', scale_factor=0.07) 
    mask = pts.glyph.mask_points
    mask.maximum_number_of_points = X.size
    mask.on_ratio = 1
    pts.glyph.mask_input_points = True
    
    figure.scene.disable_render = False 
    mlab.colorbar()
    mlab.axes()
    mlab.show()
"""select the 3 electron events which is closed to each other on the surface of
the cone"""
# print(l1)
# print(l)
#method1
def plot_N_coincidence_VS_R(R0,N_R,N_test): # N_R: the number of particle rates, N_test: the number of tests at each value of R 
    R_list = []
    N_3e_list =[]
    for i in range(N_R):
        R = R0*2**(i)
        N_list = []
        for j in range(N_test):
            N_half = int(R*dt/2)
            N_3e = N_3e = find_out_N_coincidences(N1 = N_half, lmax = ls, theta = angle)
            N_list.append(N_3e)
        N_average = np.mean(np.array(N_list))
        N_3e_list.append(N_average)
        R_list.append(R)
    y = np.array(N_3e_list)
    x = np.array(R_list)
    plt.plot(x,y)
    plt.show()
print(find_out_N_coincidences(N1=N_half, lmax= ls, theta= angle))

L = []
for i in range(1000):
    N_3e = find_out_N_coincidences(N1 = N_half, lmax = ls, theta = angle)
    print(N_3e)
    if N_3e == 1:
        L.append(N_3e)
print(L)
'''
print("3electrons decay")
N_coincidence, x_3e, y_3e, z_3e, l_3e, phi_3e, x, y, z, l, phi = find_out_coincidences(N1 = N_half, lmax = ls, theta = angle)
plot_cone_surface(x,y,z, title = 'original decays')
plot_cone_surface(x_3e,y_3e,z_3e, title = 'conicidence decays')

plot_hist_1D(sample=x_3e, xlabel='x_3e', ylabel='Pdf', title='3 electrons Probility distribution along x')
plot_hist_1D(sample=x, xlabel='x', ylabel='Pdf', title='Probility distribution along x')

plot_hist_1D(sample=y_3e, xlabel='y_3e', ylabel='Pdf', title='3 electrons Probility distribution along y')
plot_hist_1D(sample=y, xlabel='y', ylabel='Pdf', title='Probility distribution along y')

plot_hist_1D(sample=z_3e, xlabel='z_3e', ylabel='Pdf', title='3 electrons Probility distribution along z')
plot_hist_1D(sample=z, xlabel='z', ylabel='Pdf', title='Probility distribution along z')

plot_hist_1D(sample=phi_3e, xlabel='phi_3e', ylabel='Pdf', title='3 electrons Probility distribution along phi')
plot_hist_1D(phi, 'Phi', 'Pdf', 'Probility distribution on the angle phi')

plot_hist_1D(sample=l_3e, xlabel='l_3e', ylabel='Pdf', title='3 electrons Probility distribution along l')
plot_hist_1D(sample=l, xlabel='Slant height l', ylabel='Pdf', title='Probility distribution along slant height')



plot_hist_2D(sample1=x_3e, sample2=y_3e, xlabel='x_3e',
             ylabel='y_3e', title='3electrons 2D distribution on (x_3e,y_3e)',
             axis='equal')
plot_hist_2D(sample1=x, sample2=y, xlabel='x',
             ylabel='y', title='2D distribution on (x,y)',
             axis='equal')
plot_hist_2D(sample1=y_3e, sample2=z_3e, xlabel='y_3e',
             ylabel='z_3e', title='3electrons 2D distribution on (y_3e,z_3e)',
             axis='equal')
plot_hist_2D(sample1=y, sample2=z, xlabel='y',
             ylabel='z', title='2D distribution on (x,y)',
             axis='equal')
plot_hist_2D(sample1=x_3e, sample2=z_3e, xlabel='x_3e',
             ylabel='z_3e', title='3electrons 2D distribution on (x_3e,z_3e)',
             axis='equal')
plot_hist_2D(sample1=x, sample2=z, xlabel='x',
             ylabel='z', title='2D distribution on (x,z)',
             axis='equal')

plot_hist_2D(sample1=phi_3e, sample2=l_3e, xlabel='phi',
             ylabel='l', title='3electrons 2D distribution on (phi,l)',
             axis='equal')
plot_hist_2D(sample1=phi, sample2=l, xlabel='phi',
             ylabel='l', title='2D distribution on (phi,l)',
             axis='equal')
'''


