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
# from numba import vectorize    # to define a universal function for array
"""adjust the N to change the number of samples
   adjust lmax and theta to change the geometry of the cone
"""
N_bins = 40  # number of bins
N1 = 8000

N2 = N1
N = N1 + N2  # sample number
lmax = 10  # maximun slant length
theta = np.pi / 12  # angles
A = 2 / lmax ** 2  # normalised factor

"""def functions to plot distribution"""


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

def plot_cone_surface(x,y,z):
    # Some parameters (note size only really changes scale on axis)
    #sizeX,sizeY = 1,2
    #amplitude = 1
    #nPoints = 500

    # Calculate wavenumbers
    #kx = nX*pi/sizeX
    #ky = nY*pi/sizeY

    X = np.array(x)
    Y = np.array(y)
    Z = np.array(z)#amplitude*np.sin(kx*X)*np.sin(ky*Y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X,Y,Z,cmap=cm.hsv) # https://matplotlib.org/examples/mplot3d/trisurf3d_demo.html 
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.zlabel("z")
    #plt.title(title)
    plt.show()
"""This cell is used to generate the distribution along slant height of the cone"""
x1 = np.random.uniform(0, 1, N1).astype('f')  # uniform disribution
l1 = lmax * np.sqrt(x1)
x2 = np.random.uniform(0, 1, N2).astype('f')  # uniform disribution
l2 = -lmax * np.sqrt(x2)
l = np.concatenate([l1, l2])
# print(l1)
# print(l)
plot_hist_1D(sample=l, xlabel='Slant height l', ylabel='Pdf', title='Probility distribution along slant height')

'''count, bins, ignored = plt.hist(l, bins = "auto", normed=True) # draw the histagram of l
#print(bins) 
#plt.plot(bins, A * bins, linewidth=2, color='r')
plt.xlabel('Slant height l')
plt.ylabel('Pdf') 
plt.title('Probility distribution along slant height' )
plt.show()'''

"""This cell is used to generate the distribution of phi angle of the cone"""
phi_max = 2 * np.pi
phi = np.random.uniform(0, phi_max, N).astype('f')
plot_hist_1D(phi, 'Phi', 'Pdf', 'Probility distribution on the angle phi')
'''count, bins, ignored = plt.hist(phi, bins = "auto", normed=True) # draw the histagram of l
#print(bins) 
plt.plot(bins,1/phi_max * np.ones_like(bins), linewidth=2, color='r')
plt.xlabel('Phi')
plt.ylabel('Pdf') 
plt.title('Probility distribution on the angle phi')
plt.show()'''

"""This cell is used to 2D distrbution on l and phi"""
plot_hist_2D(sample1=l, sample2=phi, xlabel='Slant height l',
             ylabel='azimuthal angle Phi', title='2D distribution on (l,phi)',
             axis='auto')
'''plt.colorbar()
plt.xlabel('Slant height l')
plt.ylabel('azimuthal angle Phi') 
plt.title('2D distribution on (l,phi)')
plt.show()'''
"""use coordinates transformation to calculate the distribution on (x,y,z)
   In a 2D manifold(cone surface), there are 2 degree of freedom (l,phi)"""
"""x-direction"""
x = l * np.sin(theta) * np.cos(phi)
plot_hist_1D(sample=x, xlabel='x', ylabel='Pdf', title='decay distribution in x-direction')
'''count, bins, ignored = plt.hist(x, bins= 'auto', normed=True)
plt.xlabel('x')
plt.ylabel('Pdf') 
#plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
plt.title('decay distribution in x-direction')
plt.show()'''
"""y-direction"""
y = l * np.sin(theta) * np.sin(phi)
plot_hist_1D(sample=y, xlabel='y', ylabel='Pdf', title='decay distribution in y-direction')
'''count, bins, ignored = plt.hist(y, bins= 'auto', normed=True)
plt.xlabel('y')
plt.ylabel('Pdf') 
#plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
plt.title('decay distribution in y-direction')
plt.show()'''
"""z-direction"""
z1 = (l1 - lmax) * np.cos(theta)  # shift the cone by lmax*cos(theta) in z-direction
z2 = (l2 + lmax) * np.cos(theta)
z = np.concatenate([z1, z2])
plot_hist_1D(sample=z, xlabel='z', ylabel='Pdf', title='decay distribution in z-direction')

"""2d distribution on (x,y)"""
plot_hist_2D(sample1=x, sample2=y, xlabel='x',
             ylabel='y', title='2D distribution on (x,y)',
             axis='equal')
'''plt.hist2d(x, y, bins=(N_bins , N_bins), cmap=plt.cm.jet) # the graph is resonable 
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y') 
plt.title('2D distribution on (x,y)')
plt.axis('equal')
plt.show()'''
"""2d distribution on (x,z)"""
plot_hist_2D(sample1=z, sample2=x, xlabel='z',
             ylabel='x', title='2D distribution on (x,z)',
             axis='equal')
'''plt.hist2d(z, x, bins=(N_bins , N_bins), cmap=plt.cm.jet) # the graph is resonable 
plt.colorbar()
plt.xlabel('z')
plt.ylabel('x') 
plt.title('2D distribution on (x,z)')
plt.axis('equal')
plt.show()'''
"""2d distribution on (z,x)"""
plot_hist_2D(sample1=z, sample2=y, xlabel='z',
             ylabel='y', title='2D distribution on (z,y)',
             axis='equal')


plot_cone_surface(x,y,z)
'''plt.hist2d(z, y, bins=(N_bins , N_bins), cmap=plt.cm.jet) # the graph is resonable 
plt.colorbar()
plt.xlabel('z')
plt.ylabel('y') 
plt.title('2D distribution on (z,y)')
plt.axis('equal')
plt.show()'''
#fig = plt.figure()
'''ax = fig.add_subplot(111, projection='3d')
ax.scatter(z, y, x, marker='o')
ax.set_xlabel('Z Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('X Label')
plt.show()'''
