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
N_bins = 40  # number of bins
R =10 * 3*10**9 # particles per unit time -- unit: 1/s
dt = 10**(-6)    #time resolution 
dx = 50*10**(-1) #space resolution
dy = 50*10**(-1)
dz = 50*10**(-1)
N_in_dt = R * dt #number of particles decays on the cone in dt
N1 = int(N_in_dt/2)
#N = N1 + N2  # sample number
#lmax =  # maximun slant length
#theta =   # angles


"""def functions to plot distribution"""
def creat_events_on_a_cone(N1, lmax, theta): # use lib: mayavi from mlab
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
    return x,y,z,l,phi,


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
"""select the 3 electron events which is closed to each other on the surface of
the cone"""
x,y,z,l,phi = creat_events_on_a_cone(N1, lmax = 10, theta = np.pi / 12)
x1,x2,x3 = np.array_split(x, 3)
y1,y2,y3 = np.array_split(y, 3)
z1,z2,z3 = np.array_split(z, 3)
#distance of each points in x,y,z directions 
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
x_all = np.concatenate([x1_accept, x2_accept, x3_accept])
y_all = np.concatenate([y1_accept, y2_accept, y3_accept])
z_all = np.concatenate([z1_accept, z2_accept, z3_accept])
#allfalse = a = np.full(len(x1), False)
#if mask.all() == false:
#    print("there are no three points which are indistinguishable")
#else:
figure = mlab.figure('DensityPlot')
mlab.points3d(x_all,y_all,z_all,scale_mode='none', scale_factor=0.07)
mlab.axes()
mlab.show()

# print(l1)
# print(l)
print("3electrons decay")
plot_hist_1D(sample=x_all, xlabel='x', ylabel='Pdf', title='3 electrons Probility distribution along x')
plot_hist_1D(sample=y_all, xlabel='y', ylabel='Pdf', title='3 electrons Probility distribution along y')
plot_hist_1D(sample=x_all, xlabel='z', ylabel='Pdf', title='3 electrons Probility distribution along z')
plot_hist_2D(sample1=x_all, sample2=y_all, xlabel='x',
             ylabel='y', title='3electrons 2D distribution on (x,y)',
             axis='equal')
plot_hist_2D(sample1=y_all, sample2=z_all, xlabel='y',
             ylabel='z', title='3electrons 2D distribution on (y,z)',
             axis='equal')
plot_hist_2D(sample1=x_all, sample2=z_all, xlabel='x',
             ylabel='z', title='3electrons 2D distribution on (x,z)',
             axis='equal')

#%%
print("single electron decay")
plot_hist_1D(sample=l, xlabel='Slant height l', ylabel='Pdf', title='Probility distribution along slant height')

'''count, bins, ignored = plt.hist(l, bins = "auto", normed=True) # draw the histagram of l
#print(bins) 
#plt.plot(bins, A * bins, linewidth=2, color='r')
plt.xlabel('Slant height l')
plt.ylabel('Pdf') 
plt.title('Probility distribution along slant height' )
plt.show()'''

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

plot_hist_1D(sample=x, xlabel='x', ylabel='Pdf', title='decay distribution in x-direction')
'''count, bins, ignored = plt.hist(x, bins= 'auto', normed=True)
plt.xlabel('x')
plt.ylabel('Pdf') 
#plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
plt.title('decay distribution in x-direction')
plt.show()'''

plot_hist_1D(sample=y, xlabel='y', ylabel='Pdf', title='decay distribution in y-direction')
'''count, bins, ignored = plt.hist(y, bins= 'auto', normed=True)
plt.xlabel('y')
plt.ylabel('Pdf') 
#plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
plt.title('decay distribution in y-direction')
plt.show()'''

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


#plot_cone_surface(x,y,z)
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

