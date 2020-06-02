#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:33:39 2020

@author: wanxiangfan
"""

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
'''
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
'''  

#N_3e = find_out_N_coincidences(N1 = N_half, lmax = ls, theta = angle)
#print("N_3e = ",N_3e)

#print(find_out_N_coincidences(N1=N_half, lmax= ls, theta= angle))
def generating_N_events_poisson(N_mean, no_of_events):
    Events_list = np.random.poisson(N_mean, no_of_events)
    return Events_list



#2    
'''L = []
count = 0
for i in range(10000):
    N_3e = find_out_N_coincidences(N1 = N_half, lmax = ls, theta = angle)
    count += 1
    print("N_3e =", N_3e,"count =", count)
    if N_3e == 1:
        L.append(N_3e)'''
#4 Calculate N as in 1 or 2 above, but instead of generating N events you generate 
#a random number N' from a Poisson pdf with mean N, then generate N' events (and so on).
def poisson_simulation():
    N_poisson = generating_N_events_poisson(N_mean = R*dt, no_of_events = 10)
    L = []
    L_3e = [] #select the event when conincidence happens.
    count = 0
    print("N_mean =", R*dt)
    print("N is the number of muons decay on the target at each time interval dt")
    print("| N_3e ","| N ","| time(ns)|" )
    for i in range(len(N_poisson)):
        N_3e = find_out_N_coincidences(N=N_poisson[i], lmax = ls, theta = angle)
        count += 1
        print(N_3e,"|",N_poisson[i],"|", count)
        L.append([N_3e,N_poisson[i],count])
        if N_3e > 0:
            L_3e.append([N_3e,N_poisson[i],count])