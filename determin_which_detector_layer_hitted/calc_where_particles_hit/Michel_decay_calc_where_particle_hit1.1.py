#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:18:49 2020

@author: wanxiangfan
"""
#so far the simulation only consider the Outer pixel layers;
#the simulation can be easily extended to Recurl pixel layers


#from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
# factor of Natural unit --> SI unit

#initial conditions of electrons/positrons
#four momentum in natural units			
#E0 = 33.018*10**(-3)* 1.6022 * 10**(-10)# J     #33.018MeV/c2
#px0 = -6.16226/1000 * 5.3444 * 10**(-19) #kg m s−1    #MeV/c2
#py0 = -27.2006 /bins0 * 5.3444 * 10**(-19) #   #MeV/c2
#pz0 = 17.6658 /bins0 * 5.3444 * 10**(-19) #    #MeV/c2



# =============================================================================
# px0, py0, pz0, E0 = 0.957442,	5.39776,	-2.32925,	5.97821
# px0, py0, pz0, E0 = px0 * f_mom, py0 * f_mom, pz0 * f_mom, E0 * f_en
# 
# 
# x0, y0, z0 =  0, 0, 50/bins0  # metre
# P_mu0 = np.array([E0, px0, py0, pz0]) #4-momentum
# R0 = np.array([x0, y0, z0]) # initial position
#  
# rest_mass = np.sqrt(E0*E0 -(px0*px0 + py0*py0 + pz0*pz0))
# =============================================================================
def phase_phi(cos, sin):
    tan = sin / cos
    if (sin > 0) and (cos > 0):
        phi = np.arctan(tan)
    elif (sin > 0) and (cos < 0):
        phi = np.arctan(tan) + np.pi
    elif (sin < 0) and (cos < 0):
        phi = np.arctan(tan) - np.pi
    elif (sin < 0) and (cos > 0):
        phi = np.arctan(tan)
    return phi


def time_hitting_on_detector_layer(P_x0, P_y0, E0, x0, y0): #use SI units
    #set constants:
    q = e # for positron
    Bz =  1 # Tesla
    # transverse momentum
    P_T0 = np.sqrt(P_x0**2 + P_y0**2)
    R_detectorlayer = 0.07 #m, 70mm
    R = P_T0 / (q * Bz)
    q = 1.602176634*10**(-19)
    c_x, c_y = P_y0/(q*Bz) + x0 , y0 -P_x0/ (q*Bz)

    omega = q*Bz*c**2 / E0
    
    # calculate two phase angle: alpha and beta
    cos_alpha, sin_alpha = P_y0 / P_T0, P_x0 / P_T0
    alpha  = phase_phi(cos_alpha, sin_alpha)
    cos_beta, sin_beta = c_x / np.sqrt(c_x**2 + c_y**2), c_y / np.sqrt(c_x**2 + c_y**2)
    beta = phase_phi(cos_beta, sin_beta)
    #use % modulus to ensure the time within 1 period
    a = (R**2 + c_x**2 + c_y**2 - R_detectorlayer**2)
    b = 2 * R * np.sqrt(c_x**2 + c_y**2)
    if abs(a/b) <= 1:
        omega_t1 = (np.arccos(a/b) - alpha - beta) % (2*np.pi) 
        omega_t2 = (-np.arccos(a/b) - alpha - beta) % (2*np.pi) 
        t1 = omega_t1 / omega
        t2 = omega_t2 / omega
        if t1 > t2:
            t_min = t2
            t_max = t1
        if t1 < t2:
            t_min = t1
            t_max = t2
        return t_min, t_max
    else:
        return 0, 0


def where_particle_hit_on_detector_layer(t,px0, py0, pz0, E0, x0, y0, z0): #SI unit, calculate where the particle hit on the layer
    q = e # charge of electron    
    omega =q*B_z*c*c/E0 # angular frequency of circular motion in xy-plane
    x_t = 1/(q*B_z) * (px0 * np.sin(omega * t) + py0 * (1 - np.cos(omega * t))) + x0
    y_t = 1/(q*B_z) * (py0 * np.sin(omega * t) - px0 * (1 - np.cos(omega * t))) + y0
    z_t = (c*c/E0) * pz0 * t + z0
    return x_t , y_t, z_t



def plot_hist_2D(sample1, sample2, xlabel, ylabel, title, axis, N_bins):
    plt.hist2d(sample1, sample2, bins=(N_bins, N_bins))  # the graph is resonable
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis(axis)
    plt.savefig(fname=title)
    plt.show()


def plot_hist_1D(sample, xlabel, ylabel, title, bins):
    count, bins, ignored = plt.hist(sample, bins=bins, density=True)  # draw the histagram of l
    # print(bins)
    # plt.plot(bins, A * bins, linewidth=2, color='r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(fname=title)
    plt.show()


def PhasePhi(cos, sin):
    tan = sin / cos
    if (sin > 0) and (cos > 0):
        phi = np.arctan(tan)
    elif (sin > 0) and (cos < 0):
        phi = np.arctan(tan) + np.pi
    elif (sin < 0) and (cos < 0):
        phi = np.arctan(tan) - np.pi
    elif (sin < 0) and (cos > 0):
        phi = np.arctan(tan)
    return phi


# parameters: 
    

f_en = 1.6022 * 10**(-10) * 10**(-3) # energy: Mev --> kg m^2 s^(-2)
f_mom = 5.3444 * 10 ** (-19) * 10**(-3) # momentum: Mev --> kg m s^(-1)
f_mass = 1.7827 * 10**(-27) * 10**(-3) # mass : Mev -->  kg 
# constant
e = 1.602176634*10**(-19) # (C) electron charge magnitude; SI:s⋅A
Electron_MASS = 9.1093837015*10**(-31) #kg   or # 0.000510998910*1000 #MeV/c2
Positron_MASS = 9.1093837015*10**(-31) #kg   or # 0.000510998910*1000 #MeV/c2
# magnetic field
B_z = 1     # Tesla , B_z: 1 - 1.5 T ; SI : kg⋅s^(−2)⋅A^(−1)

c = 299792458   # m/s, speed of light 
# m   The outer pixel layer (double layer) has length 36 cm in z-direction,
# and radius $r3 = 70 mm and r4 = 82 mm$.
# To be simple, firstly, treat the layer as a single layer, set r = 70 mm
R_detector_layer = 76 /1000

bins = 100
#load the initial conditions:
initial_position = np.loadtxt("position.txt")
# for ii in range(2):
#     locals()['initial_4momentum%d'%(ii)] = np.loadtxt("michel_decay_momentum_energy_%d.txt"%(ii))
#initial_4momentum1 = np.loadtxt("michel_decay_momentum_energy_1.txt")
initial_4momentum = np.loadtxt("michel_decay_momentum_energy.txt")
initial_position = np.loadtxt("position.txt")

xi = np.transpose(initial_position)[0]
print(len(xi[xi<0]))
print(len(xi[xi>0]))
xi_asymmetry = abs(len(xi[xi<0])-len(xi[xi>0]))/(len(xi[xi<0])+len(xi[xi>0]))
print('xi_asymmetry=',xi_asymmetry )

yi = np.transpose(initial_position)[2]
print(len(yi[yi<0]))
print(len(yi[yi>0]))
yi_asymmetry = abs(len(yi[yi<0])-len(yi[yi>0]))/(len(yi[yi<0])+len(yi[yi>0]))
print('yi_asymmetry=',yi_asymmetry  )

zi = np.transpose(initial_position)[2]
print(len(zi[zi<0]))
print(len(zi[zi>0]))
zi_asymmetry = abs(len(zi[zi<0])-len(zi[zi>0]))/(len(zi[zi<0])+len(zi[zi>0]))
print('zi_asymmetry=',zi_asymmetry  )

time_position12_initialposition_initial4moemtum_list=[]
#time_position_initialposition_initial4moemtum_list2=[]
for i in range(100000):
    # unit trans mm --> m
    x0 = initial_position[i][0]/1000  
    y0 = initial_position[i][1]/1000
    z0 = initial_position[i][2]/1000
   
    # natural units --> SI units
    px0 = initial_4momentum[i][0] * f_mom
    py0 = initial_4momentum[i][1] * f_mom
    pz0 = initial_4momentum[i][2] * f_mom
    E0 = initial_4momentum[i][3] * f_en
    #calculate the time and position
    t1,t2 = time_hitting_on_detector_layer(px0, py0, E0, x0, y0) #t1 < t2
    if (t1 != 0) and (t2 != 0):
        x1, y1, z1 = where_particle_hit_on_detector_layer(t1,px0, py0, pz0, E0, x0, y0, z0)
        x2, y2, z2 = where_particle_hit_on_detector_layer(t2,px0, py0, pz0, E0, x0, y0, z0)
            # calculate the polar angel in x y plane
        r1 = np.sqrt(x1**2 + y1**2)
        cos1 = x1/r1
        sin1 = y1/r1
        phi1 = PhasePhi(cos1,sin1) 
        
        r2 = np.sqrt(x2**2 + y2**2)
        cos2 = x2/r2
        sin2 = y2/r2
        phi2 = PhasePhi(cos2,sin2) 
        #save the data in SI unit
        time_position12_initialposition_initial4moemtum_list.append([t1, x1, y1,z1, t2, x2, y2, z2, phi1, phi2])#x0, y0, z0, px0, py0, pz0, E0])
    #time_position_initialposition_initial4moemtum_list2.append([])


#statistics of 1st hits
x1_array = np.transpose(time_position12_initialposition_initial4moemtum_list)[1]
y1_array = np.transpose(time_position12_initialposition_initial4moemtum_list)[2]
z1_array = np.transpose(time_position12_initialposition_initial4moemtum_list)[3]
phi1_array = np.transpose(time_position12_initialposition_initial4moemtum_list)[8]


print('x1<0:',len(x1_array[x1_array<0]))
print('x1>0:',len(x1_array[x1_array>0]))
x_asymmetry = abs(len(x1_array[x1_array<0])-len(x1_array[x1_array>0]))/(len(x1_array[x1_array<0])+len(x1_array[x1_array>0]))
print('x_asymmetry:',x_asymmetry)

print('y1<0:',len(y1_array[y1_array<0]))
print('y1>0:',len(y1_array[y1_array>0]))
y_asymmetry = abs(len(y1_array[y1_array<0])-len(y1_array[y1_array>0]))/(len(y1_array[y1_array<0])+len(y1_array[y1_array>0]))
print('y_asymmetry:',y_asymmetry)


print('z1<0:',len(z1_array[z1_array<0]))
print('z1>0:',len(z1_array[z1_array>0]))
z_asymmetry = abs(len(z1_array[z1_array<0])-len(z1_array[z1_array>0]))/(len(z1_array[z1_array<0])+len(z1_array[z1_array>0]))
print('z_asymmetry:',z_asymmetry)

mask1 = (np.abs(z1_array) < 0.18)  
L = ['x','y','z','phi']
for s in L:
    locals()['%s1_array' % s] = locals()['%s1_array' % s][mask1]
    plot_hist_1D(sample = locals()['%s1_array' % s], xlabel = '%s (m)' % s, ylabel = 'PDF', title = 'Distribution of %s (1st hit)' % s, bins =bins)

arc1_array = phi1_array*R_detector_layer
#x1_array = x1_array[mask1]
#y1_array = y1_array[mask1]
#z1_array = z1_array[mask1]
#print("number of particles hitting on the layer at t1",len(x1_array[mask1]))
print('z1<0:',len(z1_array[z1_array<0]))
print('z1>0:',len(z1_array[z1_array>0]))
z_asymmetry = abs(len(z1_array[z1_array<0])-len(z1_array[z1_array>0]))/(len(z1_array[z1_array<0])+len(z1_array[z1_array>0]))
print('z_asymmetry:',z_asymmetry)

plot_hist_2D(sample1=x1_array, sample2=z1_array, xlabel='x(m)',
             ylabel='z(m)', title='2D distribution on (x,z) (1st hit)',
             axis='tight',N_bins = bins) 
plot_hist_2D(sample1=y1_array, sample2=z1_array, xlabel='y(m)',
             ylabel='z(m)', title='2D distribution on (y,z) (1st hit)',
             axis='tight',N_bins = bins) 
plot_hist_2D(sample1=x1_array, sample2=y1_array, xlabel='x(m)',
             ylabel='y(m)', title='2D distribution on (x,y) (1st hit)',
             axis='equal',N_bins = bins )
plot_hist_2D(sample1=arc1_array, sample2=z1_array, xlabel='arc (m)',
             ylabel='z (m)', title='2D distribution on (arc,z) (1st hit)',
             axis='equal',N_bins = bins ) 

# for s in L:
#     plot_hist_1D(sample = locals()['%s1_array' % s], xlabel = '%s (m)' % s, ylabel = 'PDF', title = 'Distribution of %s 1st hit' % s, bins =100)
# plot_hist_1D(sample = y1_array, xlabel = 'y (m)', ylabel = 'PDF', title = 'Distribution of y 1st hit', bins =100)
# plot_hist_1D(sample = z1_array, xlabel = 'z (m)', ylabel = 'PDF', title = 'Distribution of z 1st hit', bins =100)    
# plot_hist_1D(sample = arc1_array, xlabel = 'arc (m)', ylabel = 'PDF', title = 'Distribution of Arc 1st hit', bins =100)
#expand the cylinder surface into a plane:


 
# statistics of second hits
x2_array = np.transpose(time_position12_initialposition_initial4moemtum_list)[5]
y2_array = np.transpose(time_position12_initialposition_initial4moemtum_list)[6]
z2_array = np.transpose(time_position12_initialposition_initial4moemtum_list)[7]
phi2_array = np.transpose(time_position12_initialposition_initial4moemtum_list)[9]
mask2 = (np.abs(z2_array) < 0.18)  
L = ['x','y','z', 'phi']
for s in L:
    locals()['%s2_array' % s] = locals()['%s2_array' % s][mask2]
    plot_hist_1D(sample = locals()['%s2_array' % s], xlabel = '%s (m)' % s, ylabel = 'PDF', title = 'Distribution of %s (2nd hit)' % s, bins = bins)

arc2_array = phi2_array*R_detector_layer


plot_hist_2D(sample1=x2_array, sample2=z2_array, xlabel='x(m)',
             ylabel='z(m)', title='2D distribution on (x,z) (2nd hit)',
             axis='tight',N_bins = bins)  
plot_hist_2D(sample1=y2_array, sample2=z2_array, xlabel='y(m)',
             ylabel='z(m)', title='2D distribution on (y,z) (2nd hit)',
             axis='tight',N_bins = bins) 
plot_hist_2D(sample1=x2_array, sample2=y2_array, xlabel='x(m)',
             ylabel='y(m)', title='2D distribution on (x,y) (2nd hit)',
             axis='equal',N_bins = bins )
plot_hist_2D(sample1=arc2_array, sample2=z2_array, xlabel='arc (m)',
             ylabel='z (m)', title='2D distribution on (arc,z) (2nd hit)',
             axis='equal',N_bins = bins ) 
# for s in L: 
#     locals()['%s12_array' % s] = np.concatenate((locals()['%s1_array' % s], locals()['%s2_array' % s]),axis= None)
#     plot_hist_1D(sample = locals()['%s12_array' % s], xlabel = '%s (m)' % s, ylabel = 'PDF', title = 'Distribution of %s 1st + 2nd hit' % s, bins =100)

# plot_hist_2D(sample1=x12_array, sample2=z12_array, xlabel='x(m)',
#              ylabel='z(m)', title='2D distribution on (x,z) 1st + 2nd hit',
#              axis='equal',N_bins = 40) 
# plot_hist_2D(sample1=y12_array, sample2=z12_array, xlabel='y(m)',
#              ylabel='z(m)', title='2D distribution on (y,z) 1st + 2nd hit',
#              axis='equal',N_bins = 40) 
# plot_hist_2D(sample1=x12_array, sample2=y12_array, xlabel='x(m)',
#              ylabel='y(m)', title='2D distribution on (y,z) 1st + 2nd hit',
#              axis='equal',N_bins = 40 )
# plot_hist_2D(sample1=arc12_array, sample2=z12_array, xlabel='arc (m)',
#               ylabel='z (m)', title='2D distribution on (arc,z) 1st hit',
#               axis='equal',N_bins = 40 ) 
# plot_hist_1D(sample = x2_array, xlabel = 'x (m)', ylabel = 'PDF', title = 'Distribution of x 2nd hit',bins=bins)
# plot_hist_1D(sample = y2_array, xlabel = 'y (m)', ylabel = 'PDF', title = 'Distribution of y 2nd hit',bins=bins)
# plot_hist_1D(sample = z2_array, xlabel = 'z (m)', ylabel = 'PDF', title = 'Distribution of z 2nd hit',bins=bins)  

# px0, py0, pz0, E0 = 0.957442,	5.39776,	-2.32925,	5.97821
# px0, py0, pz0, E0 = px0 * f_mom, py0 * f_mom, pz0 * f_mom, E0 * f_en
# 
# 
# x0, y0, z0 =  0, 0, 50/bins0  # metre
# P_mu0 = np.array([E0, px0, py0, pz0]) #4-momentum
# R0 = np.array([x0, y0, z0]) # initial position
#  
# rest_mass = np.sqrt(E0*E0 -(px0*px0 + py0*py0 + pz0*pz0))

# =============================================================================
# starting_guess = 0
# t = fsolve(x_t,starting_guess, x, E0, px0, py0 , x0)
# print(t)
# =============================================================================
# =============================================================================
# def position(t, Ei, pxi, pyi, pzi, xi, yi, zi): # return the position after t
#     q = -e # charge of electron    
#     #E0, px0, py0, pz0 = P_mu[0], P_mu[1], P_mu[2], P_mu[3]
#     #x0, y0, z0 = R[0], R[1], R[2]
#     omega =q*B_z*c*c/Ei # angular frequency of circular motion in xy-plane
#     x_t = 1/(q*B_z)*(pxi * np.sin(omega * t) + pyi * (1 - np.cos(omega * t))) + xi
#     y_t = 1/(q*B_z)*(pyi * np.sin(omega * t) - pxi * (1 - np.cos(omega * t))) + yi
#     z_t = (c*c/Ei) * pzi * t + zi
#     return x_t, y_t, z_t
# print(position(1,E0, px0, py0, pz0, x0, y0, z0))
# print(position(1,P_mu0, R0))
# 
# 
# def func(t, P_mu, R):
#     #x_t, y_t, z_t = position(t, P_mu, R)
#     q = -e # charge of electron    
#     E0, px0, py0, pz0 = P_mu[0], P_mu[1], P_mu[2], P_mu[3]
#     x0, y0, z0 = R[0], R[1], R[2]
#     omega =q*B_z*c*c/E0 # angular frequency of circular motion in xy-plane
#     x_t = 1/(q*B_z) * (px0 * np.sin(omega * t) + py0 * (1 - np.cos(omega * t))) + x0
#     y_t = 1/(q*B_z) * (py0 * np.sin(omega * t) - px0 * (1 - np.cos(omega * t))) + y0
#     #z_t = (c*c/E0) * pz0 * t + z0
#     return x_t**2 + y_t**2 - R_detector_layer ** 2
# 
# t = fsolve(func, P_mu0, R0)
# =============================================================================
