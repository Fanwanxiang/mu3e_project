#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:18:49 2020

@author: wanxiangfan
"""

#from scipy.optimize import fsolve
import numpy as np
# factor of Natural unit --> SI unit
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
R_detector_layer = 60/1000 # m   or 60  mm

#initial conditions of electrons/positrons
#four momentum in natural units			
#E0 = 33.018*10**(-3)* 1.6022 * 10**(-10)# J     #33.018MeV/c2
#px0 = -6.16226/1000 * 5.3444 * 10**(-19) #kg m s−1    #MeV/c2
#py0 = -27.2006 /1000 * 5.3444 * 10**(-19) #   #MeV/c2
#pz0 = 17.6658 /1000 * 5.3444 * 10**(-19) #    #MeV/c2



# =============================================================================
# px0, py0, pz0, E0 = 0.957442,	5.39776,	-2.32925,	5.97821
# px0, py0, pz0, E0 = px0 * f_mom, py0 * f_mom, pz0 * f_mom, E0 * f_en
# 
# 
# x0, y0, z0 =  0, 0, 50/1000  # metre
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
    Bz = 1 # Tesla
    # transverse momentum
    P_T0 = np.sqrt(P_x0**2 + P_y0**2)
    R_detectorlayer = 0.06 #m, 60mm
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
        else:
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


initial_position = np.loadtxt("position.txt")
print(initial_position[0])
initial_4momentum = np.loadtxt("electron4momentum.txt")
print(initial_4momentum[0])

time_position12_initialposition_initial4moemtum_list=[]
#time_position_initialposition_initial4moemtum_list2=[]
for i in range(10):
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
    t1,t2 = time_hitting_on_detector_layer(px0, py0, E0, x0, y0)
    if (t1 != 0) and (t2 != 0):
        x1, y1, z1 = where_particle_hit_on_detector_layer(t1,px0, py0, pz0, E0, x0, y0, z0)
        x2, y2, z2 = where_particle_hit_on_detector_layer(t2,px0, py0, pz0, E0, x0, y0, z0)
        #save the data in SI unit
        time_position12_initialposition_initial4moemtum_list.append([t1, x1, y1,z1, t2, x2, y2, z2, x0, y0, z0, px0, py0, pz0, E0])
    #time_position_initialposition_initial4moemtum_list2.append([])

        
# px0, py0, pz0, E0 = 0.957442,	5.39776,	-2.32925,	5.97821
# px0, py0, pz0, E0 = px0 * f_mom, py0 * f_mom, pz0 * f_mom, E0 * f_en
# 
# 
# x0, y0, z0 =  0, 0, 50/1000  # metre
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