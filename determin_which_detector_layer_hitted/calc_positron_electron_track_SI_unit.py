from scipy.optimize import fsolve
import numpy as np
from mayavi import mlab
# factor of Natural unit --> SI unit
f_en = 1.6022 * 10**(-10) * 10**(-3) # energy: Mev --> kg m^2 s^(-2)
f_mom = 5.3444 * 10 ** (-19) * 10**(-3) # momentum: Mev --> kg m s^(-1)
f_mass = 1.7827 * 10**(-27) * 10**(-3) # mass : Mev -->  kg 
# constant
e = 1.602176634*10**(-19) # (C) electron charge magnitude; SI:s⋅A
Electron_MASS = 9.1093837015*10**(-31) #kg   or # 0.000510998910*1000 #MeV/c2
Positron_MASS = 9.1093837015*10**(-31) #kg   or # 0.000510998910*1000 #MeV/c2
B_z = 1     # T , B_z: 1 - 1.5 T ; SI : kg⋅s^(−2)⋅A^(−1)
c = 299792458   # m/s, speed of light 
R_detector_layer = 60/1000 # m   or 60  mm
#initial conditions of electrons/positrons
#four momentum in natural units			
#E0 = 33.018*10**(-3)* 1.6022 * 10**(-10)# J     #33.018MeV/c2
#px0 = -6.16226/1000 * 5.3444 * 10**(-19) #kg m s−1    #MeV/c2
#py0 = -27.2006 /1000 * 5.3444 * 10**(-19) #   #MeV/c2
#pz0 = 17.6658 /1000 * 5.3444 * 10**(-19) #    #MeV/c2
 
px0, py0, pz0, E0 = 0.957442,	5.39776,	-2.32925,	5.97821
 # Mev



x0, y0, z0 =  0, 0, 0
P_mu0 = np.array([E0, px0, py0, pz0]) #4-momentum
R0 = np.array([x0, y0, z0]) # initial position
 
rest_mass = np.sqrt(E0*E0 -(px0*px0 + py0*py0 + pz0*pz0))


def position(t, P_mu, R): # return the position after t
    q = -e # charge of electron    
    #trans MeV --> SI unit
    E0, px0, py0, pz0 = P_mu[0] * f_en, P_mu[1] * f_mom, P_mu[2] * f_mom, P_mu[3] * f_mom 
    x0, y0, z0 = R[0], R[1], R[2]
    omega =q*B_z*c*c/E0 # angular frequency of circular motion in xy-plane
    x_t = 1/(q*B_z) * (px0 * np.sin(omega * t) + py0 * (1 - np.cos(omega * t))) + x0
    y_t = 1/(q*B_z) * (py0 * np.sin(omega * t) - px0 * (1 - np.cos(omega * t))) + y0
    z_t = (c*c/E0) * pz0 * t + z0
    return x_t, y_t, z_t
'''
def position(t, Ei, pxi, pyi, pzi, xi, yi, zi): # return the position after t
    q = -e # charge of electron    
    #E0, px0, py0, pz0 = P_mu[0], P_mu[1], P_mu[2], P_mu[3]
    #x0, y0, z0 = R[0], R[1], R[2]
    omega =q*B_z*c*c/Ei # angular frequency of circular motion in xy-plane
    x_t = 1/(q*B_z)*(pxi * np.sin(omega * t) + pyi * (1 - np.cos(omega * t))) + xi
    y_t = 1/(q*B_z)*(pyi * np.sin(omega * t) - pxi * (1 - np.cos(omega * t))) + yi
    z_t = (c*c/Ei) * pzi * t + zi
    return x_t, y_t, z_t
print(position(1,E0, px0, py0, pz0, x0, y0, z0))'''
print(position(1,P_mu0, R0))


def func(t, P_mu, R):
    #x_t, y_t, z_t = position(t, P_mu, R)
    q = -e # charge of electron    
    E0, px0, py0, pz0 = P_mu[0], P_mu[1], P_mu[2], P_mu[3]
    x0, y0, z0 = R[0], R[1], R[2]
    omega =q*B_z*c*c/E0 # angular frequency of circular motion in xy-plane
    x_t = 1/(q*B_z) * (px0 * np.sin(omega * t) + py0 * (1 - np.cos(omega * t))) + x0
    y_t = 1/(q*B_z) * (py0 * np.sin(omega * t) - px0 * (1 - np.cos(omega * t))) + y0
    #z_t = (c*c/E0) * pz0 * t + z0
    return x_t**2 + y_t**2 - R_detector_layer ** 2

t = fsolve(func, P_mu0, R0)
#print(t)


#plot_3d()