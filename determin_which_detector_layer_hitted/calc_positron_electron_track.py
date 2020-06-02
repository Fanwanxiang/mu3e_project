from scipy.optimize import fsolve
import numpy as np
# constant
e = 1.602176634*10**(-19) # (C) electron charge magnitude
Electron_MASS = 9.1093837015*10**(−31) #kg   or # 0.000510998910*1000 #MeV/c2
Positron_MASS = 9.1093837015*10**(−31) #kg   or # 0.000510998910*1000 #MeV/c2
B_z = 1     # T , B_z: 1 - 1.5 T
c = 299792458   # m/s, speed of light 
R_detector_layer = 60/1000  # m   or 60  mm
#initial conditions of electrons/positrons
#four momentum in natural units			
E0 = 33.018    #MeV/c2
px0 = -6.16226    #MeV/c2
py0 = -27.2006    #MeV/c2
pz0 = 17.6658     #MeV/c2
x0, y0, z0 =  0, 0, 0
P_mu = np.array([E0, px0, py0, pz0]) #4-momentum
R = np.array([x0, y0, z0]) # initial position
 
rest_mass = np.sqrt(E0*E0 -(px0*px0 + py0*py0 + pz0*pz0))



def R(t, P_mu, R): # return the position after t
    q = -e # charge of electron    
    E0, px0, py0, pz0 = P_mu[0], P_mu[1], P_mu[2], P_mu[3]
    x0, y0, z0 = R[0], R[1], R[2]
    omega =q*B_z*c*c/E0 # angular frequency of circular motion in xy-plane
    x_t = 1/(q*B_z)(px0 * np.sin(omega * t) + py0 * (1 - np.cos(omega * t))) + x0
    y_t = 1/(q*B_z)(py0 * np.sin(omega * t) - px0 * (1 - np.cos(omega * t))) + y0
    z_t = (c*c/E0) * pz0 * t + z0
    return x_t, y_t, z_t

def func(t, P_mu, R):
    x_t, y_t, z_t = R(t, P_mu, R)
    return x_t**2 + y_t**2 - R_detector_layer ** 2

t = fsolve(func,0, P_mu, R)