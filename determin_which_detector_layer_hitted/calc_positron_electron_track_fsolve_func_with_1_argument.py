from scipy.optimize import fsolve
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# factor of Natural unit --> SI unit
f_en = 1.6022 * 10**(-10) * 10**(-3) # energy: Mev --> kg m^2 s^(-2)
f_mom = 5.3444 * 10 ** (-19) * 10**(-3) # momentum: Mev --> kg m s^(-1)
f_mass = 1.7827 * 10**(-27) * 10**(-3) # mass : Mev -->  kg 


e = 1.602176634*10**(-19) # (C) electron charge magnitude
Electron_MASS = 9.1093837015*10**(-31) #kg   or # 0.000510998910*1000 #MeV/c2
Positron_MASS = 9.1093837015*10**(-31) #kg   or # 0.000510998910*1000 #MeV/c2
B_z = 1     # T , B_z: 1 - 1.5 T
c = 299792458   # m/s, speed of light 
R_detector_layer = 60/1000  # m   or 60  mm


#initial conditions of electrons/positrons
#four momentum in natural units			
# =============================================================================
# E0 = 33.018    #MeV/c2
# px0 = -6.16226    #MeV/c2
# py0 = -27.2006    #MeV/c2
# pz0 = 17.6658     #MeV/c2
# =============================================================================
px0, py0, pz0, E0 = 0.957442,	5.39776,	-2.32925,	5.97821
px0, py0, pz0, E0 = px0 * f_mom, py0 * f_mom, pz0 * f_mom, E0 * f_en
#x0, y0, z0 =  0, 0, 0
x0, y0, z0 =  0, 0, 0
#P_mu = np.array([E0, px0, py0, pz0]) #4-momentum
#R = np.array([x0, y0, z0]) # initial position
 
rest_mass = np.sqrt(E0*E0 -(px0*px0 + py0*py0 + pz0*pz0))



def R(t): # return the position after t
    q = -e # charge of electron    
    #E0, px0, py0, pz0 = P_mu[0], P_mu[1], P_mu[2], P_mu[3]
    #x0, y0, z0 = R[0], R[1], R[2]
    omega =q*B_z*c*c/E0 # angular frequency of circular motion in xy-plane
    period = 2*np.pi / omega
    print('period =', period)
    x_t = 1/(q*B_z)*(px0 * np.sin(omega * t) + py0 * (1 - np.cos(omega * t))) + x0
    #print(x_t)
    y_t = 1/(q*B_z)*(py0 * np.sin(omega * t) - px0 * (1 - np.cos(omega * t))) + y0
    z_t = (c*c/E0) * pz0 * t + z0
    #r_t = [x_t, y_t, z_t]
    return x_t, y_t, z_t

def func(t):
    x_t, y_t = R(t)[0], R(t)[1]
    return x_t**2 + y_t**2 - R_detector_layer ** 2

#t = fsolve(func,0.5)[0]
#print(t)
# =============================================================================
# x, y, z = R(t)
# print(x,y,z)
# print(np.sqrt(x**2+y**2))
# =============================================================================


t_array = np.linspace(0,0.5, 100)
xyz = np.array([R(t) for t in t_array])
xyzT = np.transpose(xyz)
X = xyzT[0]
Y = xyzT[1]
Z = xyzT[2]
#print(xyz)

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot(X,Y,Z)

plt.show()



# =============================================================================
# x_list = []
# y_list = []
# z_list = []
# for t in t_array:
#     x_list.append(R(t)[0])
#     y_list.append(R(t)[1])
#     z_list.append(R(t)[2])
# =============================================================================
#xyz = np.transpose(np.array([R(t)[0] for x in t_array]))