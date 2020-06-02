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
B_z = 1     # T , B_z: 1 - 1.5 T ; SI : kg⋅s^(−2)⋅A^(−1)

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


def position(t,x, P_mu, R): # return the position after t
    q = -e # charge of electron    
    #trans MeV --> SI unit
    E0, px0, py0, pz0 = P_mu[0] * f_en, P_mu[1] * f_mom, P_mu[2] * f_mom, P_mu[3] * f_mom 
    x0, y0, z0 = R[0], R[1], R[2]
    omega =q*B_z*c*c/E0 # angular frequency of circular motion in xy-plane
    x_t = 1/(q*B_z) * (px0 * np.sin(omega * t) + py0 * (1 - np.cos(omega * t))) + x0
# =============================================================================
#     y_t = 1/(q*B_z) * (py0 * np.sin(omega * t) - px0 * (1 - np.cos(omega * t))) + y0
#     z_t = (c*c/E0) * pz0 * t + z0
# =============================================================================
    return x_t - x #, y_t, z_t
# =============================================================================
# 
# def x_t(t,  x, E0, px0, py0 , x0): # return the position after t 
#     # x is the position of the intersections of two circles
#     q = -e # charge of electron   
#     #trans MeV --> SI unit
#     px0 = px0 * f_mom
#     py0 = py0 * f_mom
#     E0 = E0 * f_en    
# # =============================================================================
# #     E0, px0, py0, pz0 = P_mu[0] * f_en, P_mu[1] * f_mom, P_mu[2] * f_mom, P_mu[3] * f_mom 
# #     x0, y0, z0 = R[0], R[1], R[2]
# # =============================================================================
#     omega =q*B_z*c*c/E0 # angular frequency of circular motion in xy-plane
#     x_t = 1/(q*B_z) * (px0 * np.sin(omega * t) + py0 * (1 - np.cos(omega * t))) + x0
# # =============================================================================
# #     y_t = 1/(q*B_z) * (py0 * np.sin(omega * t) - px0 * (1 - np.cos(omega * t))) + y0
# #     z_t = (c*c/E0) * pz0 * t + z0
# # =============================================================================
#     return x_t - x #, y_t, z_t
# =============================================================================


#Finding the intersection of two circles
#https://stackoverflow.com/questions/55816902/finding-the-intersection-of-two-circles
def get_intersections(a, b, c, d, px, py): #
    # x1, y1 are the position of coincidence , r1 is the radius of the track
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1
    #px_SI = px*f_mom # transfer to 
    #py_SI = py*f_mom
    
    pxy = np.sqrt(px**2 + py**2)
    
    #radius of the detector layer
    r0 = 60/1000
    # radius of the positron/electron track
    r1 = pxy / (e*B_z)
    #print("r1=", r1)
    r0 = 60/1000  # metre circle of detector layer

    # non intersecting
    if d > r0 + r1 :
        print('d > r0 + r1, no intersections')
        return 0,0,0,0
    # One circle within other
    if d < abs(r0-r1):
        print('d < abs(r0-r1), no intersections')
        return 0,0,0,0
    # coincident circles
    if d == 0 and r0 == r1:
        return 0,0,0,0
    else:
        #Distance between two circles centers:  
        D = np.sqrt( (c-a)**2 + (d-b)**2 )
        # area is the area of the triangle formed by the two circle centers and one of
        #the intersection point. The sides of this triangle are S, r0 and r1 , the 
        #area is calculated by Heron' s formula.
        area = 0.25 * np.sqrt( (D + r0 + r1) * (D + r0 - r1) * (D - r0 + r1) * (- D + r0 + r1) )
        # Two circles intersection points:
        x1 = 0.5 * (a + c) + (c - a)*(r0**2 - r1**2) / (2*D**2) + 2*(b - d)*area/D**2
        x2 = 0.5 * (a + c) + (c - a)*(r0**2 - r1**2) / (2*D**2) - 2*(b - d)*area/D**2
        y1 = 0.5 * (b + d) + (d - b)*(r0**2 - r1**2) / (2*D**2) - 2*(a - c)*area/D**2
        y2 = 0.5 * (b + d) + (d - b)*(r0**2 - r1**2) / (2*D**2) + 2*(a - c)*area/D**2
        return x1, y1, x2, y2   
    

def time(xf, xi, pxi, pyi, Ei): 
    # find the time interval from initial position to final position
    # 
# =============================================================================
#     pxi = pxi * f_mom
#     pyi = pyi * f_mom
#     Ei = Ei * f_en
# =============================================================================
    q = e
    pxyi = np.sqrt(pxi**2 + pyi**2) #
    omega = q*B_z*c*c/Ei
    alpha = np.arctan(-pyi/pxi)
    a = (xf - xi) * q * B_z - pyi
    print("a/pxyi = ", a/pxyi)
    b = np.arccos(a/pxyi)
    t = (b + alpha) / omega
    return t
# =============================================================================
# x3, y3, x4, y4 = get_intersections(a = 0, b = 0, c = x0 ,d = 0, px = px0, py = py0) 
# print(x3, y3, x4, y4)
# 
# t = time(x3, x0, px0, py0, E0)
# print(t)
# =============================================================================
# print()
initial_position = np.loadtxt("position.txt")
print(initial_position[0])

initial_4momentum = np.loadtxt("electron4momentum.txt")
print(initial_4momentum[0])
t_list=[]
for i in range(10**5):
    x0 = initial_position[i][0]/1000
    y0 = initial_position[i][1]/1000
    px0 = initial_4momentum[i][0] * f_mom
    py0 = initial_4momentum[i][1] * f_mom
    E0 = initial_4momentum[i][3] * f_en
    x1, y1, x2, y2 = get_intersections(a = 0, b = 0, c = x0 ,d = y0, px = px0, py = py0)
    if (x1==0) and (y1==0) and (x2==0) and (y2==0):
        print('no intersections')
    else:
        t = time(x1,x0,px0,py0,E0)
        t_list.append(t)
        
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