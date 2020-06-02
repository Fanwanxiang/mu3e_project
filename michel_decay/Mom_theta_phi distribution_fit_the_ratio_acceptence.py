#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:58:43 2019

@author: wanxiangfan
"""
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize    # to define a universal function for array
import matplotlib.ticker as mticker
# 2.) Define fit function.
def polynomial(x, a): # a is the P_max in this case
    return 6/a**3 * x**2 - 4/a**4 * x**3 


def fit_ploynomial(sample,count, bins, fit_function, xlabel, ylabel, title, fname):
    #Add histograms of data.
    #data_entries = count
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    # 5.) Fit the function to the histogram data.
    popt, pcov = curve_fit(fit_function, xdata=binscenters, ydata=count, p0=0.15)
    perr = np.sqrt(np.diag(pcov))

    # 6.)
    # Generate enough x values to make the curves look smooth.
    #xspace = np.linspace(0, 52.8 , 10000)
    
    # Plot the histogram and the fitted function.
    #plt.hist(sample, bins="auto", density=True, histtype = 'step')
    #plt.step(binscenters, data_entries)
    #plt.bar(binscenters, data_entries, width=bins[1] - bins[0], color='navy', label=r'Histogram entries')
    #plt.plot(xspace, fit_function(xspace, *popt), color='darkorange', linewidth=1.5, label=r'Fitted function')
    # Make the plot nicer.
    #plt.xlim(0,6)
    #plt.ylim(0, 0.32)
# =============================================================================
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
# =============================================================================
    #plt.savefig(fname= fname,dpi=600)
    #plt.show()
    #plt.clf()

    a = popt[0]
    err_a =perr[0]
    #print("[a] = ",a)
    #print("[err_a] = ",err_a)
    return a, err_a # return the P_max and its error


#def a func that calculate the average fitted data: Pmax provided No of samples
# and no of simulations

#N = 10**7 # 10**7                    # sample number
electron_Mass = 0.5109989461   # MeV , rest mass of electrons
@vectorize                     # to define a universal function for array
def p(x):  
    rho = 3/4                      # rho is constant and =3/4 in SM                    # rho = 3/4
    p=4*x**2*(3*(1-x)+2*rho*(4/3*x-1))
    return p
 # sample number 
def Accept_Reject_Mom(Nsmaples):
    N=Nsmaples
    momMax = 52.82                  # max monmentum of electrons
    mom_e = np.random.uniform(0,momMax,N) # mon_e momentum of electron
    x = mom_e/momMax
    P_mom = p(x)/momMax# Pdf of electron
    y = np.random.uniform(0,0.04,N)
    mom_accept = mom_e[P_mom > y]
    Nmom_accept = len(mom_accept)
    ratio_accept_smaples = Nmom_accept/Nsmaples
    count, bins, ignored = plt.hist(mom_accept, bins= 40, density=True,histtype = 'step') # draw the histagram of s_theta
    ##print(bins) 
    P_max, err_P_max = fit_ploynomial(mom_accept,count, bins, polynomial,'Momentum of positrons (MeV)','PDF', 'Momentum spectrum of Michel decay', 'fname')
    
    
# =============================================================================
#     #inverse sampling method to generate the samples of theta
#     N_mom_accept = len(mom_accept) # number of momentum samples 
#     s = np.random.uniform(0,1,N_mom_accept) 
#     s_theta = np.arccos(1-2*s)
#     # generate the samples of phi
#     phi = np.random.uniform(0,2*np.pi,N_mom_accept) 
#     
#     px = mom_accept*np.sin(s_theta)*np.cos(phi)
#     py = mom_accept*np.sin(s_theta)*np.sin(phi)
#     pz = mom_accept*np.cos(s_theta)
# =============================================================================
    return P_max, err_P_max, Nmom_accept, ratio_accept_smaples  #px, py, pz,
    
# Nsample = the number of samples
# Nsim = the No of simulation repeated
def cal_Pmaxbar_for_sampleNO(Nsample, Nsim):   
    P_max_list = []
    Nmom_accept_list = []
    ratio_acceptence = []
    for i in range(Nsim):
        P_max, err_P_max,   Nmom_accept, ratio_accept_smaples = Accept_Reject_Mom(Nsample)
        P_max_list.append(P_max)
        Nmom_accept_list.append(Nmom_accept)
        ratio_acceptence.append(ratio_accept_smaples)
    P_max,Err_P_max = np.mean(P_max_list), np.std(P_max_list)/np.sqrt(Nsim)
    N_mon_accept, Err_N_mon_accept = np.mean(Nmom_accept_list), np.std(Nmom_accept_list)/np.sqrt(Nsim)
    N_ratio_accept, Err_ratio_accept = np.mean(ratio_acceptence), np.std(ratio_acceptence)/np.sqrt(Nsim)
    return P_max,Err_P_max, N_mon_accept, Err_N_mon_accept, N_ratio_accept, Err_ratio_accept

def fit_P_max(x,y0,err,func,xlabel,ylabel,title,imagename):
    # curve fit [with only y-error]
    def Pmax_T(x):
        return x*0 + 1/(52.82*0.04)
    y = y0 
    popt, pcov = curve_fit(func, x, y, sigma=err,p0=52.8)#, sigma=1./(noise*noise))
    perr = np.sqrt(np.diag(pcov))
    
    ##print fit parameters and 1-sigma estimates
    #print('fit parameter 1-sigma error')
    #print('———————————–')
    #for i in range(len(popt)):
        ##print(str(popt[i])+ ' +- ' +str(perr[i]))
    
    # prepare confidence level curves
    nstd = 5. # to draw 5-sigma intervals
    popt_up = popt + nstd * perr
    popt_dw = popt - nstd * perr
    
    theo_value = Pmax_T(x)
    fit = func(x, *popt)
    fit_up = func(x, *popt_up)
    fit_dw = func(x, *popt_dw)
    
    #plot
    fig, ax = plt.subplots(1)
    plt.errorbar(x, y0, yerr=noise, xerr=0,fmt='ko', label='data',lw=1,ms =1)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=12)
    plt.plot(x,theo_value, 'k--', lw=1, label='Theorerical ratio')
    plt.plot(x, fit, 'r', lw=1, label='best fit curve')
    plt.fill_between(x, fit_up, fit_dw, alpha=.25, label='5-sigma interval')
    plt.legend(fontsize=12,loc='best')
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0),useMathText= True )
    plt.xscale("log")
    plt.savefig(fname = imagename, dpi=600)
    plt.tight_layout()
    plt.show()
    
    
    
    # curve fit [with only y-error]
    popt, pcov = curve_fit(func, x, y)
    Pmax=popt[0]
    Pmax_err= perr[0]
    return Pmax,Pmax_err


imagename = 'Fit_Ratio_104_107' 
a1=np.arange(10**4,10**5,int(0.5*10**4))
a2=np.arange(10**5,10**6,int(0.5*10**5))
a3=np.arange(10**6,10**7,int(0.5*10**6))
sample_size_l =np.concatenate((a1,a2,a3),axis = None)#, np.arange(10**3,2*10**3,10**2)
L =[]
for i in sample_size_l:
    P_max,Err_P_max, N_mon_accept, Err_N_mon_accept, N_ratio_accept, Err_ratio_accept = cal_Pmaxbar_for_sampleNO(Nsample = i, Nsim=100)
    L.append([P_max, Err_P_max, N_mon_accept, Err_N_mon_accept, N_ratio_accept, Err_ratio_accept])
#print(L)

def func(x, a):
    return 0*x + a

# test data and error
#x_fitted_data  =  sample_size_l
#z_fitted_data  =  L
x = sample_size_l
y0 = np.transpose(L)[4] # Pmax
noise = np.transpose(L)[5] # Pmaxerr

N_ratio_accept, Err_ratio_accept = fit_P_max(x,y0,noise,func,'Sample size','Acceptance rate','Fit acceptance Rate ',imagename)
print(N_ratio_accept, Err_ratio_accept)
# =============================================================================
# #%%
# N_mom_accept = len(mom_accept) # number of momentum samples 
# s = np.random.uniform(0,1,N_mom_accept) 
# s_theta = np.arccos(1-2*s)
# 
# 
# 
# count, bins, ignored = plt.hist(s_theta, bins= 'auto', normed=True) # draw the histagram of s_theta
# ##print(bins) 
# plt.plot(bins, 0.5*np.sin(bins), linewidth=2, color='r')
# plt.xlabel('Theta')
# plt.ylabel('Pdf') 
# plt.title('Theta Angle spectrum of Michel muon decay')
# plt.show()
# 
# #===============================================================================
# #%%
# phi = np.random.uniform(0,2*np.pi,N_mom_accept) 
# count, bins, ignored = plt.hist(phi, bins= 'auto', normed=True) # draw the histagram of s_theta
# plt.xlabel('phi')
# plt.ylabel('Pdf') 
# plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
# plt.title('Phi Angle spectrum of Michel muon decay')
# plt.show()
# 
# #===============================================================================
# #%%
# '''Sphereical coordinates transform to the Cartisian coordiantes
# '''
# px = mom_accept*np.sin(s_theta)*np.cos(phi)
# count, bins, ignored = plt.hist(px, bins= 'auto', normed=True)
# plt.xlabel('momentum in x direction')
# plt.ylabel('Pdf') 
# #plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
# plt.title('Momentum spectrum of Michel muon decay in x-direction')
# plt.show()
# 
# #%%
# py = mom_accept*np.sin(s_theta)*np.sin(phi)
# count, bins, ignored = plt.hist(px, bins= 'auto', normed=True)
# plt.xlabel('momentum in y direction')
# plt.ylabel('Pdf') 
# #plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
# plt.title('Momentum spectrum of Michel muon decay in y-direction')
# plt.show()
# 
# #%%
# pz = mom_accept*np.cos(s_theta)
# count, bins, ignored = plt.hist(pz, bins= 'auto', normed=True)
# plt.xlabel('momentum in z direction')
# plt.ylabel('Pdf') 
# #plt.plot(bins, np.ones_like(bins)/(2*np.pi), linewidth=2, color='r')
# plt.title('Momentum spectrum of Michel muon decay in z-direction')
# plt.show()
# 
# # calculate energy of the electron using relativistic dispersion relation
# eletron_energy = np.sqrt(mom_accept**2 + electron_Mass**2)
# 
# list1= np.array([px,py,pz,eletron_energy])
# mometum_energy_list = np.transpose(list1)
# np.savetxt("michel_decay_momentum_energy.txt", mometum_energy_list)
# 
# =============================================================================
