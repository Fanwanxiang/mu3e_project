#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:35:33 2020

@author: wanxiangfan
"""
from phasespace import GenParticle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()
def plot_hist_1D(sample, xlabel, ylabel, title):
    count, bins, ignored = plt.hist(sample, bins="auto", density=False)  # draw the histagram of l
    # print(bins)
    # plt.plot(bins, A * bins, linewidth=2, color='r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()



Muon_MASS = 0.105658367*1000 #MeV/c2
Electron_MASS = 0.000510998910*1000  #MeV/c2
Positron_MASS = 0.000510998910*1000 #

#Antimuon = GenParticle('mu+', Muon_MASS)
Electron = GenParticle('e-', Electron_MASS)
Positron1 = GenParticle('e+1', Positron_MASS)
Positron2 = GenParticle('e+2', Positron_MASS)
ElectronNeutrino = GenParticle('nu_e', 0)
AntiMuonNeutrino = GenParticle('nu_muon_bar', 0)
bz = GenParticle('mu+', Muon_MASS).set_children(Electron, Positron1,Positron2,
                ElectronNeutrino,AntiMuonNeutrino)

weights, particles = bz.generate(n_events=10**5)    # 10**5)
# this is the total 4-momentum of e- e+ e+
#print(particles['e-'].numpy())
E_electron = particles['e-'].numpy()
np.savetxt("electron4momentum.txt", E_electron)
E_positron1 = particles['e+1'].numpy()
E_positron2 = particles['e+2'].numpy() 
E_ElectronNeutrino =particles['nu_e'].numpy()
E_AntiMuonNeutrino =particles['nu_muon_bar'].numpy()
E_tot = E_electron +E_positron1 + E_positron2# + E_ElectronNeutrino + E_AntiMuonNeutrino

y=E_tot[:,3]/Muon_MASS # last entry is the tot energy 
plot_hist_1D(y,xlabel="y",ylabel="dR/dy",title="Total enegy spectrum")
plt.show()
Xe= E_electron[:,3]/Muon_MASS
plot_hist_1D(Xe,xlabel="Xe",ylabel="dR/dXe",title="electron spectrum")
plt.show()
Xp = np.concatenate((E_positron1[:,3],E_positron2[:,3]))# E_positron1[:,3] + E_positron2[:,3]
plot_hist_1D(Xp,xlabel="Xp",ylabel="dR/dXp",title="positron spectrum")
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))