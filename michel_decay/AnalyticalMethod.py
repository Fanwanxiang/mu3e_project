# -*- coding: utf-8 -*-
""" 
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
s = np.random.uniform(0,1,10000) 
''' generate uniform distribution sample, 
    s is a array which is the set of the data I generate in this project'''
s_theta = np.arccos(1-2*s)
#hist, bin_edges  = np.histogram(s, bins= 20)

'''np.all(s >= 0)
True
np.all(s < 1)
True
count, bins, ignored = plt.hist(s, 15, normed=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()'''
np.all(s_theta >= 0)
True
np.all(s_theta <= np.pi)
True
count, bins, ignored = plt.hist(s_theta, bins= np.linspace(0,np.pi,20), normed=True) # draw the histagram of s_theta
#print(bins) 
plt.plot(bins, 0.5*np.sin(bins), linewidth=2, color='r')
plt.show()