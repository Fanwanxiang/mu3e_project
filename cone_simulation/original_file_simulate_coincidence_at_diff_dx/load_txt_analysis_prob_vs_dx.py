#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 20:09:19 2020

@author: wanxiangfan
"""

import numpy as np#import statsmodels as sm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#dx005mm=np.loadtxt("dx=0.05mm.txt") 
R=10**7
dx02mm=np.loadtxt("dx=0.2mm.txt")
dx03mm=np.loadtxt("dx=0.3mm.txt")
dx04mm=np.loadtxt("dx=0.4mm.txt")
dx05mm=np.loadtxt("dx=0.5mm.txt")
#print("0.05 :",sum(dx005mm))
#print(dx05mm)
print("0.2 :",sum(dx02mm))
print("0.3 :",sum(dx03mm))
print("0.4 :",sum(dx04mm))
print("0.5 :",sum(dx05mm))
#import statsmodels
#print(statsmodels.__version__)  
def polynomial1(x,a,b):
    return a*x**b
def ploynomial2(x,a):
    return a*x**4 
 
def fit_polynomial1(x,y,yerr): #fit N_3e -- dx space resolution
    #xerr = np.random.random_sample(10)
    fig, ax = plt.subplots()
    
    ax.errorbar(x, y,
                yerr=yerr,
                fmt='-o')
    
    ax.set_xlabel('dx (mm)')
    ax.set_ylabel('prob')
    ax.set_title('prob--space resolution')
    plt.show()
    
    plt.plot(x, y, 'b-', label='data')    
    popt, pcov = curve_fit(polynomial1, x, y)
    print(popt)
    plt.plot(x, polynomial1(x, *popt), 'r-',
              label='fit: a=%f ,b=%f' % tuple(popt))#,b=%5.3f, c=%5.3f, d=%5.3f
    plt.xlabel('dx (mm)')
    plt.ylabel('prob')
    plt.legend()
    plt.show()
def fit_polynomial2(x,y,yerr): #fit N_3e -- dx space resolution
    #xerr = np.random.random_sample(10)
    fig, ax = plt.subplots()
    
    ax.errorbar(x, y,
                yerr=yerr,
                fmt='-o')
    
    ax.set_xlabel('dx (mm)')
    ax.set_ylabel('prob')
    ax.set_title('prob--space resolution')
    plt.show()
    
    plt.plot(x, y, 'b-', label='data')    
    popt, pcov = curve_fit(ploynomial2, x, y)
    print(popt)
    plt.plot(x, ploynomial2(x, *popt), 'r-',
              label='fit: a=%f' % tuple(popt))#,b=%5.3f, c=%5.3f, d=%5.3f
    plt.xlabel('dx (mm)')
    plt.ylabel('prob')
    plt.legend()
    plt.show()
    

def CI(x): # confidence interval
    count = sum(x)
        #number of successes, can be pandas Series or DataFrame
    nobs= len(x)*R
        #total number of trials

        #alphafloat in (0, 1)   significance level, default 0.05

        #method{'normal', 'agresti_coull', 'beta', 'wilson', 'binom_test'}
        #default: 'normal' method to use for confidence interval, currently available methods :
    ci_low, ci_upp = sm.stats.proportion_confint(count, nobs, alpha=0.95, method='wilson')
    #p = sum(x)/len(x)/R
    p = (ci_low+ci_upp)/2
    p_err = abs(ci_low-ci_upp)/2
    return [p,p_err]#, ci_low, ci_upp
L = [dx02mm,dx03mm,dx04mm,dx05mm]
p_with_err= np.array([CI(L[i]) for i in range(len(L))])
print(np.transpose(p_with_err)[0])
#for i in L:  
    #p, p_err = CI(i)
  #  print('p', 'ci_low', 'ci_upp')
   # print(p, ci_low, ci_upp)
    #x=np.array([0.2,0.3,0.4,0.5])
    #y=np.array([bernoulli(dx02mm),bernoulli(dx03mm),bernoulli(dx04mm),bernoulli(dx05mm)])
    #print (y)



# test data and error
x = np.array([0.2,0.3,0.4,0.5])
y,y_err = np.array(np.transpose(p_with_err)[0]),np.array(np.transpose(p_with_err)[1])
#fit_polynomial1(x=x,y=y,yerr=y_err)
fit_polynomial1(x=x,y=y,yerr=y_err)