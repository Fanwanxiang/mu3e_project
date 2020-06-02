#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:32:20 2020

@author: wanxiangfan
"""
import subprocess
import time
import sys
import numpy as np

start_time = time.time()


def run_cmd(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    process.communicate()
    # stdout = process.communicate()[0]
    # print ('STDOUT:{}'.format(stdout))
    process.stdout.close()
    return " "


#Factor1 = np.arange(0.1, 0.2, 0.5)
#Factor2 = np.arange(0.5, 5, 0.5)
#Factor = np.concatenate([Factor1, Factor2])
Factor = np.arange(0.5, 5.5, 0.5)
DX = np.array([0.2])
for dx in DX:
    print("\ndx = %.3f mm " % dx)
    for factor in Factor:
        print("\nfactor = ", factor)
        for i in range(int(50000/factor)):
            sys.stdout.write("\ri=%d" % (i))
            sys.stdout.flush()
            run_cmd("python Simulation_factor_dx.py %.3f %.3f" % (factor, dx))
print("--- %s seconds ---" % (time.time() - start_time))
