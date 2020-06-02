#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:32:20 2020

@author: wanxiangfan
"""
import subprocess
import time
start_time = time.time()

def run_cmd(cmd):
  process = subprocess.Popen(cmd.split(), stdout = subprocess.PIPE)
  process.communicate()
  #stdout = process.communicate()[0]
  #print ('STDOUT:{}'.format(stdout))
  process.stdout.close()
  return " "
for i in range(5000):
    print('i =',i)
    run_cmd("python Feb_23_double_cones_simulation_single_3_electrons_at_dt.py")
print("--- %s seconds ---" % (time.time() - start_time))
