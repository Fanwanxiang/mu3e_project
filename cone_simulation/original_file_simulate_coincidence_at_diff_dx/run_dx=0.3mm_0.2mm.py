#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:32:20 2020

@author: wanxiangfan
"""
import subprocess
import time
import sys
start_time = time.time()

def run_cmd(cmd):
  process = subprocess.Popen(cmd.split(), stdout = subprocess.PIPE)
  process.communicate()
  #stdout = process.communicate()[0]
  #print ('STDOUT:{}'.format(stdout))
  process.stdout.close()
  return " "
#for i in range(10000):
    #print('i =',i)
    #run_cmd("python Feb_25_dx=0.3mm.py")
for j in range(20000):
    print('j =',j)
    run_cmd("python Feb_25_dx=0.2mm.py")
print("--- %s seconds ---" % (time.time() - start_time))
