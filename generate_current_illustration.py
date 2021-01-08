#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 16:30:46 2020

@author: marius
"""

import numpy as np
import matplotlib.pyplot as plt
burst = [0,1,1,-1,-1,0,0]*5
burstx = [0,0,0.1,0,0.1,0,38.8]*5

plt.plot(np.cumsum(burstx),burst)


y = [*burst,0,*burst,0,*burst,0]
x = np.cumsum([*burstx,805, *burstx,805,*burstx,400])
plt.plot(x,y, 'k')
plt.yticks([-1,1])
plt.ylabel("Current [mA]")
plt.xlabel("Time [ms]")
plt.savefig("/mnt/data/Studies/tVNS_HRV/Figures/currentexample.svg")(bvdata)