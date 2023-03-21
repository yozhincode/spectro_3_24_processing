#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 19:16:37 2022

@author: eugene
"""

import fnmatch
import os
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from astropy.io import fits
# from matplotlib.colors import LogNorm
# import time as tm
# from matplotlib.image import NonUniformImage
# import datetime
import pylab as lab
from pathlib import Path

def hhmm_format(t,pos):
  hh = (int)(t / 3600.);
  t -= hh*3600.;
  mm = (int)(t / 60.);
  # t -= mm*60.;
  # ss = (int)(t / 1.);
  return '%02d:%02d' % (hh,mm);
  # return '%02d:%02d:%02d' % (hh,mm,ss);
dt_major = 3600.;
dt_minor = 600.;

home = str(Path.home())
dirpath = home+"/Data/Spectro_3_24G/2023/03/17/"
# filename = "spectro324_20221214T160516.fit"
# hdulist = fits.open(dirpath+filename)
dirlist = os.listdir(dirpath)
filelist = sorted(fnmatch.filter(dirlist, 'spectro324G_20230317T*.fit'))
print(len(filelist))
print(filelist)
i = 0;
channel = 42

for filename in filelist:
    hdulist = fits.open(dirpath+filename)
    # hdulist.info()
    if i == 0:
        dataR = hdulist[1].data['DataRCP']
        dataL = hdulist[1].data['DataLCP']
        time = hdulist[1].data['time']
        freq = hdulist[1].data['frequency']
        # channels = int(hdulist[0].header['CHANNELS'])
        date = hdulist[0].header['DATE-OBS']
        # hh   = float(hdulist[0].header['TIME-OBS'].split(":")[0]) 
        # mm   = float(hdulist[0].header['TIME-OBS'].split(":")[1])
        # ss   = float(hdulist[0].header['TIME-OBS'].split(":")[2]) 
    else:
        dataR = np.concatenate([dataR, hdulist[1].data['DataRCP']])
        dataL = np.concatenate([dataL, hdulist[1].data['DataLCP']])
        time = np.concatenate([time, hdulist[1].data['time']])
        # print(data.shape)
    print(filename)
    i = i + 1
    hdulist.close()
    # if i == 3:
    #     break

print (dataR.shape)
print (time[:,0].shape)
time_axis = np.array(time[:,0])
timenum = len(time_axis)


spltarr = np.hsplit(dataR,3)
freq = freq.T
print (freq[channel,])
chfreq = freq[channel,]/1000.0
print (chfreq)

data36 = spltarr[0]
data612 = spltarr[1]
data1224 = spltarr[2]
min36 = np.min(data36)
min612 = np.min(data612)
min1224 = np.min(data1224)

plt.figure(figsize = (15,5))
# plt.ylim(4.0e-12, 5.0e-12)
line, = plt.plot(time_axis, dataR[:,channel])
# line.set_label('3.0GHz')
# line, = plt.plot(time_axis, dataR[:,channel+1])
# line.set_label('3.2GHz')
# line, = plt.plot(time_axis, dataL[:,channel])
# line, = plt.plot(time_axis, dataR[:,channel]-dataL[:,channel])

ax = plt.gca()
locator = mdates.AutoDateLocator()
# formatter = mdates.AutoDateFormatter(locator)
ax.xaxis.set_major_locator(lab.MultipleLocator(dt_major));
ax.xaxis.set_major_formatter(lab.FuncFormatter(hhmm_format));
ax.xaxis.set_minor_locator(lab.MultipleLocator(dt_minor));
# ax.xaxis.set_minor_formatter(lab.FuncFormatter(hhmm_format));
plt.xlim(time_axis[0],time_axis[timenum-1])
plt.xlabel("UT Time, hours",fontsize=12)
plt.title(date + "   Badary SMDD 3-24 GHz" + ",    F, MHz ="+str("%.2i" % chfreq), fontsize=12)
# plt.legend(loc='lower right',fontsize=10)
plt.show()


