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
  return '%02d:%02d' % (hh,mm);
dt_major = 3600.;
dt_minor = 600.;

home = str(Path.home())
dirpath = home+"/Data/Spectro_3_24G/2023/02/26/"
dirlist = os.listdir(dirpath)
filelist = sorted(fnmatch.filter(dirlist, 'spectro324G_20230226T*.fit'))
print(len(filelist))
print(filelist)
i = 0;

for filename in filelist:
    hdulist = fits.open(dirpath+filename)
    # hdulist.info()
    if i == 0:
        datarcp = hdulist[1].data['DataRCP']
        datalcp = hdulist[1].data['DataLCP']
        time = hdulist[1].data['time']
        freq = hdulist[1].data['frequency']
        # channels = int(hdulist[0].header['CHANNELS'])
        date = hdulist[0].header['DATE-OBS']
        # hh   = float(hdulist[0].header['TIME-OBS'].split(":")[0]) 
        # mm   = float(hdulist[0].header['TIME-OBS'].split(":")[1])
        # ss   = float(hdulist[0].header['TIME-OBS'].split(":")[2]) 
    else:
        datarcp = np.concatenate([datarcp, hdulist[1].data['DataRCP']])
        datalcp = np.concatenate([datalcp, hdulist[1].data['DataLCP']])
        time = np.concatenate([time, hdulist[1].data['time']])
        # print(data.shape)
    print(filename)
    i = i + 1
    hdulist.close()
    # if i == 3:
    #     break
print (datarcp.shape)
print (time.shape)
print (time)
time_axis = time[:,0]
timenum = len(time)
print (timenum)
print(hhmm_format(time_axis[0],0))
print(hhmm_format(time_axis[timenum-1],0))

# timereshaped = np.reshape(time, 16000)
# with np.printoptions(threshold=np.inf):
#     print (time.shape)
#     print (timereshaped.shape)
#     print (["%0.3f" % i for i in timereshaped])
# hdulist.info()
print (datarcp.shape)
print (datalcp.shape)
# data36 = data[:,0]
# data612 = data[:,16:31]
# data1224 = data[:,32:47]
# print (data36.shape)
# data_repack = np.concatenate((data36, data612, data1224), axis=1)
# print (data_repack.shape)

avgFrcp = np.zeros(48)
avgFlcp = np.zeros(48)

# plt.figure(figsize = (15,5))

for j in range(48):
    dataFrcp = datarcp[:,j]
    dataFlcp = datalcp[:,j]
    avgFrcp[j] = np.average(dataFrcp)
    avgFlcp[j] = np.average(dataFlcp)
    datarcp[:,j] = datarcp[:,j]/avgFrcp[j]
    datalcp[:,j] = datalcp[:,j]/avgFlcp[j]
    # line, = plt.plot(time_axis, dataF)
    # ax = plt.gca()
    
    
# locator = mdates.AutoDateLocator()
# # formatter = mdates.AutoDateFormatter(locator)
# # ax.xaxis.set_major_locator(locator)
# # ax.xaxis.set_major_formatter(formatter)
# ax.xaxis.set_major_locator(lab.MultipleLocator(dt_major));
# ax.xaxis.set_major_formatter(lab.FuncFormatter(hhmm_format));
# ax.xaxis.set_minor_locator(lab.MultipleLocator(dt_minor));
    
plt.figure(figsize = (15,7))

plt.imshow(datarcp.T,
            cmap="rainbow", 
            aspect = "auto",
            origin='lower',
            extent=[time_axis[0], time_axis[timenum-1], 0.0, 47.0],
            interpolation='bilinear',
            )

ax = plt.gca()
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(lab.MultipleLocator(dt_major));
ax.xaxis.set_major_formatter(lab.FuncFormatter(hhmm_format));
ax.xaxis.set_minor_locator(lab.MultipleLocator(dt_minor));
ax.xaxis.set_minor_formatter(lab.FuncFormatter(hhmm_format));
y = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.0, 6.4, 6.8, 7.2, 7.6, 8.0, 8.4, 8.8, 9.2, 9.6, 10.0, 10.4, 10.8, 11.2, 11.6, 12.0, 12.0, 12.8, 13.6, 14.4, 15.2, 16.0, 16.8, 17.6, 18.4, 19.2, 20.0, 20.8, 21.6, 22.4, 23.2, 24.0]
plt.yticks(range(len(y)),y)
plt.xlabel("UT Time, hours",fontsize=10)
plt.title(date + "   3-24GHZ RCP", fontsize=12)
plt.show()

plt.figure(figsize = (15,7))

plt.imshow(datalcp.T,
            cmap="rainbow", 
            aspect = "auto",
            origin='lower',
            extent=[time_axis[0], time_axis[timenum-1], 0.0, 47.0],
            interpolation='bilinear',
            )
ax = plt.gca()
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(lab.MultipleLocator(dt_major));
ax.xaxis.set_major_formatter(lab.FuncFormatter(hhmm_format));
ax.xaxis.set_minor_locator(lab.MultipleLocator(dt_minor));
ax.xaxis.set_minor_formatter(lab.FuncFormatter(hhmm_format));
plt.xlabel("UT Time, hours",fontsize=10)
plt.yticks(range(len(y)),y)
plt.title(date + "   3-24GHZ LCP", fontsize=12)
plt.show()
