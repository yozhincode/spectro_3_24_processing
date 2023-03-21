#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 19:16:37 2022

@author: eugene
"""

import fnmatch
import os
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from astropy.io import fits
from datetime import date
from datetime import timedelta
import pylab as lab
from pathlib import Path

def hhmm_format(t,pos):
  hh = (int)(t / 3600.);
  t -= hh*3600.;
  mm = (int)(t / 60.);
  return '%02d:%02d' % (hh,mm);
dt_major = 3600.;
dt_minor = 600.;

channel = 47

yesterday = timedelta(days = 1)
today = date.today() - yesterday
home = str(Path.home())
dirpath = home+"/Data/Spectro_3_24G/"+today.strftime("%Y/%m/%d/")
pngfilename = home+"/Data/Spectro_3_24G/Processed/site_pics/"+'bssd324_'+today.strftime("%Y_%m_%d")+'.png'
print(pngfilename)
# filename = "spectro324_20221214T160516.fit"
# hdulist = fits.open(dirpath+filename)
dirlist = os.listdir(dirpath)

filelist = sorted(fnmatch.filter(dirlist, 'spectro324G_*.fit'))
print(len(filelist))
print(filelist)
i = 0;

for filename in filelist:
    hdulist = fits.open(dirpath+filename)
    # hdulist.info()
    if i == 0:
        dataR = hdulist[1].data['DataRCP']
        dataL = hdulist[1].data['DataLCP']
        time = hdulist[1].data['timeRCP']
        freq = hdulist[1].data['frequency']
        # channels = int(hdulist[0].header['CHANNELS'])
        date = hdulist[0].header['DATE-OBS']
        # hh   = float(hdulist[0].header['TIME-OBS'].split(":")[0]) 
        # mm   = float(hdulist[0].header['TIME-OBS'].split(":")[1])
        # ss   = float(hdulist[0].header['TIME-OBS'].split(":")[2]) 
    else:
        dataR = np.concatenate([dataR, hdulist[1].data['DataRCP']])
        dataL = np.concatenate([dataL, hdulist[1].data['DataLCP']])
        time = np.concatenate([time, hdulist[1].data['timeRCP']])
        # print(data.shape)
    print(filename)
    i = i + 1
    hdulist.close()
    # if i == 3:
    #     break

# print (dataR.shape)
# print (time[:,0].shape)
time_axis = np.array(time[:,0])
timenum = len(time_axis)

freq = freq.T
# chfreq = freq[channel,]/1000.0

coeff = np.zeros(48)
sefd = np.zeros(48)
coeff,sefd = np.loadtxt('2023_03_17_calibration.csv', delimiter=',')
print(coeff)
print(sefd)

plt.ioff()
# plt.figure(figsize = (16,6))
# saveFreqs = np.array([0,5,10,15,21,26,31,37,42,47])
saveFreqs = np.array([0,5,10,15,21,26,31])
lines_number = saveFreqs.shape[0];
# print(lines_number)
c_list = matplotlib.colors.LinearSegmentedColormap.from_list(lab.cm.datad['gist_rainbow'], colors=['r','g','b'], N = lines_number);

fig = lab.figure(figsize = (16,8));
sub = fig.add_subplot(1,1,1);
for linenum in range(saveFreqs.shape[0]):
    procfreq = saveFreqs[linenum]
    # plt.ylim(4.0e-12, 5.0e-12)
    sub.scatter(time_axis, (0.5*dataR[:,procfreq]+0.5*dataL[:,procfreq])*coeff[procfreq]-sefd[procfreq],color=c_list(linenum),edgecolors='none',marker='.', s=1.0, label='%d MHz'%(freq[procfreq]/1000.0))
    sub.scatter(time_axis, (0.5*dataR[:,procfreq]-0.5*dataL[:,procfreq])*coeff[procfreq],color=c_list(linenum),edgecolors='none',marker='.', s=1.0)
    sub.legend(markerscale=10)
    # line = plt.scatter(time_axis, dataR[:,linenum*2],color=c_list(linenum),edgecolors='none',marker='.', s=0.5)
    # chfreq = freq[linenum*3,]/1000.0
    # line.set_label("%.2i" % chfreq)
    # line, = plt.plot(time_axis, dataR[:,channel+1])
    # line.set_label('3.2GHz')
    # line, = plt.plot(time_axis, dataL[:,channel])
    # line, = plt.plot(time_axis, dataR[:,channel]-dataL[:,channel])

# ax = plt.gca()
locator = mdates.AutoDateLocator()
# formatter = mdates.AutoDateFormatter(locator)
sub.set_xlim(-1200,37200)
sub.xaxis.set_major_locator(lab.MultipleLocator(dt_major));
sub.xaxis.set_major_formatter(lab.FuncFormatter(hhmm_format));
sub.xaxis.set_minor_locator(lab.MultipleLocator(dt_minor));
# ax.xaxis.set_minor_formatter(lab.FuncFormatter(hhmm_format));
sub.set_ylim(-1e2,1e3)
sub.set_xlabel("UT Time",fontsize=12)
sub.set_ylabel("Solar flux, sfu",fontsize=12)
fig.suptitle(date + "   Badary spectro polarimeter BSSD flux plot    Stokes I,V at 3-24 GHz", fontsize=12, y = 0.92)
# plt.legend(loc='upper right',fontsize=10)
# plt.show()
# plt.savefig('image.png', bbox_inches='tight',pad_inches = 0.01)
lab.savefig(pngfilename, bbox_inches='tight',pad_inches = 0.01)
lab.close()
plt.close()


