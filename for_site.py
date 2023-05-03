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
from matplotlib import cm
import matplotlib.dates as mdates
import numpy as np
from astropy.io import fits
from datetime import date
from datetime import timedelta
import pylab as lab
from pathlib import Path
import pandas as pd
import ftplib


def hhmm_format(t,pos):
  hh = (int)(t / 3600.);
  t -= hh*3600.;
  mm = (int)(t / 60.);
  return '%02d:%02d' % (hh,mm);
dt_major = 3600.;
dt_minor = 600.;

channel = 18

daysbefore = timedelta(days = 0)
today = date.today()-daysbefore
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
        nchannels = hdulist[0].header['CHANNELS']
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

print (nchannels)
# print (time[:,0].shape)
time_axis = np.array(time[:,0])
timenum = len(time_axis)

freq_list = freq[0:48]

coeff_R = np.zeros(48)
coeff_L = np.zeros(48)
sefd_R = np.zeros(48)
sefd_L = np.zeros(48)
coeff_R,coeff_L,sefd_R,sefd_L = np.loadtxt('2023_04_30_calibration.csv', delimiter=',')
# print(coeff)
# print(sefd)

plt.ioff()
# plt.figure(figsize = (16,6))
# saveFreqs = np.array([0,5,10,15,21,26,31,37,42,47])
saveFreqs = np.array([0,5,10,15,21,26,31])
# saveFreqs = np.array([0,2,4,6,8,10,12,14,17,19,21,23,25,27,29,31])
lines_number = saveFreqs.shape[0];
# print(lines_number)
c_list = matplotlib.colors.LinearSegmentedColormap.from_list(lab.cm.datad['gist_rainbow'], colors=['r','g','b'], N = lines_number);

# fig = lab.figure(figsize = (16,8));
# sub = fig.add_subplot(1,1,1);
# for linenum in range(saveFreqs.shape[0]):
#     procfreq = saveFreqs[linenum]
#     # plt.ylim(4.0e-12, 5.0e-12)
#     # print(coeff_R[procfreq],coeff_L[procfreq],sefd_R[procfreq],sefd_L[procfreq])
#     sub.scatter(time_axis, (dataR[:,procfreq]*coeff_R[procfreq]-sefd_R[procfreq])/2.0 + (dataL[:,procfreq]*coeff_L[procfreq]-sefd_L[procfreq])/2.0, color=c_list(linenum),edgecolors='none',marker='.', s=1.0, label='%d MHz'%(freq[procfreq]/1000.0))
#     sub.scatter(time_axis, (dataR[:,procfreq]*coeff_R[procfreq]-sefd_R[procfreq])/2.0 - (dataL[:,procfreq]*coeff_L[procfreq]-sefd_L[procfreq])/2.0, color=c_list(linenum),edgecolors='none',marker='.', s=1.0)
#     # sub.scatter(time_axis, (0.5*(dataR[:,procfreq]*coeff_R[procfreq]-sefd_R[procfreq])+0.5*(dataL[:,procfreq])*coeff_L[procfreq]-sefd_L[procfreq]),color=c_list(linenum),edgecolors='none',marker='.', s=1.0, label='%d MHz'%(freq[procfreq]/1000.0))
#     # sub.scatter(time_axis, (0.5*(dataR[:,procfreq]*coeff_R[procfreq]-sefd_R[procfreq])-0.5*(dataL[:,procfreq])*coeff_L[procfreq]-sefd_L[procfreq]),color=c_list(linenum),edgecolors='none',marker='.', s=1.0)
#     sub.legend(markerscale=10,loc='upper right')
#     # line = plt.scatter(time_axis, dataR[:,linenum*2],color=c_list(linenum),edgecolors='none',marker='.', s=0.5)
#     # chfreq = freq[linenum*3,]/1000.0
#     # line.set_label("%.2i" % chfreq)
#     # line, = plt.plot(time_axis, dataR[:,channel+1])
#     # line.set_label('3.2GHz')
#     # line, = plt.plot(time_axis, dataL[:,channel])
#     # line, = plt.plot(time_axis, dataR[:,channel]-dataL[:,channel])

# # ax = plt.gca()
# locator = mdates.AutoDateLocator()
# # formatter = mdates.AutoDateFormatter(locator)
# sub.set_xlim(-1200,37200)
# sub.xaxis.set_major_locator(lab.MultipleLocator(dt_major));
# sub.xaxis.set_major_formatter(lab.FuncFormatter(hhmm_format));
# sub.xaxis.set_minor_locator(lab.MultipleLocator(dt_minor));
# # ax.xaxis.set_minor_formatter(lab.FuncFormatter(hhmm_format));
# sub.set_ylim(-1e2,1e3)
# sub.set_xlabel("UT Time",fontsize=12)
# sub.set_ylabel("Solar flux, sfu",fontsize=12)
# fig.suptitle(date + "   3-24 GHz spectropolarimeter flux plot    Stokes I,V", fontsize=12, y = 0.92)
# # plt.legend(loc='upper right',fontsize=10)
# # plt.show()
# # plt.savefig('image.png', bbox_inches='tight',pad_inches = 0.01)
# # lab.show()
# lab.savefig(pngfilename, bbox_inches='tight',pad_inches = 0.15)
# lab.close()
# plt.close()

print (dataR.shape)
dataI = np.zeros(dataR.shape);
dataV = np.zeros(dataR.shape);

for freqnummap in range(48):
    # print (freqnummap)
    # dataR[:,freqnummap] = dataR[:,freqnummap]*coeff_R[freqnummap]-sefd_R[freqnummap]
    # dataL[:,freqnummap] = dataL[:,freqnummap]*coeff_L[freqnummap]-sefd_L[freqnummap]
    # dataI[:,freqnummap] = dataR[:,freqnummap]/2.0 + dataL[:,freqnummap]/2.0
    # dataV[:,freqnummap] = dataR[:,freqnummap]/2.0 - dataL[:,freqnummap]/2.0
    dataI[:,freqnummap] = (dataR[:,freqnummap]*coeff_R[freqnummap]-sefd_R[freqnummap])/2.0 + (dataL[:,freqnummap]*coeff_L[freqnummap]-sefd_L[freqnummap])/2.0
    dataV[:,freqnummap] = (dataR[:,freqnummap]*coeff_R[freqnummap]-sefd_R[freqnummap])/2.0 - (dataL[:,freqnummap]*coeff_L[freqnummap]-sefd_L[freqnummap])/2.0

# plt.ion()
# print (dataI)
# print (dataV)

# plt.figure(figsize = (15,7))

# freqs = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.0, 6.4, 6.8, 7.2, 7.6, 8.0, 8.4, 8.8, 9.2, 9.6, 10.0, 10.4, 10.8, 11.2, 11.6, 12.0, 12.0, 12.8, 13.6, 14.4, 15.2, 16.0, 16.8, 17.6, 18.4, 19.2, 20.0, 20.8, 21.6, 22.4, 23.2, 24.0]
# freqs = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.0, 6.4, 6.8, 7.2, 7.6, 8.0, 8.4, 8.8, 9.2, 9.6, 10.0, 10.4, 10.8, 11.2, 11.6, 12.0]
freqs = ['', '', '', '', '', 4.0, '', '', '', '', 5.0, '', '', '', '', 6.0, '', '', '', '', '', 8.0, '', '', '', '', 10.0, '', '', '', '', 12.0]
# freqs = [3.0, '', 4.0, '', 5.0, '', 6.0, 6.0, '', 8.0, '', 10.0, '', '']

x = np.array(time_axis)
# y = np.array(freqs)
# X,Y = np.meshgrid(time_axis,freqs)


# plt.pcolormesh(X,Y,dataI[:,0:32].T,cmap="rainbow",vmin = 0.0, vmax=1000.0, rasterized = True,shading='gouraud')
# plt.show()

dataI_slice = dataI[:,0:32]
dataV_slice = dataV[:,0:32]
dfI = pd.DataFrame(dataI_slice, columns=freqs)
dfV = pd.DataFrame(dataV_slice, columns=freqs)

dfI['time'] = x
dfI.time = pd.to_timedelta(dfI.time,unit="s");
dfI.set_index("time",inplace=True)

dfV['time'] = x
dfV.time = pd.to_timedelta(dfV.time,unit="s");
dfV.set_index("time",inplace=True)
# print(df)

dataI_resampled = dfI.resample("1s").mean()
dataV_resampled = dfV.resample("1s").mean()

figrows, figcolumns = 3, 1
# plt.close()
fig = lab.figure(figsize = (13.5,7.25))
# fig.set_constrained_layout(True)
fig.subplots_adjust(wspace=0, hspace=0,left = 0.05, right = 1.07, top = 0.91, bottom = 0.05)
fig.tight_layout()
sub3 = fig.add_subplot(figrows, figcolumns, 3);
for linenum in range(saveFreqs.shape[0]):
    procfreq = saveFreqs[linenum]
    # plt.ylim(4.0e-12, 5.0e-12)
    # print(coeff_R[procfreq],coeff_L[procfreq],sefd_R[procfreq],sefd_L[procfreq])
    pic3 = sub3.scatter(time_axis, (dataR[:,procfreq]*coeff_R[procfreq]-sefd_R[procfreq])/2.0 + (dataL[:,procfreq]*coeff_L[procfreq]-sefd_L[procfreq])/2.0, color=c_list(linenum),edgecolors='none',marker='.', s=1.0, label='%2.1f GHz'%(freq[procfreq]/1000000.0))
    sub3.scatter(time_axis, (dataR[:,procfreq]*coeff_R[procfreq]-sefd_R[procfreq])/2.0 - (dataL[:,procfreq]*coeff_L[procfreq]-sefd_L[procfreq])/2.0, color=c_list(linenum),edgecolors='none',marker='.', s=1.0)
    # sub.scatter(time_axis, (0.5*(dataR[:,procfreq]*coeff_R[procfreq]-sefd_R[procfreq])+0.5*(dataL[:,procfreq])*coeff_L[procfreq]-sefd_L[procfreq]),color=c_list(linenum),edgecolors='none',marker='.', s=1.0, label='%d MHz'%(freq[procfreq]/1000.0))
    # sub.scatter(time_axis, (0.5*(dataR[:,procfreq]*coeff_R[procfreq]-sefd_R[procfreq])-0.5*(dataL[:,procfreq])*coeff_L[procfreq]-sefd_L[procfreq]),color=c_list(linenum),edgecolors='none',marker='.', s=1.0)
    # sub3.legend(markerscale=10,loc='upper right',fontsize=10)
    # line = plt.scatter(time_axis, dataR[:,linenum*2],color=c_list(linenum),edgecolors='none',marker='.', s=0.5)
    # chfreq = freq[linenum*3,]/1000.0
    # line.set_label("%.2i" % chfreq)
    # line, = plt.plot(time_axis, dataR[:,channel+1])
    # line.set_label('3.2GHz')
    # line, = plt.plot(time_axis, dataL[:,channel])
    # line, = plt.plot(time_axis, dataR[:,channel]-dataL[:,channel])
order = [6,5,4,3,2,1,0]
handles, labels = sub3.get_legend_handles_labels()
sub3.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(1.10, 1.0),markerscale=9,loc='upper right',fontsize=9)
# sub3.legend(bbox_to_anchor=(1.10, 1.0),markerscale=10,loc='upper right',fontsize=10)
# ax = plt.gca()
locator = mdates.AutoDateLocator()
# formatter = mdates.AutoDateFormatter(locator)
sub3.set_xlim(-1200,37200)
sub3.xaxis.set_major_locator(lab.MultipleLocator(dt_major));
sub3.xaxis.set_major_formatter(lab.FuncFormatter(hhmm_format));
sub3.xaxis.set_minor_locator(lab.MultipleLocator(dt_minor));
# ax.xaxis.set_minor_formatter(lab.FuncFormatter(hhmm_format));
sub3.set_ylim(-100.0,800.0)
sub3.set_xlabel("UT Time",fontsize=11)
sub3.xaxis.set_label_coords(1.03, -0.05)
sub3.set_ylabel("Stokes I,V plot, sfu",fontsize=12)
cbar = fig.colorbar(pic3, shrink=0.0, pad = 0.007)
cbar.ax.tick_params(size=0)
cbar.set_ticks([])

sub2=fig.add_subplot(figrows, figcolumns, 2)
pic2 = sub2.imshow(dataV_resampled.T,
            cmap="seismic", 
            aspect = "auto",
            origin='lower',
            interpolation='bilinear',
            extent=[time_axis[0], time_axis[timenum-1], 0.0, 31.0],
            vmin = -50.0,
            vmax = 50.0)
ax = plt.gca()
ax.set_xlim(-1200,37200)
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(lab.MultipleLocator(dt_major));
# ax.xaxis.set_major_formatter(lab.FuncFormatter(hhmm_format));
ax.xaxis.set_minor_locator(lab.MultipleLocator(dt_minor));
# ax.xaxis.set_minor_formatter(lab.FuncFormatter(hhmm_format));
ax.set_xticklabels([])
# plt.xlabel("UT Time, hours",fontsize=10)
plt.ylabel("Frequency, GHz",fontsize=12)
# plt.title(date + "   3-24 GHz spectropolarimeter flux map    Stokes V", fontsize=12)#, y = 0.92)
plt.yticks(range(len(freqs)),freqs)
cbr2 = fig.colorbar(pic2, shrink=0.9, pad = 0.007,)
cbr2.set_label('Stokes V map, sfu',fontsize=12)

sub1 = fig.add_subplot(figrows, figcolumns, 1,)
pic1 = sub1.imshow(dataI_resampled.T,
            cmap="turbo", 
            aspect = "auto",
            origin='lower',
            interpolation='bicubic',
            extent=[time_axis[0], time_axis[timenum-1], 0.0, 31.0],
            vmin = 0.0,
            vmax = 1000.0)
ax = plt.gca()
ax.set_xlim(-1200,37200)
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(lab.MultipleLocator(dt_major));
ax.xaxis.set_major_formatter(lab.FuncFormatter(hhmm_format));
ax.xaxis.set_minor_locator(lab.MultipleLocator(dt_minor));
# ax.xaxis.set_minor_formatter(lab.FuncFormatter(hhmm_format));
ax.set_xticklabels([])
ax2 = ax.twiny()
ax2.set_xlim(-1200,37200)
ax2.xaxis.set_major_locator(lab.MultipleLocator(dt_major));
ax2.xaxis.set_major_formatter(lab.FuncFormatter(hhmm_format));
ax2.xaxis.set_minor_locator(lab.MultipleLocator(dt_minor));
ax2.tick_params(axis='x', which='major', pad=0)
# ax.xaxis.set_minor_formatter(lab.FuncFormatter(hhmm_format));
# plt.xlabel("UT Time, hours",fontsize=10)
sub1.set_ylabel("Frequency, GHz",fontsize=12)
# sub1.title(date + "   3-24 GHz spectropolarimeter flux map    Stokes I, V", fontsize=12)#, y = 0.92)
fig.suptitle(date + "     3-24 GHz spectropolarimeter flux   Stokes I,V              ", fontsize=12, y = 0.98)
ax.set_yticks(range(len(freqs)))
ax.set_yticklabels(freqs)
cbr1 = fig.colorbar(pic1, shrink=0.9, pad = 0.007)
cbr1.set_label('Stokes I map, sfu',fontsize=12)

# fig.suptitle(date + "   3-24 GHz spectropolarimeter flux plot    Stokes I,V", fontsize=12, y = 0.92)
# plt.legend(loc='upper right',fontsize=10)
# plt.show()
# plt.savefig('image.png', bbox_inches='tight',pad_inches = 0.15)
# plt.savefig('image.png')
# lab.show()
# fig.tight_layout()
lab.savefig(pngfilename)
# plt.show()

lab.close()
plt.close()

# print (time_axis)
# dataFormat = str(time_axis.shape[0]) + 'E'
freqsColumn = fits.Column(name='frequencies',format='D',array=freq_list)
timeColumn = fits.Column(name='time',format="48E",array=time)
IColumn = fits.Column(name='I',format="48E",array=dataI)
VColumn = fits.Column(name='V',format="48E",array=dataV)

fTableHdu = fits.BinTableHDU.from_columns([freqsColumn]);
dTableHdu = fits.BinTableHDU.from_columns([timeColumn, IColumn, VColumn])
pHeader = fits.Header()

pHdu = fits.PrimaryHDU(header=pHeader)
hduList = fits.HDUList([pHdu,fTableHdu, dTableHdu])
hduList.writeto('bssd324_'+today.strftime("%Y_%m_%d")+'.fits',clobber=True)
hduList.close()


