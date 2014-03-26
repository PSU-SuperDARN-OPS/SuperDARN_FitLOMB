# functions to parse and plot fitlomb hdf5 files
# jon klein, jtklein@alaska.edu 

import argparse
import h5py
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pdb
import numpy as np

MAX_LOMBDEPTH = 2

VEL_CMAP = plt.cm.RdBu
FREQ_CMAP = plt.cm.spectral
NOISE_CMAP = plt.cm.autumn
SPECW_CMAP = plt.cm.hot
POWER_CMAP = plt.cm.jet

# gets a parameter across the beam for the file with a mask
# for example, param 'p_l' with maskparam 'qflg' will return a range x time x lombdepth array of power for the beam
def getParam(lombfit, beam, param, maskparam = False):
    gpath = str(beam) + '/' # group path
    pulses = getPulses(lombfit, gpath)
     
    times = [datetime.datetime.fromtimestamp(p.attrs['epoch.time']) for p in pulses]
    rgates = [p.attrs['nrang'] for p in pulses] 
    ranges = [np.arange(p.attrs['nrang']) * p.attrs['rsep'] + p.attrs['frang'] for p in pulses]

    # powers is long max(ranges) in case the number of range gates changes over the file
    rtiparam = np.zeros([len(times), max(rgates), MAX_LOMBDEPTH])
    
    for (t,pulse) in enumerate(pulses):

        zeropad = np.zeros([max(rgates) - pulse.attrs['nrang'], MAX_LOMBDEPTH])

        if maskparam:
            rtiparam[t,:,:] = np.vstack((pulse[param][:] * pulse[maskparam][:], zeropad))
        else:
            rtiparam[t,:,:] = np.vstack((pulse[param][:], zeropad))
        
    return times, ranges, rtiparam 

# returns a time sorted list of pulses 
def getPulses(lombfit, group_path):
    pulses = [lombfit[group_path + t] for t in lombfit[group_path]]
    pulses.sort()
    return pulses


def PlotRTI(times, ranges, z, cmap = SPECW_CMAP):
    for i in range(MAX_LOMBDEPTH):
        plt.subplot(MAX_LOMBDEPTH, 1, i+1)

        x = dates.date2num(times)
        y = ranges[0]

        plt.pcolor(x, y, z[:,:,i].T, cmap = SPECW_CMAP)
        plt.clim([0,100])
        plt.axis([x.min(), x.max(), y.min(), y.max()])
        plt.grid(True)
        ax = plt.gca()

        locator = dates.AutoDateLocator()
        dateformatter = dates.AutoDateFormatter(locator)

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(dateformatter)

def FormatRTI(xlabel, ylabel, title, cbarlabel):
    for i in range(MAX_LOMBDEPTH):
        cb = plt.colorbar()

        #cb.ax.yaxis.set_ylabel_position('right')
        cb.ax.set_ylabel(cbarlabel, rotation='vertical')

        plt.subplot(MAX_LOMBDEPTH, 1, i+1)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title + ' FitLomb iteration ' + str(i + 1))

# parameter specific plotting functions 
def Plot_p_l(beam):
    times, ranges, powers = getParam(lombfit, beam, 'p_l', 'qflg')
    PlotRTI(times, ranges, powers)
    FormatRTI('time', 'range (km)', 'p_l (dB)', 'p_l (dB)')
    plt.show()

def Plot_w_l(beam):
    times, ranges, powers = getParam(lombfit, beam, 'w_l', 'qflg')
    PlotRTI(times, ranges, powers)
    FormatRTI('time', 'range (km)', 'w_l (m/s)', 'w_l (m/s)')
    plt.show()

def Plot_v_l(beam):
    times, ranges, powers = getParam(lombfit, beam, 'v_l', 'qflg')
    PlotRTI(times, ranges, powers)
    FormatRTI('time', 'range (km)', 'v_l (m/s)', 'v_l (m/s)')
    plt.show()


if __name__ == '__main__':
    lombfit = h5py.File('20140324.1600.04.kod.fitlomb.hdf5', 'r')

    Plot_p_l(beam = 9)
    Plot_w_l(beam = 9)
    Plot_v_l(beam = 9)

    lombfit.close()
