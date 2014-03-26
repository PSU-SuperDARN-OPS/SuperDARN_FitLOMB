import argparse
import h5py
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pdb
import numpy as np

MAX_LOMBDEPTH = 3

VEL_CMAP = plt.cm.RdBu
FREQ_CMAP = plt.cm.spectral
NOISE_CMAP = plt.cm.autumn
SPECW_CMAP = plt.cm.hot
POWER_CMAP = plt.cm.jet


def getParam(lombfit, beam, param, maskparam = False):
    gpath = str(beam) + '/' # group path
    pulses = getPulses(lombfit, gpath)
     
    times = [datetime.datetime.fromtimestamp(p.attrs['epoch.time']) for p in pulses]
    ranges = [p.attrs['nrang'] for p in pulses] 
    rgates = [np.arange(p.attrs['nrang']) * p.attrs['rsep'] for p in pulses]

    # powers is long max(ranges) in case the number of range gates changes over the file
    rtiparam = np.zeros([len(times), max(ranges), MAX_LOMBDEPTH])
    
    for (t,pulse) in enumerate(pulses):

        zeropad = np.zeros([max(ranges) - pulse.attrs['nrang'], MAX_LOMBDEPTH])

        if maskparam:
            rtiparam[t,:,:] = np.vstack((pulse[param][:] * pulse[maskparam][:], zeropad))
        else:
            rtiparam[t,:,:] = np.vstack((pulse[param][:], zeropad))
        
    return times, rgates, rtiparam 

# returns a time sorted list of pulses 
def getPulses(lombfit, group_path):
    pulses = [lombfit[group_path + t] for t in lombfit[group_path]]
    pulses.sort()
    return pulses


def PlotRTI(times, ranges, z):
    for i in range(MAX_LOMBDEPTH):
        x = dates.date2num(times)
        y = ranges[0]
        plt.subplot(MAX_LOMBDEPTH, 1, i)
        pdb.set_trace()
        plt.pcolor(x, y, powers[:,:,i].T, cmap = SPECW_CMAP)
        plt.clim([0,100])
        plt.axis([x.min(), x.max(), y.min(), y.max()])
        #plt.axis([x.min(), x.max(), y.min(), y.max()])
        plt.grid(True)
        plt.colorbar()
        ax = plt.gca()

        locator = dates.AutoDateLocator()
        dateformatter = dates.AutoDateFormatter(locator)

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(dateformatter)
    plt.show()

if __name__ == '__main__':
    lombfit = h5py.File('20130320.0801.00.mcm.fitlomb.hdf5', 'r')
    times, ranges, powers = getParam(lombfit, 9, 'p_l', 'qflg')
    
    PlotRTI(times, ranges, powers)

    lombfit.close()
