# functions to parse and plot fitlomb hdf5 files
# jon klein, jtklein@alaska.edu 

import argparse
import h5py
import pdb
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import numpy as np
import glob
from pytz import timezone

MAX_LOMBDEPTH = 2
DATADIR = './data/'
VEL_CMAP = plt.cm.RdBu
FREQ_CMAP = plt.cm.spectral
NOISE_CMAP = plt.cm.autumn
SPECW_CMAP = plt.cm.hot
POWER_CMAP = plt.cm.jet
WHITE = 3e3

cdict3 = {'red':  ((0.0, 0.0, 0.0),
                   (0.25, 1.0, 1.0),
                   (0.5, 1.0, 0.0),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.9, 0.9)),

         'green': ((0.0, 1.0, 1.0),
                   (0.25, 1.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 0.9, 0.9)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 1.0),
                   (1.0, 1.0, 1.0))
        }
plt.register_cmap(name='SD_V', data=cdict3)


# gets a parameter across the beam for the file with a mask
# for example, param 'p_l' with maskparam 'qflg' will return a range x time x lombdepth array of power for the beam
def getParam(lombfit, beam, param, maskparam = False):
    gpath = str(beam) + '/' # group path
    pulses = getPulses(lombfit, gpath)
     
    times = [datetime.datetime.fromtimestamp(p.attrs['epoch.time']) for p in pulses]
    rgates = [p.attrs['nrang'] for p in pulses] 
    ranges = [np.arange(p.attrs['nrang']) * p.attrs['rsep'] + p.attrs['frang'] for p in pulses]

    # powers is long max(ranges) in case the number of range gates changes over the file
    rtiparam = np.ones([len(times), max(rgates), MAX_LOMBDEPTH])
    
    for (t,pulse) in enumerate(pulses):

        zeropad = np.zeros([max(rgates) - pulse.attrs['nrang'], MAX_LOMBDEPTH])

        if maskparam:
            rtiparam[t,:,:] = np.vstack((pulse[param][:] * pulse[maskparam][:], zeropad))
            rtiparam[t,:,:] += (np.vstack((pulse[maskparam][:], zeropad)) == 0) * WHITE
        else:
            rtiparam[t,:,:] = np.vstack((pulse[param][:], zeropad))
    return times, ranges, rtiparam 

# creates a file with all data from a radar in a folder using soft links
def createMergefile(radar, day, datadir):
    hdf5files = glob.glob(datadir + '*' + str(day) + '.*' + radar + '*.hdf5')

    filename = radar + '.hdf5' 
    mergefile = h5py.File(datadir + filename, 'w')
    
    for h5f in hdf5files:
        print h5f
        f = h5py.File(h5f, 'r')
        for beam in f:
            for pulse in f[beam]:
                dset = beam + '/' + pulse 
                mergefile[dset] = h5py.ExternalLink(h5f.split('/')[-1], dset)


        f.close()
    mergefile.close()

    return datadir + filename
# returns a time sorted list of pulses 
def getPulses(lombfit, group_path):
    # grap all pulses from a path (for example, a beam number)
    pulses = [lombfit[group_path + t] for t in lombfit[group_path]]
    # sort list of pulses by epoch time (hdf5 doesn't sort datasets within a group)
    pulses = sorted(pulses, key = lambda pulse: pulse.attrs['epoch.time'])
    return pulses

def PlotRTI(times, ranges, z, cmap, lim):
    times = [t + datetime.timedelta(hours = 8) for t in times] # correct for python automatically adding timezones... 

    for i in range(MAX_LOMBDEPTH):
        plt.subplot(MAX_LOMBDEPTH, 1, i+1)
        x = dates.date2num(times) 

        maxranges = ranges[np.argmax([len(r) for r in ranges])]
        y = maxranges

        # replicate data for plotting if number of range gates is lower than maximum 
        # this assumes frang doesn't change
        for (j,r) in enumerate(ranges):
            repfactor = int(len(maxranges) / len(r))
            ztemp = (z[j][:len(r)]).copy()

            for k in range(len(maxranges)):
                z[j][k] = ztemp[k/repfactor]
        # so, number of range gates varies from 75 to 225
        # need to fit into array of size 255...
        plt.pcolor(x, y, z[:,:,i].T, cmap = cmap)
        plt.axis([x.min(), x.max(), y.min(), y.max()])

        plt.clim(lim)
        ax = plt.gca()

        hfmt = dates.DateFormatter('%m/%d %H:%M', tz = None) 
        ax.xaxis.set_major_locator(dates.MinuteLocator(interval = 60))
        ax.xaxis.set_major_formatter(hfmt)
        ax.set_ylim(bottom = 0)
        plt.xticks(rotation=-45)
        plt.subplots_adjust(bottom=.3)

        plt.tight_layout()
        #locator = dates.AutoDateLocator()
        #dateformatter = dates.AutoDateFormatter(locator)

        #ax.xaxis.set_major_locator(locator)
        #ax.xaxis.set_major_formatter(dateformatter)
        #plt.xticks(rotation='vertical')

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
def Plot_p_l(beam, cmap = POWER_CMAP):
    times, ranges, powers = getParam(lombfit, beam, 'p_l', 'qflg')
    PlotRTI(times, ranges, powers, cmap, [0, 50])
    FormatRTI('time', 'range (km)', 'p_l (dB)', 'p_l (dB)')
    plt.show()

def Plot_w_l(beam, cmap = FREQ_CMAP):
    times, ranges, powers = getParam(lombfit, beam, 'w_l', 'qflg')
    PlotRTI(times, ranges, powers, cmap, [0, 500])
    FormatRTI('time', 'range (km)', 'w_l (m/s)', 'w_l (m/s)')
    plt.show()

def Plot_v_l(beam, cmap = plt.cm.get_cmap("SD_V")):
    times, ranges, vels = getParam(lombfit, beam, 'v_l', 'qflg')
    PlotRTI(times, ranges, vels, cmap, [-500, 500])
    FormatRTI('time', 'range (km)', 'v_l (m/s)', 'v_l (m/s)')
    plt.show()


if __name__ == '__main__':
    day = 18
    createMergefile('kod', day, DATADIR)
    lombfit = h5py.File(DATADIR + 'kod.hdf5', 'r')

    Plot_p_l(beam = 9)
    Plot_w_l(beam = 9)
    Plot_v_l(beam = 9)

    lombfit.close()
