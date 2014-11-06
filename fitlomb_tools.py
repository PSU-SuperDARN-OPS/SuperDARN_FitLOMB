# functions to parse and plot fitlomb hdf5 files
# jon klein, jtklein@alaska.edu 

# todo: add niterations detection

import argparse
import h5py
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import numpy as np
import glob
import time 
import getpass
import pdb

BEAMS = 16
MAX_LOMBDEPTH = 1
DATADIR = '/home/' + getpass.getuser() + '/fitlomb/'
PLOTDIR = './plots/'
VEL_CMAP = plt.cm.RdBu
FREQ_CMAP = plt.cm.spectral
NOISE_CMAP = plt.cm.autumn
SPECW_CMAP = plt.cm.hot
POWER_CMAP = plt.cm.jet
WHITE = 3e3
EPOCH = datetime.datetime(1970,1,1)
ALLBEAMS = [str(b) for b in range(BEAMS)]
MINRANGE = 75 
MAXRANGE = 5000 
TIMEINT = 20
BEAMS = [3]#[6, 7,8,9,10]#ALLBEAMS# [9]
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

QWMIN = -1000 # m/s
QVMIN = -1500 # m/s
PMIN = 3 # dB
WMAX = 1000 # m/s
WMIN = -WMAX 
VMAX = 1500 # m/s
VMIN = -VMAX

def dbscale(vec):
    return 50 * np.log10(vec)

def prettyify():
    plt.rcParams.update({'font.size': 16})
    #plt.rcParams.update({'font.weight': 'bold'})
    plt.rcParams.update({'legend.loc': 'best'})
    plt.legend(fancybox=True)

# gets a scalar value, possibly across all beams
def getScalar(lombfit, param, beams, starttime, endtime):
    pulses = getPulses(lombfit, beams, starttime, endtime)
    times = [datetime.datetime.utcfromtimestamp(p.attrs['epoch.time']) for p in pulses]

    return [p.attrs[param] for p in pulses], times

# gets record times which can index pydarn dmaps
def getRecordtimes(lombfit, beams, starttime, endtime):
    pulses = getPulses(lombfit, beams, starttime, endtime)
    rtimes = [datetime.datetime(p.attrs['time.yr'], p.attrs['time.mo'], p.attrs['time.dy'], p.attrs['time.hr'], p.attrs['time.mt'], p.attrs['time.sc'], p.attrs['time.us']) for p in pulses]
    return rtimes

# gets a parameter across the beam for the file with a mask
# for example, param 'p_l' with maskparam 'qflg' will return a range x time x lombdepth array of power for the beam
def getParam(lombfit, beam, param, starttime, endtime,  maskparam = False, blank = WHITE):
    pulses = getPulses(lombfit, beam, starttime, endtime)#[beam])
     
    times = [datetime.datetime.utcfromtimestamp(p.attrs['epoch.time']) for p in pulses]
    rgates = [p.attrs['nrang'] for p in pulses] 
    ranges = [np.arange(p.attrs['nrang']) * p.attrs['rsep'] + p.attrs['frang'] for p in pulses]
    # powers is long max(ranges) in case the number of range gates changes over the file
    rtiparam = np.ones([len(times), max(rgates), MAX_LOMBDEPTH])
    
    for (t,pulse) in enumerate(pulses):

        zeropad = np.zeros([max(rgates) - pulse.attrs['nrang'], MAX_LOMBDEPTH])

        if maskparam:
            rtiparam[t,:,:] = np.vstack((pulse[param][:] * pulse[maskparam][:], zeropad))
            rtiparam[t,:,:] += (np.vstack((pulse[maskparam][:], zeropad)) == 0) * blank 
        else:
            rtiparam[t,:,:] = np.vstack((pulse[param][:], zeropad))
    return times, ranges, rtiparam 

# creates a file with all data from a radar in a folder using soft links
def createMergefile(radar, starttime, endtime, datadir, beams = None):
    # todo: work with starttime and endttime datetimes
    
    # for each day between starttime and endtime
    # loop, adding 1 day to starttime until delta between starttime and endtime is <= 1 day

    filename = radar + starttime.strftime('%Y%m%d') + 'to' + endtime.strftime('%Y%m%d') + '.hdf5'
    mergefile = h5py.File(datadir + filename, 'w')
    while starttime < endtime:
        globname = datadir + starttime.strftime('/%Y/%m.%d/%Y%m%d.*.' + radar + '.fitlomb.hdf5') 
        hdf5files = glob.glob(globname)
        for h5f in hdf5files:
            try:
                f = h5py.File(h5f, 'r')
                for pulse in f['/']:
                    if beams == None or f[pulse].attrs['bmnum'] in beams:
                        dset = pulse
                        mergefile[dset] = h5py.ExternalLink(h5f.split('//')[-1], dset) # I've made a terrible mistake..

                f.close()
            except:
                print 'trouble opening ' + h5f + ', skipping..' 
        starttime = starttime + datetime.timedelta(days=1)
    mergefile.close()

    return datadir + filename

def PlotFreq(lombfit, beams, starttime, endtime, image = False):
    f, t = getScalar(lombfit, 'tfreq', beams, starttime, endtime)
    x = dates.date2num(t) 
    ax = plt.gca()
    hfmt = dates.DateFormatter('%m/%d %H:%M') 
    ax.xaxis.set_major_locator(dates.MinuteLocator(interval = TIMEINT))
    ax.xaxis.set_major_formatter(hfmt)
    plt.plot(x, f)
    plt.xticks(rotation=-45)
    plt.subplots_adjust(bottom=.3)
    plt.tight_layout()
    plt.xlabel('time (UTC)')
    plt.ylabel('frequency (kHz)')
    plt.grid(True)
    if not image:
        plt.show()
    else:
        imgname = get_imagename(t[0], t[-1], RADAR, 'freq', beam)
        plt.savefig(imgname, bbox_inches='tight')
        plt.clf()


def dt2epoch(dt):
    return (dt - EPOCH).total_seconds()

# returns a time sorted list of pulses 
# beams is a list of beam numbers
def getPulses(lombfit, beams, starttime, endtime):
    # grap all pulses from a path (for example, a beam number)0
    pulses = []
    group_path =  '/'
    for t in lombfit[group_path]:
        pulse = lombfit[group_path + t]
        if pulse.attrs['epoch.time'] >= dt2epoch(starttime) \
                and pulse.attrs['epoch.time'] <= dt2epoch(endtime)\
                and pulse.attrs['bmnum'] in [int(b) for b in beams]:
            pulses.append(pulse)

    # sort list of pulses by epoch time (hdf5 doesn't sort datasets within a group)
    pulses = sorted(pulses, key = lambda pulse: pulse.attrs['epoch.time'])
    return pulses

def PlotRTI(times, ranges, z, cmap, lim):
    for i in range(1):
        x = dates.date2num(times) 
        maxranges = ranges[np.argmax([len(r) for r in ranges])]
        y = maxranges

        # replicate data for plotting if number of range gates is lower than maximum 
        # this assumes frang doesn't change 
        for (j,r) in enumerate(ranges):
            repfactor = int(len(maxranges) / len(r))
            ztemp = (z[j][:len(r)]).copy()

            for k in range(len(maxranges)):
                if k/repfactor < len(ztemp):
                    z[j][k] = ztemp[k/repfactor]

        # so, number of range gates varies from 75 to 225
        # need to fit into array of size 255...
        rangemask = (y < MAXRANGE) * (y > MINRANGE)

        y = y[rangemask]
        z = z[:,rangemask]
        plt.pcolor(x, y, z[:,:,i].T, cmap = cmap)
        plt.axis([x.min(), x.max(), y.min(), y.max()])
        plt.clim(lim)
        ax = plt.gca()
        hfmt = dates.DateFormatter('%m/%d %H:%M') 
        ax.xaxis.set_major_locator(dates.MinuteLocator(interval = TIMEINT))
        ax.xaxis.set_major_formatter(hfmt)
        ax.set_ylim(bottom = MINRANGE)
        plt.xticks(rotation=-45)
        plt.subplots_adjust(bottom=.3)

        plt.tight_layout()

def FormatRTI(xlabel, ylabel, title, cbarlabel, beam):
    cb = plt.colorbar()

    #cb.ax.yaxis.set_ylabel_position('right')
    cb.ax.set_ylabel(cbarlabel, rotation='vertical')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title + ', ' + RADAR + ', ' + 'beam: ' + str(beam))

# file name
# 2014.18.20.20.to.2014.18.20.50.p_l.kod.d.png
def get_imagename(tstart, tstop, radar, param, beam):
    return PLOTDIR + tstart.strftime('%Y.%m.%d.%H%M') + '.to.' + tstop.strftime('%Y.%m.%d.%H%M') + '.' + param + '.' + radar + '_beam' + str(beam) + '.png'

# parameter specific plotting functions 
def Plot_p_l(lombfit, beam, starttime, endtime, cmap = POWER_CMAP, image = False):
    times, ranges, powers = getParam(lombfit, beam, 'p_l', starttime, endtime, maskparam = 'qflg', blank = 0)
    PlotRTI(times, ranges, powers, cmap, [0, 50])
    FormatRTI('time (UTC)', 'slant range (km)', 'p_l (dB)', 'p_l (dB)', beam)
    if not image:
        plt.show()
    else:
        imgname = get_imagename(times[0], times[-1], RADAR, 'p_l', beam)
        print imgname
        plt.savefig(imgname, bbox_inches='tight')
        plt.clf()

def Plot_w_l(lombfit, beam, starttime, endtime, cmap = FREQ_CMAP, image = False):
    times, ranges, powers = getParam(lombfit, beam, 'w_l', starttime, endtime, maskparam = 'qflg')
    PlotRTI(times, ranges, powers, cmap, [0, 1000])
    FormatRTI('time (UTC)', 'slant range (km)', 'w_l (m/s)', 'w_l (m/s)', beam)
    if not image:
        plt.show()
    else:
        imgname = get_imagename(times[0], times[-1], RADAR, 'w_l', beam)
        print imgname
        plt.savefig(imgname, bbox_inches='tight')
        plt.clf()


def Plot_v(lombfit, beam, starttime, endtime, cmap = plt.cm.get_cmap("SD_V"), image = False):
    times, ranges, vels = getParam(lombfit, beam, 'v', starttime, endtime, maskparam  ='qflg')
    PlotRTI(times, ranges, vels, cmap, [-1000, 1000])
    FormatRTI('time (UTC)', 'slant range (km)', 'v (m/s)', 'v (m/s)', beam)
    if not image:
        plt.show()
    else:
        imgname = get_imagename(times[0], times[-1], RADAR, 'v', beam)
        print imgname
        plt.savefig(imgname, bbox_inches='tight')
        plt.clf()

def plot_vector(lombfit, beam, param, flag, starttime, endtime, vmax = 1500, vmin = -1500, cmap = plt.cm.get_cmap("SD_V"), image = False, scale = None):
    times, ranges, vec = getParam(lombfit, beam, param, starttime, endtime, maskparam = flag)

    if scale:
        vec = scale(vec)

    PlotRTI(times, ranges, vec, cmap, [vmin, vmax])
    FormatRTI('time (UTC)', param, param, param, beam)
    if not image:
        plt.show()
    else:
        imgname = get_imagename(times[0], times[-1], RADAR, param, beam)
        print imgname
        plt.savefig(imgname, bbox_inches='tight')
        plt.clf()


# recalculate qflg to experiment with different data quality thresholds
def remask(lombfit, starttime, endtime, beams, pmin, qwmin, qvmin, wmax, wmin, vmax, vmin, median = False, snr = False):
    pulses = getPulses(lombfit, beams, starttime, endtime)
    if not snr:
        for (i,pul) in enumerate(pulses):
            qmask = (pul['p_l'][...] > pmin) * \
                    (pul['v'][...] < vmax) * \
                    (pul['v'][...] > vmin) * \
                    (pul['w_l'][...] < wmax) * \
                    (pul['w_l'][...] > wmin) * \
                    (pul['w_l_e'][...] < qwmin) * \
                    (pul['fit_snr_l'][...] > .25) * \
                    (pul['v_e'][...] < qvmin)
            pul['qflg'][:,:] = qmask
        # todo: fix median filter for differing nrang
        if median: 
            # apply spatial/temporal filter
            for (i,pul) in enumerate(pulses):
                # spatial filter
                qmask = pul['qflg'][:,:]
                sqmask = (qmask * np.append(qmask[1:],[[0,0]], axis=0)) + \
                         (qmask * np.append([[0,0]],qmask[1:], axis=0))

                # temporal filter
                tmask = np.zeros(qmask.shape)
                if i > 0:
                    tmask += qmask * pulses[i-1]['qflg'][:,:]
                if i < len(pulses) -1:
                    tmask += qmask * pulses[i+1]['qflg'][:,:]

                qmask = (sqmask * tmask) > 0
                pul['qflg'][:,:] = qmask
    else:
        for (i,pul) in enumerate(pulses):
            qmask = (50 * np.log10(pul['fit_snr_l'][...]) > .05 )
            pul['qflg'][:,:] = qmask



def PlotTime(radar, starttime, endtime, directory, beams):
    mergefile = createMergefile(RADAR, starttime, endtime, DATADIR, beams)
    lombfit = h5py.File(mergefile, 'r+')
    #remask(lombfit, starttime, endtime, beams, PMIN, QWMIN, QVMIN, WMAX, WMIN, VMAX, VMIN, median = False, snr = False)
    plot_vector(lombfit, beams, 'fit_snr_l' , '', starttime, endtime, vmax = 10, vmin = 0, cmap = POWER_CMAP, image=True, scale = dbscale)
    PlotFreq(lombfit, beams, starttime, endtime, image = True)
    Plot_p_l(lombfit, beams, starttime, endtime, image = True)
    Plot_w_l(lombfit, beams, starttime, endtime, image = True)
    Plot_v(lombfit, beams, starttime, endtime, image = True)


    plot_vector(lombfit, beams, 'v_e' , '', starttime, endtime, vmax = 200, vmin = 0, cmap = plt.cm.get_cmap("SD_V"), image=True)
    plot_vector(lombfit, beams, 'w_l_e' , '', starttime, endtime, vmax = 200, vmin = 0, cmap = plt.cm.get_cmap("SD_V"), image=True)
    plot_vector(lombfit, beams, 'nlag' , '', starttime, endtime, vmax = 25, vmin = 0, cmap = plt.cm.get_cmap("SD_V"), image=True)
    #plot_vector(lombfit, beams, 'r2_phase_l' , '', starttime, endtime, vmax = 1, vmin = -1, cmap = plt.cm.get_cmap("SD_V"), image=True)
    lombfit.close()

if __name__ == '__main__':
    prettyify() # set matplotlib parameters for larger text

    parser = argparse.ArgumentParser(description='Processes RawACF files with a Lomb-Scargle periodogram to produce FitACF-like science data.')

    parser.add_argument("--starttime", help="start time of the plot (yyyy.mm.dd.hh) e.g 2014.03.01.00", default = "2014.03.01.00")
    parser.add_argument("--endtime", help="ending time of the plot (yyyy.mm.dd.hh) e.g 2014.03.08.12", default = "2014.03.05.00")
    parser.add_argument("--maxplotlen", help="maximum length of a rti plot, in hours", default = 24)
    parser.add_argument("--radar", help="radar to create data from", default='mcm.a')
    parser.add_argument("--beam", help="beam to plot", default=9)
    parser.add_argument("--plotdir", help="directory to place plots (defaults to ./plots/)", default=PLOTDIR)
    args = parser.parse_args()

    PLOTDIR = args.plotdir 
    # parse date string and convert to datetime object
    starttime = datetime.datetime(*time.strptime(args.starttime, "%Y.%m.%d.%H")[:6])
    endtime = datetime.datetime(*time.strptime(args.endtime, "%Y.%m.%d.%H")[:6])
    maxplotlen = datetime.timedelta(hours = args.maxplotlen)

    plot_times = {}

    while starttime < endtime:
        if starttime + maxplotlen >= endtime: 
            plot_times[starttime] = endtime
        
        else:
            plot_times[starttime] = (starttime + maxplotlen)

        starttime = plot_times[starttime]
    
    for radar in [args.radar]:
        for stime in plot_times.keys():
            timespan = (stime - plot_times[stime])
            TIMEINT = -min(int((timespan.days * 24 * 60 + timespan.seconds / 60.) / 12.),-1)
            for beam in [args.beam]:
                print 'plotting '  + str(stime)
                RADAR = radar
                try:
                    PlotTime(RADAR, stime, plot_times[stime], DATADIR, [beam]) 
                except:
                    print 'error plotting ' + str(stime) + '... skipping'
