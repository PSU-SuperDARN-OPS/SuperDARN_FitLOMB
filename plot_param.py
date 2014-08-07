import argparse
from fitlomb_tools import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot fitlomb parameter')

    parser.add_argument('--radar', help='radar (e.g. mcm.a)', default='mcm.a')
    parser.add_argument('--lombdir', help='root directory of fitlomb data files', default='./testdata')
    parser.add_argument('--plotdir', help='directory to save plots', default='./plots')
    parser.add_argument('--sfilt', help='enable spatial filtering', default=0, type=int)
    parser.add_argument('--params', help='parameter to plot', default='v')
    parser.add_argument('--beam', help='beam to plot parameter on', type=int, default=8)
    parser.add_argument('--flag', help='flag parameter (eg. qflg)', default='')

    parser.add_argument('--year', help='year to start plot', type=int, default = 2013)
    parser.add_argument('--month', help='month to start plot', type=int, default = 3)
    parser.add_argument('--day', help='day of month to start plot', type=int, default = 20)
    parser.add_argument('--hour', help='hour to start plot', type=int, default = 0)
    parser.add_argument('--min', help='minute start plot', type=int, default = 0)
    parser.add_argument('--sec', help='second to start plot', type=int, default = 0)

    parser.add_argument('--dday', help='duration of plot, in days', default = 0, type=int)
    parser.add_argument('--dhour', help='duration of plot, in hours', default = 24, type=int)
    parser.add_argument('--dmin', help='duration of plot, in minutes', default = 0, type=int)
    parser.add_argument('--dsec', help='duration of plot, in seconds', default = 0, type=int)
    
    args = parser.parse_args()
    RADAR = args.radar # TODO: fix this..
    starttime = datetime.datetime(args.year, args.month, args.day, args.hour, args.min, args.sec)
    endtime = starttime + datetime.timedelta(days=args.dday, hours=args.dhour, minutes=args.dmin, seconds=args.dsec)

    mergefile = createMergefile(RADAR, starttime, endtime, DATADIR)
    lombfit = h5py.File(mergefile, 'r+')
    
    beams = [args.beam]
    remask(lombfit, starttime, endtime, beams, PMIN, QWMIN, QVMIN, WMAX, WMIN, VMAX, VMIN)

    plot_vector(lombfit, beams, args.params, args.flag, starttime, endtime)
        
