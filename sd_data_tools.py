# jon klein, jtklein@alaska.edu
# mit license

# tools to help work with SuperDARN data
import glob
import numpy as np

DATA_DIR = '.'

# sum array of values in dB
def dbsum(array):
    return 10*np.log10(sum([10**(p/10.) for p in array]))

# retrieve filenames from a radar at a day/time  
def get_filenames(radar, day, month, year, hour, filetype = '.fitacf', ddir = DATA_DIR):
    return glob.glob(ddir + '/' + str(year) + '/' + str(month).zfill(2)+'.' + str(day).zfill(2) + '/' + '*.*' + radar + filetype)




