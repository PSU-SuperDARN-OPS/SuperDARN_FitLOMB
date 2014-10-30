# jon klein jtklein@alaska.edu
# python functions to copy rawacf data from big dipper to chiniak

import subprocess
import time
import pdb
import signal
import datetime
import os, sys
from dateutil.relativedelta import relativedelta

SSHTIMEOUT = 0 # bigdipper may take a while..

ARCHIVE_DIR = '/sam-qfs/SUPERDARN/mirror/sddata/rawacf/' 
CACHE_DIR = '/home/radar/raid0/SuperDARN/data/rawacf/'
ARCHIVE_COMP = 'bigdipper.arsc.edu' 
ARCHIVE_USER = 'jtklein'
RSYNC_PATH = '--rsync-path=/usr/local/bin/rsync'
STAGE_CMD = 'bash ~/sd_archive_tools/stage_rawacfs.bash'
 
# runs a command over ssh, returns response
def remote_command(user, radar, command, timeout = SSHTIMEOUT):
        try:
                signal.alarm(timeout)
                failed = True
                out = ''
                cmdlist = ["ssh", user + '@' + radar, '"' + command + '"']
                print cmdlist
                s = subprocess.Popen(cmdlist, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
                
                while True:
                    nextline = s.stdout.readline()
                    if nextline == '' and s.poll() != None:
                        break
                    sys.stdout.write(nextline)
                    sys.stdout.flush() 


                out, err = s.communicate()
        except:
                print 'command ' + command + ' radar ' + radar + ' failed'

        signal.alarm(0)
        return out


def cache_data(radar, startdate, enddate):
        print 'caching data for ' + radar + ' from bigdipper to /raid0 via chiniak.. this may take a few hours'
        print 'staging data on bigdipper'
        stagedate = startdate
        while stagedate.month <= enddate.month and stagedate.year <= enddate.year:
                remote_command(ARCHIVE_USER, ARCHIVE_COMP, STAGE_CMD + startdate.strftime(" %Y %m"))
                stagedate += relativedelta(months = 1)
        
        print 'copying data to chiniak for each day of data'
    
        stagedate = startdate
        while stagedate <= enddate:
                bigdipper_dir = ARCHIVE_DIR + stagedate.strftime("/%Y/%m/")
                chiniak_dir = CACHE_DIR + stagedate.strftime("/%Y/%m.%d/")
                pattern = '--include=' + stagedate.strftime('"%Y%m%d.*.' + radar + '*rawacf*"')
                cmdlist = ['rsync', '-avz', '-e', 'ssh', ARCHIVE_USER + '@' + ARCHIVE_COMP + ':' + bigdipper_dir, chiniak_dir, pattern, '--exclude="*"', RSYNC_PATH, '--ignore-existing']
                print ' '.join(cmdlist)
                s = subprocess.Popen(' '.join(cmdlist), stdout = subprocess.PIPE, shell=True)

                while True:
                    nextline = s.stdout.readline()
                    if nextline == '' and s.poll() != None:
                        break
                    sys.stdout.write(nextline)
                    sys.stdout.flush() 

                out, err = s.communicate()
                stagedate += relativedelta(days = 1)

        print 'finished copying ' + radar + ' data from ' + str(startdate) + ' to ' + str(enddate)

def mount_raid0():
    # check if raid0 is already mounted
    # if not, mount using sshfs
    if not os.path.exists('/home/radar/raid0/SuperDARN'):
        print 'mounting qnap raid...'
        cmdlist = ['sshfs', 'jtklein@chiniak:/raid0', '/home/radar/raid0']
        s = subprocess.Popen(cmdlist, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        out, err = s.communicate()
    else:
        print 'qnap raid already mounted'

if __name__ == '__main__':
    mount_raid0()

    startdate = datetime.datetime(2014,03,04)
    enddate = datetime.datetime(2014,03,05)
    cache_data('pgr', startdate, enddate)
