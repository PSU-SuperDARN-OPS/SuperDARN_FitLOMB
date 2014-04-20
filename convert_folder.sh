# script to use GNU parallel to parallelize fitlomb calculations..
#for f in /mnt/windata/sddata/rawacf/*kod.c*.rawacf ; do python2 rawacf_to_fitlomb.py --infile "$f"; done

#for f in rawacfs/*kod.d*.rawacf ; do python2 rawacf_to_fitlomb.py --infile "$f"; done
find ~/kodhaarp/*.rawacf | parallel -j 8 'python2 rawacf_to_fitlomb.py --infile {}'
