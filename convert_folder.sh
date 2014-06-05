# script to use GNU parallel to parallelize fitlomb calculations..

find ~/rawdata/*.bz2 | parallel -j 6 'bunzip2 {}'
find ~/rawdata/*.rawacf | parallel -j 6 'python2 rawacf_to_fitlomb.py --infile {}'

