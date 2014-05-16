# script to use GNU parallel to parallelize fitlomb calculations..
find ~/mcmacf/*.rawacf | parallel -j 7 'python2 rawacf_to_fitlomb.py --infile {}'

