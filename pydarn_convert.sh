# script to use GNU parallel to parallelize fitlomb calculations..

cat radars | parallel -j 6 'python2 pydarn_fitlombgen.py --radar {}'

