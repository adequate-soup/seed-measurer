Benjamin Sims
benjamis20@berkeley.edu

Blackman Lab @ UC Berkeley
Last Updated July 11, 2024

Seed Measurer takes an image of seeds on a bright background and measures each seed's
area, length, and width to a CSV file. Your seeds must be pictured inside a box of known
width. The program will attempt to correct for perspecitve skew based on the shape of your box.

Image file names must be in the format: [populationName]_[*]_[*]_[year]_[*].[imageFileExtension]

Example CSV output for filename "NUB_8_7_2016_3_X_7_6_2016_4.jpg":

genotype                        population   year    seeds_imaged	area_mm2	length_mm	width_mm
NUB_8_7_2016_3_X_7_6_2016_4     NUB          2016    222           0.1982	    0.6711	    0.3968
NUB_8_7_2016_3_X_7_6_2016_4     NUB          2016    222           0.1968	    0.6812	    0.3600
NUB_8_7_2016_3_X_7_6_2016_4     NUB          2016    222           0.1911	    0.6520	    0.3575
NUB_8_7_2016_3_X_7_6_2016_4     NUB          2016    222           0.1907	    0.6730	    0.3582

This scripts is dependent on the following python libraries:
scipy, imutils, time, math, numpy, argparse, threading, csv, opencv, os
