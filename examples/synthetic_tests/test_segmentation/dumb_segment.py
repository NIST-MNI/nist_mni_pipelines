#! /usr/bin/env python

# standard library
import string
import os
import argparse
import pickle
import cPickle
import sys
import json
import csv
# minc
import minc

# numpy
import numpy as np

def coords(string):
    c=[float(i) for i in string.split(',')]
    
    if len(c)!=3 :
        raise argparse.ArgumentTypeError('Expect three coordinates')
    return c

def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description='Perform error-correction learning and application')

    parser.add_argument('--center',type=coords, 
                    default=[0.0,0.0,0.0],action = 'store',
                    help="Center coordinate")

    parser.add_argument('input',
                    help='Input image')

    parser.add_argument('output',
                    help='Output image')

    options = parser.parse_args()

    return options


def dumb_segment(img, center):
    
    c=np.mgrid[ 0:img.shape[0] ,
                0:img.shape[1] ,
                0:img.shape[2] ]

    seg=np.zeros_like( img, dtype=np.int32 )
    
    seg=( c[2]>center[0] )*1+\
        ( c[1]>center[1] )*2+\
        ( c[0]>center[2] )*4+ 1
    
    seg[ img < 50 ] = 0
    
    return np.asarray(seg,dtype=np.int32 )

if __name__ == "__main__":
    options = parse_options()
    print(repr(options))
    input = minc.Image(options.input)
    #seg=np.zeros_like( input.data, dtype=np.int32 )
    center_vox=[(options.center[i]-input.start()[i])/input.spacing()[i] for i in xrange(3)]
    print(repr(center_vox))
    seg=dumb_segment(input.data, center_vox)

    minc.Label( data=seg ).save(name=options.output, imitate=options.input)
    
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on
