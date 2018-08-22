#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author Vladimir S. FONOV
# @date 07/05/2018
#
# QC images

import ipl.minc_qc
import argparse


def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Make QC image')

    parser.add_argument("--debug",
                    action="store_true",
                    dest="debug",
                    default=False,
                    help="Print debugging information" )
    
    parser.add_argument("--contour",
                    action="store_true",
                    dest="contour",
                    default=False,
                    help="Make contour plot" )
    
    parser.add_argument("--bar",
                    action="store_true",
                    dest="bar",
                    default=False,
                    help="Show colour-bar" )
    
    parser.add_argument("--cmap",
                    dest="cmap",
                    default=None,
                    help="Colour map" )
    
    parser.add_argument("--mask",
                    dest="mask",
                    default=None,
                    help="Add mask" )
    
    parser.add_argument("--over",
                    dest="use_over",
                    action="store_true",
                    default=False,
                    help="Overplot" )
    
    parser.add_argument("--max",
                    dest="use_max",
                    action="store_true",
                    default=False,
                    help="Use max mixing" )

    parser.add_argument("input",
                        help="Input minc file")

    parser.add_argument("output",
                        help="Output QC file")
    
    options = parser.parse_args()

    if options.debug:
        print(repr(options))

    return options    

def main():
    options = parse_options()
    if options.input is not None and options.output is not None:
        if options.contour:
            ipl.minc_qc.qc_field_contour(options.input,options.output,show_image_bar=options.bar,image_cmap=options.cmap)
        else:
            ipl.minc_qc.qc(options.input,options.output,mask=options.mask,use_max=options.use_max,use_over=options.use_over,mask_bg=0.5)
    else:
        print("Refusing to run without input data, run --help")
        exit(1)
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80
