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
    
    parser.add_argument("--image-bar",
                    action="store_true",
                    dest="image_bar",
                    default=False,
                    help="Show Image colour-bar" )

    parser.add_argument("--mask-bar",
                    action="store_true",
                    dest="mask_bar",
                    default=False,
                    help="Show mask colour-bar" )
    
    parser.add_argument("--cmap",
                    dest="cmap",
                    default="gray",
                    help="Colour map" )

    parser.add_argument("--range",type=float,
                    dest="range",
                    nargs=2,
                    default=None,
                    help="Main image range" )

    parser.add_argument("--mask_cmap",
                    dest="mask_cmap",
                    default="hot",
                    help="Colour map for overlay" )
    
    parser.add_argument("--mask",
                    dest="mask",
                    default=None,
                    help="Add mask" )

    parser.add_argument("--mask-range",type=float,
                    dest="mask_range",
                    nargs=2,
                    default=None,
                    help="Mask range" )
    
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

    parser.add_argument("--big",
                    action="store_true",
                    default=False,
                    help="Make big view" )

    parser.add_argument("--bg",
                    dest="bg",
                    default=None,
                    help="Background color" )

    parser.add_argument("--fg",
                    dest="fg",
                    default=None,
                    help="Foreground color" )

    parser.add_argument("--style",
                    dest="style",
                    default=None,
                    help="Matplotlib style, see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html#sphx-glr-gallery-style-sheets-style-sheets-reference-py" )

    parser.add_argument("--title",
                    dest="title",
                    default=None,
                    help="Graph title" )

    parser.add_argument("--dpi",type=float,
                    default=100,
                    help="Target DPI" )

    parser.add_argument("--ialpha",type=float,
                    default=0.7,
                    help="Image Alpha (for alpha blending)" )

    parser.add_argument("--mask-bg",type=float,
                    default=0.5,dest="mask_bg",
                    help="Mask background level" )

    parser.add_argument("--oalpha",type=float,
                    default=0.3,
                    help="Overlay Alpha (for alpha blending)" )

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
            ipl.minc_qc.qc_field_contour(options.input,
                options.output, show_image_bar=options.bar,
                image_cmap=options.cmap, dpi=options.dpi,
                bg_color=options.bg,fg_color=options.fg,
                style=options.style)
        else:
            ipl.minc_qc.qc(options.input,
                options.output,
                mask=options.mask,
                use_max=options.use_max,
                use_over=options.use_over,
                mask_bg=options.mask_bg,
                image_cmap=options.cmap,
                mask_cmap=options.mask_cmap,
                bg_color=options.bg,
                fg_color=options.fg,
                samples=20 if options.big else 6,
                dpi=options.dpi,
                image_range=options.range,
                mask_range=options.mask_range,
                ialpha=options.ialpha,
                oalpha=options.oalpha,
                title=options.title,
                show_image_bar=options.image_bar,
                show_overlay_bar=options.mask_bar,
                style=options.style
                )
    else:
        print("Refusing to run without input data, run --help")
        exit(1)
        
# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80
