#! /usr/bin/env python

from iplMincTools import mincTools


if __name__ == '__main__':
 m=mincTools()
 m.non_linear_register_increment(
  'ellipse_1.mnc','ellipse_2.mnc',
  'test.xfm',
  source_mask='mask.mnc',
  target_mask='mask.mnc',
  level=8
    )
                                                                                                                                                                    
 
