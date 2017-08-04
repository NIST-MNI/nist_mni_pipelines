#! /usr/bin/env python

import minc
import sys
import os
import pyezminc
import numpy as np

from  sklearn import linear_model

if __name__ == "__main__":

    inp=pyezminc.parallel_input_iterator()
    out=pyezminc.parallel_output_iterator()
    

    design_matrix=np.array( [ [ 1, -0.5,-4],
                              [ 1, 0.5 ,-4],
                              [ 1, -0.5,-3],
                              [ 1, 0.5 ,-3],
                              [ 1, -0.5,-2],
                              [ 1, 0.5 ,-2],
                              [ 1, -0.5,-1],
                              [ 1, 0.5 ,-1],
                              [ 1, -0.5, 0],
                              [ 1, 0.5 , 0],
                              [ 1, -0.5, 1],
                              [ 1, 0.5 , 1],
                              [ 1, -0.5, 2],
                              [ 1, 0.5 , 2],
                              [ 1, -0.5, 3],
                              [ 1, 0.5 , 3],
                              [ 1, -0.5, 4],
                              [ 1, 0.5 , 4]] )
    
    inp.open([  'tmp_regress/8/object_0_0.mnc.008_vel.mnc',
                'tmp_regress/8/object_0_1.mnc.008_vel.mnc',
                'tmp_regress/8/object_0_2.mnc.008_vel.mnc',
                'tmp_regress/8/object_0_3.mnc.008_vel.mnc',
                'tmp_regress/8/object_0_4.mnc.008_vel.mnc',
                'tmp_regress/8/object_0_5.mnc.008_vel.mnc',
                'tmp_regress/8/object_0_6.mnc.008_vel.mnc',
                'tmp_regress/8/object_0_7.mnc.008_vel.mnc',
                'tmp_regress/8/object_0_8.mnc.008_vel.mnc',
                'tmp_regress/8/object_1_0.mnc.008_vel.mnc',
                'tmp_regress/8/object_1_1.mnc.008_vel.mnc',
                'tmp_regress/8/object_1_2.mnc.008_vel.mnc',
                'tmp_regress/8/object_1_3.mnc.008_vel.mnc',
                'tmp_regress/8/object_1_4.mnc.008_vel.mnc',
                'tmp_regress/8/object_1_5.mnc.008_vel.mnc',
                'tmp_regress/8/object_1_6.mnc.008_vel.mnc',
                'tmp_regress/8/object_1_7.mnc.008_vel.mnc',
                'tmp_regress/8/object_1_8.mnc.008_vel.mnc',
             ])

    out.open(["tmp_regress/fit_{}.mnc".format(i) for i in range(design_matrix.shape[1])],
             'tmp_regress/8/object_0_0.mnc.008_vel.mnc' )

    out_error=pyezminc.output_iterator_real(None)
    out_error.open("tmp_regress/fit_error.mnc",reference_file="tmp_regress/8/object_1_8.mnc.008.mnc")

    inp.begin()
    out.begin()
    out_error.begin()

    # allocate sum
    v1=np.zeros(shape=[design_matrix.shape[0]], dtype=np.float64, order='C')
    v2=np.zeros(shape=[design_matrix.shape[0]], dtype=np.float64, order='C')
    v3=np.zeros(shape=[design_matrix.shape[0]], dtype=np.float64, order='C')
    
    # allocate work space
    qqq=np.empty_like(v1)

    clf=linear_model.LinearRegression(fit_intercept=False)
    
    while not inp.last():
        # assume that we are dealing with 3D vectors
        # TODO: add check somewhere to make sure it is the case
        v1=inp.value(v1);inp.next()
        v2=inp.value(v2);inp.next()
        v3=inp.value(v3)

        # put things together
        y=np.column_stack((v1,v2,v3))
        x=design_matrix

        clf.fit(x,y)

        out.value(np.ravel(clf.coef_[0,:]));out.next()
        out.value(np.ravel(clf.coef_[1,:]));out.next()
        out.value(np.ravel(clf.coef_[2,:]));out.next()

        out_error.value(clf.score(x,y));out_error.next()

        inp.next()

    print out.progress()
    print inp.progress()
    print out_error.progress()

    del inp
    del out

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80;show-tabs on;hl python
