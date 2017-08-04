#! /bin/sh

in1=data/ellipse_0_blur.mnc
in2=data/ellipse_2_blur.mnc

# run Exponential
antsRegistration --collapse-output-transforms 0 -d 3 \
 --float 0 --verbose 1 --minc 1 \
 -c '[1000x1000x1000,1e-7,100]' \
 --transform 'Exponential[0.2,1.0,1.0]'  \
 -m "CC[$in1,$in2,1.0,4,Regular,0.1]" \
 -s 8x4x2 -f 8x4x2  \
 -o "[test_exp_,test_exp_in1.mnc,test_exp_in2.mnc]"
 
# run SyN
antsRegistration --collapse-output-transforms 0 -d 3 \
 --float 0 --verbose 1 --minc 1 \
 -c '[1000x1000x1000,1e-7,100]' \
 --transform 'SyN[0.2,1.0,1.0]'  \
 -m "CC[$in1,$in2,1.0,4,Regular,0.1]" \
 -s 8x4x2 -f 8x4x2  \
 -o "[test_syn_,test_syn_in1.mnc,test_syn_in2.mnc]"