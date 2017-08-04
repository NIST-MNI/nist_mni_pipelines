#! /bin/sh
mkdir -p test_data

object_opts="-short -real_range 0 100 -background 0 -edge_value 100 -fill_value 100  -nelements 50 50 50 -step 2 2 2 -start -50 -50 -50"
mask_opts="-byte -real_range 0 1 -background 0 -edge_value 1 -fill_value 1 -no_partial  -nelements 50 50 50 -step 2 2 2 -start -50 -50 -50"

# make bunch of ellipses
make_phantom $object_opts -ellipse -center -10 0 0 -width 20 10 10  test_data/ellipse_1.mnc
make_phantom $object_opts -ellipse -center 0 0 0   -width 20 10 10  test_data/ellipse_2.mnc
make_phantom $object_opts -ellipse -center 10 0 0  -width 20 10 10  test_data/ellipse_3.mnc

make_phantom $object_opts -ellipse -center 0 -10 0 -width 10 20 10 test_data/ellipse_4.mnc
make_phantom $object_opts -ellipse -center 0 0 0   -width 10 20 10 test_data/ellipse_5.mnc
make_phantom $object_opts -ellipse -center 0 10 0  -width 10 20 10 test_data/ellipse_6.mnc

make_phantom $object_opts -ellipse -center 0 0 -10 -width 10 10 20 test_data/ellipse_7.mnc
make_phantom $object_opts -ellipse -center 0 0  0  -width 10 10 20 test_data/ellipse_8.mnc
make_phantom $object_opts -ellipse -center 0 0 10  -width 10 10 20 test_data/ellipse_9.mnc

# make mask
make_phantom $mask_opts -rectangle -center 0 0 0 -width 50 50 50 test_data/mask.mnc


# make reference
make_phantom $mask_opts -ellipse -center 0 0 0 -width 15 15 15 test_data/ref.mnc

cat - >subjects.lst <<END
test_data/ellipse_1.mnc,test_data/mask.mnc
test_data/ellipse_2.mnc,test_data/mask.mnc
test_data/ellipse_3.mnc,test_data/mask.mnc
test_data/ellipse_4.mnc,test_data/mask.mnc
test_data/ellipse_5.mnc,test_data/mask.mnc
test_data/ellipse_6.mnc,test_data/mask.mnc
test_data/ellipse_7.mnc,test_data/mask.mnc
test_data/ellipse_8.mnc,test_data/mask.mnc
test_data/ellipse_9.mnc,test_data/mask.mnc
END
