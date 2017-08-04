#! /bin/sh
set -e

mkdir -p test_data

tempdir=`mktemp -t test_XXXXXXXXX -d`
trap "rm -rf $tempdir" 0 1 2 15

object_opts="-short -real_range 0 100 -background 0 -edge_value 100 -fill_value 100  -nelements 100 100 100 -step 4 4 4 -start -200 -200 -200"
mask_opts="-byte -real_range 0 1 -background 0 -edge_value 1 -fill_value 1 -no_partial  -nelements 100 100 100 -step 4 4 4 -start -200 -200 -200"


make_phantom -short -real_range 0 1 -background 0 -edge_value 1 -fill_value 1  -nelements 100 100 100 -step 4 4 4 -start -200 -200 -200\
    -ellipse -center -10 0 0 -width 80 80 80 $tempdir/circle.mnc

fast_blur --fwhm 40 $tempdir/circle.mnc $tempdir/circle_blur.mnc


# make bunch of ellipses
make_phantom $object_opts -ellipse -center -10 0 0 -width 200 100 100 $tempdir/ellipse_1.mnc
make_phantom $object_opts -ellipse -center 0 0 0   -width 200 100 100 $tempdir/ellipse_2.mnc
make_phantom $object_opts -ellipse -center 10 0 0  -width 200 100 100 $tempdir/ellipse_3.mnc

make_phantom $object_opts -ellipse -center 0 -10 0 -width 100 200 100 $tempdir/ellipse_4.mnc
make_phantom $object_opts -ellipse -center 0 0 0   -width 100 200 100 $tempdir/ellipse_5.mnc
make_phantom $object_opts -ellipse -center 0 10 0  -width 100 200 100 $tempdir/ellipse_6.mnc

make_phantom $object_opts -ellipse -center 0 0 -10 -width 100 100 200 $tempdir/ellipse_7.mnc
make_phantom $object_opts -ellipse -center 0 0  0  -width 100 100 200 $tempdir/ellipse_8.mnc
make_phantom $object_opts -ellipse -center 0 0 10  -width 100 100 200 $tempdir/ellipse_9.mnc


for i in $(seq 1 9);do
  minccalc -express 'A[0]*A[1]' $tempdir/ellipse_$i.mnc $tempdir/circle_blur.mnc test_data/big_ellipse_$i.mnc
done

# make mask
make_phantom $mask_opts -rectangle -center 0 0 0 -width 200 200 200 test_data/big_mask.mnc

# make reference
make_phantom $object_opts -ellipse -center 0 0 0 -width 150 150 150 test_data/big_ref.mnc

cat - >big_subjects.lst <<END
test_data/big_ellipse_1.mnc,test_data/big_mask.mnc
test_data/big_ellipse_2.mnc,test_data/big_mask.mnc
test_data/big_ellipse_3.mnc,test_data/big_mask.mnc
test_data/big_ellipse_4.mnc,test_data/big_mask.mnc
test_data/big_ellipse_5.mnc,test_data/big_mask.mnc
test_data/big_ellipse_6.mnc,test_data/big_mask.mnc
test_data/big_ellipse_7.mnc,test_data/big_mask.mnc
test_data/big_ellipse_8.mnc,test_data/big_mask.mnc
test_data/big_ellipse_9.mnc,test_data/big_mask.mnc
END
