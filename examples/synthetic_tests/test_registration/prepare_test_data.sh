#! /bin/sh


#tempdir=`mktemp -t test -d`
#trap "rm -rf $tempdir" 0 1 2 15
tempdir=data
mkdir -p $tempdir

object_opts="-short -real_range 0 100 -background 0 -edge_value 100 -fill_value 100  -nelements 97 101 103 -step 4 4 4 -start -200 -200 -200"

object_opts2="-short -real_range 0 100 -background 0 -edge_value 100 -fill_value 100 -nelements 97 101 103 -step 1 1 1 -start -50 -50 -50"


# make bunch of ellipses
make_phantom $object_opts -ellipse -center 0 0 0   -width 150 150 150 $tempdir/ellipse_0.mnc
make_phantom $object_opts -ellipse -center 0 20 0 -width 100 150 100 $tempdir/ellipse_1.mnc
make_phantom $object_opts -ellipse -center 0 -20 0  -width 100 150 100 $tempdir/ellipse_2.mnc

for i in $(seq 0 2);do
  fast_blur --fwhm 8 $tempdir/ellipse_$i.mnc $tempdir/ellipse_${i}_blur.mnc
done

make_phantom $object_opts2 -ellipse -center 0 0  0 -width 37 37 37 $tempdir/ellipse_0_.mnc
make_phantom $object_opts2 -ellipse -center 0 5  0 -width 25 37 25 $tempdir/ellipse_1_.mnc
make_phantom $object_opts2 -ellipse -center 0 -5 0 -width 25 37 25 $tempdir/ellipse_2_.mnc

for i in $(seq 0 2);do
  fast_blur --fwhm 4 $tempdir/ellipse_${i}_.mnc $tempdir/ellipse_${i}_blur_.mnc
done
