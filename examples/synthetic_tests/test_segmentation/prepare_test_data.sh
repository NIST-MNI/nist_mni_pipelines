#! /bin/sh


#tempdir=`mktemp -t test -d`
#trap "rm -rf $tempdir" 0 1 2 15
tempdir=data
mkdir -p $tempdir

object_opts="-short -real_range 0 100 -background 0 -edge_value 100 -fill_value 100  -nelements 97 101 103 -step 4 4 4 -start -200 -200 -200"

seg_opts="-byte -real_range 0 1 -background 0 -edge_value 1 -fill_value 1 -no_partial  -nelements 100 100 100 -step 4 4 4 -start -200 -200 -200"


make_phantom $object_opts -ellipse -center 0 0 0 -width 150 150 150 $tempdir/ellipse_0.mnc

# make bunch of ellipses
make_phantom $object_opts -ellipse -center -10 0 0 -width 150 100 100 $tempdir/ellipse_1.mnc
make_phantom $object_opts -ellipse -center 10 0 0  -width 150 100 100 $tempdir/ellipse_2.mnc

make_phantom $object_opts -ellipse -center 0 -10 0 -width 100 150 100 $tempdir/ellipse_3.mnc
make_phantom $object_opts -ellipse -center 0 10 0  -width 100 150 100 $tempdir/ellipse_4.mnc

make_phantom $object_opts -ellipse -center 0 0 -10 -width 100 100 150 $tempdir/ellipse_5.mnc
make_phantom $object_opts -ellipse -center 0 0 10  -width 100 100 150 $tempdir/ellipse_6.mnc


for i in $(seq 0 6);do
  fast_blur --fwhm 8 $tempdir/ellipse_$i.mnc $tempdir/ellipse_${i}_blur.mnc
done

# make segmentations
./dumb_segment.py $tempdir/ellipse_0.mnc $tempdir/ellipse_0_seg.mnc --center 0,0,0
./dumb_segment.py $tempdir/ellipse_1.mnc $tempdir/ellipse_1_seg.mnc --center " -10,0,0"
./dumb_segment.py $tempdir/ellipse_2.mnc $tempdir/ellipse_2_seg.mnc --center 10,0,0
./dumb_segment.py $tempdir/ellipse_3.mnc $tempdir/ellipse_3_seg.mnc --center 0,-10,0
./dumb_segment.py $tempdir/ellipse_4.mnc $tempdir/ellipse_4_seg.mnc --center 0,10,0
./dumb_segment.py $tempdir/ellipse_5.mnc $tempdir/ellipse_5_seg.mnc --center 0,0,-10
./dumb_segment.py $tempdir/ellipse_6.mnc $tempdir/ellipse_6_seg.mnc --center 0,0,10


# create reference mask
itk_morph --threshold 10 --exp 'D[2]' $tempdir/ellipse_0_blur.mnc $tempdir/ellipse_0_mask.mnc

rm -f seg_subjects.lst

for i in $(seq 0 6);do
  echo $tempdir/ellipse_${i}_blur.mnc,$tempdir/ellipse_${i}_seg.mnc >> seg_subjects.lst
done


