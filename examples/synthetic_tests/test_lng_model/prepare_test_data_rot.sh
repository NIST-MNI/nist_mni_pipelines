#! /bin/sh

object_opts="-short -real_range 0 100 -background 0 -edge_value 100 -fill_value 100  -nelements 80 80 80 -step 1 1 1 -start -40 -40 -40"

# make growing "object"
# 1st group: body growth 4% each step, hand growth 8%
# 2nd group: body growth 8% each step, hand growth 8%

tempdir=`mktemp -t testXXXX -d`
trap "rm -rf $tempdir" 0 1 2 15
out=data_rot
mkdir -p data_rot

rm -f subjects_rot.lst

main_dim_1=10
main_dim_2=10

handle_width=20
handle_height=5

make_phantom $object_opts -ellipse -center  0 0 0   -width ${main_dim_1} ${main_dim_1} ${main_dim_1}  $tempdir/ellipse_1.mnc -clob
make_phantom $object_opts -ellipse -center  10 0 0  -width ${handle_width} ${handle_height} ${handle_height}  $tempdir/ellipse_2.mnc -clob
make_phantom $object_opts -ellipse -center  0 0 0   -width ${main_dim_2} ${main_dim_2} ${main_dim_2}  $tempdir/ellipse_3.mnc -clob

mincmath -max  $tempdir/ellipse_1.mnc $tempdir/ellipse_2.mnc $tempdir/object_0_.mnc -clob
minccalc -express 'A[0]+1.0' $tempdir/object_0_.mnc $tempdir/object_0.mnc -clob
itk_morph    --threshold 4 --exp 'D[2]' $tempdir/object_0.mnc $tempdir/mask_0.mnc --clob

mincmath -max  $tempdir/ellipse_3.mnc $tempdir/ellipse_2.mnc $tempdir/object_1.mnc -clob
itk_morph --threshold 4 --exp 'D[2]' $tempdir/object_1.mnc $tempdir/mask_1.mnc --clob


for i in $(seq 0 8);do

  param2xfm -rotations 0 0 $(($i*10-40)) $tempdir/rot_$i.xfm
  
  itk_resample --transform $tempdir/rot_$i.xfm $tempdir/object_0.mnc ${out}/object_0_$i.mnc --clob 
  itk_resample --transform $tempdir/rot_$i.xfm $tempdir/mask_0.mnc   ${out}/mask_0_$i.mnc --byte --labels --like ${out}/object_0_$i.mnc --clob

  echo ${out}/object_0_$i.mnc,${out}/mask_0_$i.mnc,1.0,1.0,$(($i-4)) >> subjects_rot.lst
  
done

cut -d , -f 1,2 subjects_rot.lst > subjects_rot_cut.lst
