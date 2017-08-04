#! /bin/sh

object_opts="-short -real_range 0 100 -background 0 -edge_value 100 -fill_value 100  -nelements 80 80 80 -step 1 1 1 -start -40 -40 -40"

# make growing "object"
# 1st group: body growth 4% each step, hand growth 8%
# 2nd group: body growth 8% each step, hand growth 8%

tempdir=`mktemp -t testXXXX -d`
trap "rm -rf $tempdir" 0 1 2 15
out=data_lin
mkdir -p data_lin

rm -f subjects.lst

for i in $(seq 0 8);do

  main_dim_1=$(echo "10+${i}"|bc -l)
  main_dim_2=$(echo "10+${i}/2"|bc -l)
  
  handle_width=$(echo "20+${i}*2"|bc -l)
  handle_height=$(echo "5+${i}"|bc -l)

  make_phantom $object_opts -ellipse -center  0 0 0   -width ${main_dim_1} ${main_dim_1} ${main_dim_1}  $tempdir/ellipse_1.mnc -clob
  make_phantom $object_opts -ellipse -center  10 0 0  -width ${handle_width} ${handle_height} ${handle_height}  $tempdir/ellipse_2.mnc -clob
  make_phantom $object_opts -ellipse -center  0 0 0   -width ${main_dim_2} ${main_dim_2} ${main_dim_2}  $tempdir/ellipse_3.mnc -clob
  
  mincmath -max  $tempdir/ellipse_1.mnc $tempdir/ellipse_2.mnc ${out}/object_0_$i.mnc -clob
  itk_morph --threshold 1 --exp 'D[2]' ${out}/object_0_$i.mnc $tempdir/mask_0_$i.mnc --clob
  mincresample -nearest -like ${out}/object_0_$i.mnc $tempdir/mask_0_$i.mnc ${out}/mask_0_$i.mnc
  
  mincmath -max  $tempdir/ellipse_3.mnc $tempdir/ellipse_2.mnc ${out}/object_1_$i.mnc -clob
  itk_morph --threshold 1 --exp 'D[2]' ${out}/object_1_$i.mnc $tempdir/mask_1_$i.mnc --clob
  mincresample -nearest -like ${out}/object_1_$i.mnc $tempdir/mask_1_$i.mnc ${out}/mask_1_$i.mnc

  echo ${out}/object_0_$i.mnc,${out}/mask_0_$i.mnc,1.0,1.0,0,$i >> subjects_lin.lst
  echo ${out}/object_1_$i.mnc,${out}/mask_1_$i.mnc,1.0,1.0,1,$i >> subjects_lin.lst

done

cut -d , -f 1,2 subjects_lin.lst > subjects_lin_cut.lst
