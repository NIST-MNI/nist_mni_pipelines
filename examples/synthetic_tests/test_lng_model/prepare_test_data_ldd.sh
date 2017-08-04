#! /bin/sh

object_opts="-short -real_range 0 100 -background 0 -edge_value 100 -fill_value 100  -nelements 100 100 100 -step 1 1 1 -start -50 -50 -50"

tempdir=`mktemp -t testXXXX -d`
trap "rm -rf $tempdir" 0 1 2 15
out=data_ldd
mkdir -p $out

rm -f subjects_ldd.lst

for i in $(seq 0 8);do

  main_dim_1=50
  main_dim_2=30
  pos=$(echo "40-${i}*2"|bc -l)
  
  make_phantom $object_opts -ellipse -center  0 0 0      -width ${main_dim_1} ${main_dim_1} ${main_dim_1}  $tempdir/ellipse_1.mnc -clob
  make_phantom $object_opts -ellipse -center  $pos 0 0   -width ${main_dim_2} ${main_dim_2} ${main_dim_2}  $tempdir/ellipse_2.mnc -clob
  
  minccalc -express 'clamp(A[0]-A[1],0,100)' $tempdir/ellipse_1.mnc $tempdir/ellipse_2.mnc $tempdir/object_0_$i.mnc -clob
  
  itk_morph --threshold 50 $tempdir/object_0_$i.mnc     $tempdir/object_0_${i}_b.mnc
  itk_distance --signed    $tempdir/object_0_${i}_b.mnc $tempdir/object_0_${i}_bd.mnc
  mincresample -nearest -like $tempdir/object_0_$i.mnc $tempdir/object_0_${i}_bd.mnc $tempdir/object_0_${i}_bd_.mnc -clob
  minccalc -express '(A[0]<0?sin(A[0]*3.14/3)*10:exp(-A[0]/4))*10+A[1]+1.0' $tempdir/object_0_${i}_bd_.mnc $tempdir/object_0_$i.mnc ${out}/object_0_${i}.mnc


  itk_morph --exp 'D[3]' $tempdir/object_0_${i}_b.mnc $tempdir/mask_0_$i.mnc --clob
  mincresample -nearest -like ${out}/object_0_$i.mnc $tempdir/mask_0_$i.mnc ${out}/mask_0_$i.mnc -clob
  
  echo ${out}/object_0_$i.mnc,${out}/mask_0_$i.mnc,1.0,1.0,$i >> subjects_ldd.lst
  #exit
done

cut -d , -f 1,2 subjects_ldd.lst > subjects_ldd_cut.lst
