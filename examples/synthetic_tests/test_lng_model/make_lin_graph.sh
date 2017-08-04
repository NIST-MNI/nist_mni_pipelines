#! /bin/sh

tempdir=`mktemp -t testXXXX -d`
trap "rm -rf $tempdir" 0 1 2 15

rm -f $tempdir/lst

for i in $(seq 0 8);do
  
  mincpik --image_range 0 100 data_lin/object_0_${i}.mnc $tempdir/object_0_${i}.miff
  
  echo $tempdir/object_0_${i}.miff >> $tempdir/lst
  echo $i
  
  for it in $(seq 2 20);do
    printf -v in "tmp_regress_LCC_lin_4mm/%d/object_0_%d.mnc_int_approx.%03d.mnc" $it $i ${it}
    mincpik --image_range 0 100 $in $tempdir/${it}_${i}.miff
    convert -shave 4x4 $tempdir/${it}_${i}.miff $tempdir/${it}_${i}.miff
    echo $tempdir/${it}_${i}.miff >> $tempdir/lst
    echo $it $i
  done
done

echo "-label Inp null:" >> $tempdir/lst

for it in $(seq -w 2 20);do
  mincpik --image_range 0 20 tmp_regress_LCC_lin_4mm/model_intensity.0${it}_RMS.mnc $tempdir/${it}_RMS.miff
  echo "-label $it $tempdir/${it}_RMS.miff" >> $tempdir/lst
  echo $it
done


montage -geometry 152x152+1+1 -background black -fill white -tile 20x10 -pointsize 40 $(cat $tempdir/lst) \
        lin_progression.png
