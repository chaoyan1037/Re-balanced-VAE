#!/bin/bash


saved_model=/mnt/Data/DL/tmp/beta_vae_mol/baseline/checkpoints/150


for i in 1 2 3 4 5 6 7 8 9 10
do
  out_dir=guided_results$i
  echo "out_dir: $out_dir"
  if [ ! -e $out_dir ];
  then
      mkdir -p $out_dir
  fi

  python run_bo.py --save_dir $out_dir --random_seed $i --restore=$saved_model > $out_dir/log.txt

done

