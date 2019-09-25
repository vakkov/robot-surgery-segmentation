#!/bin/bash

#for i in 3
for i in 0 1 2 3
do
#    python train.py \
#        --device-ids 0,1 \
#        --batch-size 22\
#        --fold $i \
#        --workers 35 \
#        --lr 0.0001 \
#        --n-epochs 10 \
#        --jaccard-weight 0.3 \
#        --model RasTerNetV2 \
#        --train_crop_height 512 \
#        --train_crop_width 640 \
#        --val_crop_height 512 \
#        --val_crop_width 640 \
#        --type parts
    python train.py \
        --device-ids 0,1 \
        --batch-size 22 \
        --fold $i \
        --workers 35 \
        --lr 0.00001 \
        --n-epochs 25 \
        --jaccard-weight 0.3 \
        --model RasTerNetV2 \
        --train_crop_height 512 \
        --train_crop_width 640 \
        --val_crop_height 512 \
        --val_crop_width 640 \
        --type parts
done
