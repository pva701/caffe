#!/usr/bin/env bash

TOOL=../../build/tools/convert_imageset

DATA_ROOT=/home/pva701/lfw_aligned/

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

rm -rf data/val_trainset
rm -rf data/val_testset

$TOOL --backend=leveldb $DATA_ROOT $DATA_ROOT/train.txt data/val_trainset
$TOOL --backend=leveldb $DATA_ROOT $DATA_ROOT/val.txt data/val_testset


