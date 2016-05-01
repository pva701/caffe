#!/usr/bin/env bash

#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

OPENFACE_DATA=examples/openface/data
TOOLS=build/examples/openface

TRAIN_DATA_ROOT=~/lfw_aligned
VAL_DATA_ROOT=~/lfw_aligned

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train leveldb..."

GLOG_logtostderr=1 $TOOLS/gen_dataset_from_images \
    --backend=leveldb
    $TRAIN_DATA_ROOT \
    $TRAIN_DATA_ROOT/train.txt \
    $OPENFACE_DATA/train_leveldb

echo "Creating test leveldb..."

GLOG_logtostderr=1 $TOOLS/gen_dataset_from_images \
    --backend=leveldb
    $VAL_DATA_ROOT \
    $VAL_DATA_ROOT/test.txt \
    $OPENFACE_DATA/test_leveldb

echo "Done."
