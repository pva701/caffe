#!/usr/bin/env bash

#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

OPENFACE_DATA=data
TOOLS=../../build/examples/openface

DATA_ROOT=/home/pva701/lfw_aligned

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

rm -rf $OPENFACE_DATA/train_leveldb
rm -rf $OPENFACE_DATA/test_leveldb

echo "Creating train leveldb..."

$TOOLS/gen_dataset_from_images --backend=leveldb $DATA_ROOT/ $DATA_ROOT/train.txt $OPENFACE_DATA/train_leveldb

echo "Creating test leveldb..."

$TOOLS/gen_dataset_from_images --backend=leveldb $DATA_ROOT/ $DATA_ROOT/val.txt $OPENFACE_DATA/test_leveldb
    
echo "Done."
