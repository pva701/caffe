#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=../../../build/examples/openface
DATA=../../../data/mnist

echo "Creating leveldb..."

rm -rf ./data/train_leveldb
rm -rf ./data/test_leveldb

$EXAMPLES/gen_mnist_triplet \
    $DATA/train-images-idx3-ubyte \
    $DATA/train-labels-idx1-ubyte \
    ./data/train_leveldb $1 $2 4

$EXAMPLES/gen_mnist_triplet \
    $DATA/t10k-images-idx3-ubyte \
    $DATA/t10k-labels-idx1-ubyte \
    ./data/test_leveldb -1 -1 4

echo "Done."
