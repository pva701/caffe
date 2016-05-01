#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=../../../build/examples/mnist_triplet
DATA=../../../data/mnist

echo "Creating leveldb..."

rm -rf ./data/train_leveldb
rm -rf ./data/test_leveldb

$EXAMPLES/convert_mnist_triplet_data2 \
    $DATA/train-images-idx3-ubyte \
    $DATA/train-labels-idx1-ubyte \
    ./data/train_leveldb $1 $2

$EXAMPLES/convert_mnist_triplet_data2 \
    $DATA/t10k-images-idx3-ubyte \
    $DATA/t10k-labels-idx1-ubyte \
    ./data/test_leveldb -1 -1

echo "Done."
