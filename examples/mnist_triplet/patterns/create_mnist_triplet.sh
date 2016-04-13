#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=../../build/examples/mnist_triplet
DATA=../../data/mnist

echo "Creating leveldb..."

rm -rf ./{dir_name}/train_leveldb
rm -rf ./{dir_name}/test_leveldb

$EXAMPLES/convert_mnist_triplet_data2 \
    $DATA/train-images-idx3-ubyte \
    $DATA/train-labels-idx1-ubyte \
    ./{dir_name}/train_leveldb {dig1} {dig2}

$EXAMPLES/convert_mnist_triplet_data2 \
    $DATA/t10k-images-idx3-ubyte \
    $DATA/t10k-labels-idx1-ubyte \
    ./{dir_name}/test_leveldb -1 -1

echo "Done."
