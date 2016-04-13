#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.
pwd

EXAMPLES=./build/examples/mnist_triplet
DATA=./data/mnist

echo "Creating leveldb..."

rm -rf ./examples/mnist_triplet/mnist_trainset
rm -rf ./examples/mnist_triplet/mnist_testset

$EXAMPLES/convert_to_leveldb \
    $DATA/train-images-idx3-ubyte \
    $DATA/train-labels-idx1-ubyte \
    ./examples/mnist_triplet/mnist_trainset

$EXAMPLES/convert_to_leveldb \
    $DATA/t10k-images-idx3-ubyte \
    $DATA/t10k-labels-idx1-ubyte \
    ./examples/mnist_triplet/mnist_testset

echo "Done."
