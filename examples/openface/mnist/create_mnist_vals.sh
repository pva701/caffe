#!/usr/bin/env bash

TOOL=../../../build/examples/mnist/convert_to_leveldb2
DATA=../../../data/mnist

rm -rf ../data/val_trainset
rm -rf ../data/val_testset

$TOOL $DATA/train-images-idx3-ubyte $DATA/train-labels-idx1-ubyte ../data/val_trainset 4

$TOOL $DATA/t10k-images-idx3-ubyte $DATA/t10k-labels-idx1-ubyte ../data/val_testset 4
