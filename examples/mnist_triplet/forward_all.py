#!/usr/bin/python2

import numpy as np
import argparse
import os
import sys

import caffe
import leveldb


def forwardData(model, weights, leveldb_path, output, limit):
    net = caffe.Net(model, weights, caffe.TEST)
    caffe.set_mode_cpu()

    if limit != -1:
        print("Limit is {}".format(limit))

    db = leveldb.LevelDB(leveldb_path)
    labels = []
    digits = []
    for key, value in db.RangeIter():
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        labels.append(int(datum.label))

        digit = caffe.io.datum_to_array(datum)
        digit = digit.astype(np.uint8)
        digits.append(digit)
        if len(digits) == limit:
            break

    print("Read leveldb\nForwarding...")
    out = net.forward_all(data=np.asarray(digits))
    print("Saving to file")
    feat = out['feat']
    items = feat.shape[0]
    dim = feat.shape[1]
    print("items and dim:", items, dim)
    output = os.path.join(os.path.split(leveldb_path)[0], output + ("_" + str(limit) if limit != -1 else "") + ".vecs")
    with open(output, 'w') as out:
        for i in xrange(items):
            out.write(str(labels[i]) + " ")
            for j in range(dim):
                out.write("{: f} ".format(feat[i][j]))
            out.write("\n")
    print("Finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--leveldb', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--limit', type=int, required=False, default=-1)
    args = parser.parse_args()
    forwardData(args.model, args.weights, args.leveldb, args.output, args.limit)

'''--model
examples/mnist_triplet/mnist_triplet.prototxt
--weights
examples/mnist_triplet/mnist_triplet_iter_1408.caffemodel
--leveldb
examples/mnist_triplet/mnist_testset
--output
mnist_testset_10000.vecs'''
