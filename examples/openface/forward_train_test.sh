set -e
./forward_all.py --model openface_test.prototxt --weights data/snapshots_iter_$1.caffemodel --leveldb data/val_trainset96 --output trainset96

./forward_all.py --model openface_test.prototxt --weights data/snapshots_iter_$1.caffemodel --leveldb data/val_testset96 --output mnist_testset96
