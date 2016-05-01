set -e
./forward_all.py --model openface_test.prototxt --weights data/snapshots_iter_$1.caffemodel --leveldb data/val_trainset --output trainset --limit 1000

./forward_all.py --model openface_test.prototxt --weights data/snapshots_iter_$1.caffemodel --leveldb data/val_testset --output testset
