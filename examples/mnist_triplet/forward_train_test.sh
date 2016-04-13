set -e
./forward_all.py --model $1/mnist_triplet.prototxt --weights $1/mnist_triplet_iter_$2.caffemodel --leveldb mnist_trainset --output $1/mnist_trainset --limit 20000

./forward_all.py --model $1/mnist_triplet.prototxt --weights $1/mnist_triplet_iter_$2.caffemodel --leveldb mnist_testset --output $1/mnist_testset
