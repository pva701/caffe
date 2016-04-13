#!/usr/bin/python3

__author__ = 'pva701'
import os, stat
import sys
import argparse
import shutil


def inflate_patterns(dir_name, dig1, dig2, margin, noutputs):
    os.mkdir(dir_name)
    for pattern in os.listdir('patterns'):
        print('Generating {0}/{1}'.format(dir_name, pattern))

        file = os.path.join('patterns', pattern)
        with open(file, 'r') as inp:
            with open(os.path.join(dir_name, pattern), 'w') as out:
                out.write(inp.read().format(dir_name=dir_name, dig1=dig1, dig2=dig2, margin=margin, noutputs=noutputs))
            if pattern.endswith(".sh"):
                os.chmod(os.path.join(dir_name, pattern), stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)


def main(args):
    if args.directory is None:
        dirs = os.listdir('.')
        mx_num = 0
        for it in dirs:
            if it.startswith('nn_config'):
                num = int(it[len('nn_config'):])
                if num > mx_num:
                    mx_num = num
        dir_name = 'nn_config' + str(mx_num + 1)
    elif os.path.exists(args.directory):
        ans = input('This directory already exists? Replace? [y/N] ')
        if ans == '':
            print("Configuration isn't created")
            sys.exit(0)
        else:
            if ans == 'y' or ans == 'Y':
                shutil.rmtree(args.directory)
                dir_name = args.directory
            else:
                print("Invalid answer")
                sys.exit(0)
    else:
        dir_name = args.directory
    print('Generating NN in directory ' + dir_name)
    inflate_patterns(dir_name, args.digit1, args.digit2, args.margin, args.noutputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool for generate NN configuration.')
    parser.add_argument('-d1', '--digit1', action='store', type=int, required=True, help='First digit')
    parser.add_argument('-d2', '--digit2', action='store', type=int, required=True, help='Second digit')
    parser.add_argument('-m', '--margin', action='store', type=str, required=True, help='Margin for triplet loss layer')
    parser.add_argument('-n', '--noutputs', action='store', default=8, type=int, help='Num outputs in NN')
    parser.add_argument('-d', '--directory', action='store', type=str, required=False,
                        help='Directory for saving NN config')
    main(parser.parse_args())

'''path = '/home/pva701/github/caffe_triplet/caffe/examples/mnist_triplet/patterns/mnist_triplet_train_test.prototxt'
with open(path, 'r') as patt:
    s = patt.read()
    s = s.replace('{', '{{').replace('}', '}}')

with open(path, 'w') as patt:
    patt.write(s)'''
