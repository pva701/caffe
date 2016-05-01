__author__ = 'pva701'


#!/usr/bin/env python3

import argparse
import os
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Path to input directory")
    parser.add_argument("output", type=str, help="Path to output directory")

    parser.add_argument('--numImagesThreshold', type=int,
                        help="Delete directories with less than this many images.",
                        default=2)
    args = parser.parse_args()

    exts = ["jpg", "png"]

    for subdir, dirs, files in os.walk(args.input):
        if subdir == args.input:
            continue
        nImgs = 0
        imageClass = os.path.basename(subdir)
        for fName in files:
            if any(fName.lower().endswith("." + ext) for ext in exts):
                nImgs += 1
        if nImgs >= args.numImagesThreshold:
            print("Copying {}".format(subdir))
            dest_dir = os.path.join(args.output, imageClass)
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            shutil.copytree(subdir, dest_dir)