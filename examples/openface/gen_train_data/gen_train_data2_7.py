#!/usr/bin/env python2

__author__ = 'pva701'

import argparse
import os
from multiprocessing import Pool as ThreadPool

from skimage import io

from openface.data import iterImgs
from openface.alignment import NaiveDlib
import cv2

def alignImage(bundle):
    (args, img_object, aligner) = bundle

    out_dir = os.path.join(args.output, img_object.cls)
    dest_file = os.path.join(out_dir, img_object.name)
    if os.path.exists(dest_file):
        return

    rgb = img_object.getRGB(cache=False)
    #rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)

    aligned_image = aligner.alignImg(args.method, args.size, rgb)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    if aligned_image is not None:
        io.imsave(dest_file, aligned_image)


def detectAlignImages(args):
    pool = ThreadPool(5)
    imgs = list(iterImgs(args.input))
    imgs.sort(key=lambda x: x.path)
    print("All images is listed")

    aligner = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor)
    bundles = []
    current_class = None
    class_num = -1
    with open(os.path.join(args.output, 'description.txt'), 'w') as descr:
        for imgObject in imgs:
            if imgObject.cls != current_class:
                class_num += 1
                current_class = imgObject.cls
            descr.write(os.path.join(imgObject.cls, imgObject.name) + " " + str(class_num) + "\n")
            bundles.append((args, imgObject, aligner))
    print("Description is generated")
    pool.map(alignImage, bundles)

    pool.close()
    pool.join()


def main(args):
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    args.dlibFacePredictor = 'dlib_model/shape_predictor_68_face_landmarks.dat'
    args.dlibFaceMean = 'dlib_model/mean.csv'
    detectAlignImages(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Path to input directory")
    parser.add_argument("output", type=str, help="Path to output directory")

    parser.add_argument('--method', type=str,
                                 choices=['tightcrop', 'affine', 'perspective', 'homography'], default='affine',
                                 help="Alignment method.")

    parser.add_argument('--size', type=int, help="Default image size.", default=96)

    main(parser.parse_args())
