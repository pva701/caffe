#!/usr/bin/env python3

__author__ = 'pva701'

import argparse
import os
import concurrent.futures

import PIL
from PIL import Image
from skimage import io

from openface.data import iterImgs
from openface.alignment import NaiveDlib
import cv2

BASE_HEIGHT = 360

def resizeImage(src_file, dest_file):
    frame = Image.open(src_file)
    if frame.size[1] >= BASE_HEIGHT:
        wpercent = (BASE_HEIGHT / float(frame.size[1]))
        wsize = int((float(frame.size[0]) * float(wpercent)))
        frame = frame.resize((wsize, BASE_HEIGHT), PIL.Image.ANTIALIAS)
    frame.save(dest_file)

def alignImage(src_file, dest_file):
    pass

def resizeAllImages(input_dir, output_dir):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        person_dirs = os.listdir(input_dir)
        futures = []
        for dir in person_dirs:
            person_inp_dir = os.path.join(input_dir, dir)
            person_out_dir = os.path.join(output_dir, dir)
            if not os.path.exists(person_out_dir):
                os.mkdir(person_out_dir)

            for file in os.listdir(person_inp_dir):
                src_file = os.path.join(input_dir, dir, file)
                dest_file = os.path.join(output_dir, dir, file)
                future = executor.submit(resizeImage, src_file, dest_file)
                futures.append(future)
        print("All submitted")

        count_success = 0
        n = len(futures)
        #for future in concurrent.futures.as_completed(futures):
        for future in futures:
            future.result()
            count_success += 1
            if count_success % 1000 == 0:
                print("{} photo resized from {}".format(count_success, n))

def alignImage(args, img_object, aligner):
    rgb = img_object.getRGB(cache=False)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)

    out = aligner.alignImg(args.method, args.size, rgb)
    dest_image = os.path.join(args.output, img_object.cls, img_object.name)
    if out is not None:
        io.imsave(dest_image, out)


def detectAlignImages(args):#input_dir, output_dir):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        imgs = list(iterImgs(args.input))
        aligner = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor)
        futures = []

        for imgObject in imgs:
            future = executor.submit(alignImage, args, imgObject, aligner)
            futures.append(future)

        count_success = 0
        n = len(futures)
        #for future in concurrent.futures.as_completed(futures):
        for future in futures:
            future.result()
            count_success += 1
            if count_success % 1000 == 0:
                print("{} photo resized from {}".format(count_success, n))


def main(args):
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if args.resize:
        dest_resized = args.output + "_resized"
        resizeAllImages(args.input, dest_resized)
        args.input = dest_resized
    detectAlignImages(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Path to input directory", default=".")
    parser.add_argument("output", type=str, default="./output", help="Path to output directory")
    parser.add_argument("-o", "--output", type=str, help="Output directory", default="./output")

    parser.add_argument("-n", "--resize", action="store_true", default=False, help="Flag for disable resizing image")

    #main(parser.parse_args())
