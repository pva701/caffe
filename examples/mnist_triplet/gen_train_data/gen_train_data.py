#!/usr/bin/env python3

__author__ = 'pva701'

import argparse
import subprocess
import os
import sys

import PIL
from PIL import Image

BASE_HEIGHT = 360

def resize(file):
    frame = Image.open(file)
    if frame.size[1] >= BASE_HEIGHT:
        wpercent = (BASE_HEIGHT / float(frame.size[1]))
        wsize = int((float(frame.size[0]) * float(wpercent)))
        frame = frame.resize((wsize, BASE_HEIGHT), PIL.Image.ANTIALIAS)
        return frame
    return frame


def runProcess(args, supressOut=False):
    try:
        if supressOut:
            subprocess.check_call(args, stdout=subprocess.PIPE)
        else:
            subprocess.check_call(args)
    except subprocess.CalledProcessError as exitcode:
        print("Script " + args[1] + " is crushed")
        sys.exit(0)

def appendCache(cacheName, stage):
    with open(cacheName, 'r') as cacheIn:
        cache = cacheIn.readlines()
    cache.append(stage)
    with open(cacheName, 'w') as cacheOut:
        cacheOut.writelines(cache)

def removeCache(cacheName):
    os.remove(cacheName)

def contains(cacheName, stage):
    if os.path.exists(cacheName):
        with open(cacheName, 'r') as cacheIn:
            return stage in cacheIn.readlines()
    else:
        open(cacheName, 'w')
        return False

CACHE_NAME = None
#run filter or resizing
def runAction(stage, action):
    if contains(CACHE_NAME, stage + "\n"):
        print(stage + " from cache")
    else:
        print(stage)
        action()
        appendCache(CACHE_NAME, stage + "\n")

def resizeAllImages(fullPathDir):
    j = 0
    percent = 0
    files = os.listdir(fullPathDir)
    n = len(files)
    for file in files:
        resize(os.path.join(fullPathDir, file))
        j += 1
        if 100 * j > percent * n:
            print('Resizing {0}%'.format(percent), end='\r')
            percent += 10
    print("Resizing 100%")

def main(args):
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    videoName = os.path.basename(args.video).split('.')[0]
    fullPathDir = os.path.join(args.output, videoName)
    global CACHE_NAME
    CACHE_NAME = os.path.join(args.output, "." + videoName + ".cache")

    if args.force:
        removeCache(CACHE_NAME)

    runAction("Extracting", lambda: runProcess(["python3", "extract.py", "build", "-v", args.video, "-o", args.output]))

    if not args.noresize:
        runAction("Resizing", lambda: resizeAllImages(fullPathDir))

    runAction("Aligning", lambda: runProcess(["python3", "straight_filter.py", "-dir", fullPathDir, "-o", args.output]))

    hideVectors = os.path.join(args.output, "." + videoName + "_vectors")
    straightedFrames = os.path.join(args.output, videoName + "_straighted")
    runAction("Calculating-Vectors",  lambda: runProcess(["python2", "calc_vectors.py", "--modelDir", args.modelDir,
                                                  "--directory", straightedFrames, "--outputdir", hideVectors]))

    topFrames = straightedFrames + "_top"
    runAction("Deduplicating", lambda: runProcess(["python3", "deduplicate.py", "--directory", hideVectors,
                                                   "--inputdir", straightedFrames, "--outputdir", topFrames]))

    absOutDir = os.path.abspath(args.output)
    runAction("Filter-report", lambda: runProcess(["python3", "./filter-reporter/filter-reporter.py",
                                                         "--dir", absOutDir, "--outDir", absOutDir, videoName]))
    print("All filter successfully done. Filter report for {0} successfully created and contains in {1}".format(videoName, args.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Path to input directory", default=".")
    parser.add_argument("output", type=str, default="./output", help="Path to output directory")
    parser.add_argument("-o", "--output", type=str, help="Output directory", default="./output")

    parser.add_argument("-n", "--resize", action="store_true", default=False, help="Flag for disable resizing image")

    main(parser.parse_args())
