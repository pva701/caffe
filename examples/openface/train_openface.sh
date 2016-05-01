#!/usr/bin/env sh

TOOLS=../../build/tools

$TOOLS/caffe train --solver=./quick_solver.prototxt
