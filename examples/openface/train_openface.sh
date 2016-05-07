#!/usr/bin/env bash

TOOLS=../../build/tools

$TOOLS/caffe train --solver=./quick_solver.prototxt
