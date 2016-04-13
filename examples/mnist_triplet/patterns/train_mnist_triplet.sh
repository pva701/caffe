#!/usr/bin/env sh

TOOLS=../../build/tools

$TOOLS/caffe train --solver=./{dir_name}/solver.prototxt
