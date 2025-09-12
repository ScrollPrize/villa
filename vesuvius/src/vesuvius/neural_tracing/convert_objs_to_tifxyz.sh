#!/usr/bin/env bash

path=/home/paul/projects/vesuvius-scrolls/neural-tracing/data/PHerc1667
#path=/home/paul/projects/vesuvius-scrolls/neural-tracing/data/PHercParis4

for f in $path/obj/* ; do
  /home/paul/projects/vesuvius-scrolls/villa/volume-cartographer/cmake-build-relwithdebinfo/bin/vc_obj2tifxyz_legacy $f/$(basename $f).obj $path/tifxyz/$(basename $f) 1000 1 10
done
