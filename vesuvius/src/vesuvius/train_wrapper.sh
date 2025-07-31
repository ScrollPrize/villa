#!/bin/bash
# Wrapper script for vesuvius.train that sets LD_PRELOAD to use system libstdc++
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
exec vesuvius.train "$@"