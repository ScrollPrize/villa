#!/bin/bash

VERSION=$1
DIR=$(readlink -f $(dirname $0))

docker build $DIR \
    --progress plain --builder kube --platform linux/amd64 --target gpu --push \
    --tag 585768151128.dkr.ecr.us-east-1.amazonaws.com/scrollprize/ink-detection:$VERSION

docker build $DIR \
    --progress plain --builder kube --platform linux/amd64,linux/arm64 --target cpu --push \
    --tag 585768151128.dkr.ecr.us-east-1.amazonaws.com/scrollprize/ink-detection:$VERSION-cpu
