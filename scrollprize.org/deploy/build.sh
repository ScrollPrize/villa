#!/bin/bash

set -e

VERSION=$1
DIR=$(readlink -f $(dirname $0))

docker build $DIR/.. -f $DIR/Dockerfile \
   --progress plain --builder kube-sp --platform linux/amd64,linux/arm64 --push \
   --tag 585768151128.dkr.ecr.us-east-1.amazonaws.com/scrollprize/scrollprize-org:$VERSION

git tag -a docker/scrollprize-org/$VERSION -m "Built and published scrollprize-org/$VERSION to docker"
