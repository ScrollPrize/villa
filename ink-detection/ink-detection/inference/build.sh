#!/bin/bash

set -e

if [[ "${AGENTS_AGENT_MODE:-0}" == "1" && "${AGENTS_ALLOW_INSTALL:-0}" != "1" ]]; then
  echo "INFO: build.sh is disabled by default in agent mode."
  echo "Set AGENTS_ALLOW_INSTALL=1 to run this script."
  echo "Example: AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 ./ink-detection/optimized_inference/build.sh"
  exit 0
fi

VERSION=$1
DIR=$(readlink -f $(dirname $0))

docker build $DIR \
    --progress plain --builder kube --platform linux/amd64 --target gpu --push \
    --tag 585768151128.dkr.ecr.us-east-1.amazonaws.com/scrollprize/ink-detection:$VERSION

docker build $DIR \
    --progress plain --builder kube --platform linux/amd64,linux/arm64 --target cpu --push \
    --tag 585768151128.dkr.ecr.us-east-1.amazonaws.com/scrollprize/ink-detection:$VERSION-cpu

git tag -a docker/ink-detection/$VERSION -m "Published docker version $VERSION / ${VERSION}-cpu"
