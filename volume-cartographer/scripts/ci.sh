#!/usr/bin/env bash
# Canonical CI driver. Same script runs in GHA, on dev, on EC2.
#
# Usage:
#   ci.sh                                          # all (default for dev/EC2)
#   ci.sh all                                      # full matrix + coverage
#   ci.sh builder <image>                          # build a builder docker image
#   ci.sh test <image> <compiler> <preset>         # configure + build + test
#   ci.sh coverage [image]                         # run coverage report job
#
# In GitHub Actions ($GITHUB_ACTIONS=true) the builder step uses GHA buildx
# layer cache; locally it falls back to the local docker layer cache.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGES=(ubuntu-24.04 ubuntu-26.04 alpine-edge)
DOCKERFILES=(ubuntu-24.04-noble.Dockerfile ubuntu-26.04.Dockerfile alpine.Dockerfile)
COMPILERS=(gcc clang)
PRESETS=(ci-tests ci-asan-ubsan)

dockerfile_for() {
    local image=$1
    for i in "${!IMAGES[@]}"; do
        if [[ "${IMAGES[$i]}" == "$image" ]]; then
            echo "${DOCKERFILES[$i]}"
            return
        fi
    done
    echo "Unknown image: $image (valid: ${IMAGES[*]})" >&2
    return 1
}

run_in_builder() {
    local image=$1; shift
    docker run --rm -v "$PWD:/src" -w /src "vc-builder:$image" bash -c "$*"
}

cmd_builder() {
    local image=$1
    local dockerfile
    dockerfile="$(dockerfile_for "$image")"

    local cache_args=()
    if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
        cache_args+=(
            --cache-from "type=gha,scope=vc-builder-$image"
            --cache-to "type=gha,mode=max,scope=vc-builder-$image"
        )
    fi

    docker buildx build \
        --target builder \
        --tag "vc-builder:$image" \
        --file "$dockerfile" \
        --load \
        "${cache_args[@]}" \
        .
}

cmd_test() {
    local image=$1 compiler=$2 preset=$3
    run_in_builder "$image" "
        cmake --preset $preset-$compiler &&
        cmake --build --preset $preset-$compiler &&
        ctest --preset $preset-$compiler"
}

cmd_coverage() {
    local image=${1:-ubuntu-24.04}
    run_in_builder "$image" "
        cmake --preset ci-coverage-gcc &&
        cmake --build --preset ci-coverage-gcc &&
        ctest --preset ci-coverage-gcc &&
        mkdir -p coverage &&
        gcovr --root . \
          --filter '^core/' --filter '^apps/' --filter '^utils/' \
          --exclude '.*/_deps/.*' --exclude 'build/.*' --exclude 'libs/.*' \
          --gcov-ignore-errors=no_working_dir_found \
          --gcov-ignore-parse-errors=negative_hits.warn_once_per_file \
          --html-details coverage/index.html \
          --cobertura coverage/cobertura.xml \
          --txt coverage/summary.txt \
          build/ci-coverage-gcc &&
        find . -name '*.gcov' -not -path './coverage/*' -delete"
}

cmd_all() {
    for image in "${IMAGES[@]}"; do
        echo "=== Builder: $image ==="
        cmd_builder "$image"
    done
    for image in "${IMAGES[@]}"; do
        for compiler in "${COMPILERS[@]}"; do
            for preset in "${PRESETS[@]}"; do
                echo "=== $image: $preset-$compiler ==="
                cmd_test "$image" "$compiler" "$preset"
            done
        done
    done
    echo "=== Coverage (gcc, gcov) ==="
    cmd_coverage ubuntu-24.04

    echo
    echo "All CI passed."
    echo "Coverage HTML: $REPO_ROOT/coverage/index.html"
    tail -5 "$REPO_ROOT/coverage/summary.txt"
}

case "${1:-all}" in
    all)      cmd_all ;;
    builder)  shift; cmd_builder "$@" ;;
    test)     shift; cmd_test "$@" ;;
    coverage) shift; cmd_coverage "$@" ;;
    *)
        echo "Usage: $0 [all|builder <image>|test <image> <compiler> <preset>|coverage [image]]" >&2
        exit 1
        ;;
esac
