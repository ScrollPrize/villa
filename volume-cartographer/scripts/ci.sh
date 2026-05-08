#!/usr/bin/env bash
# Canonical CI driver. Same script runs in GHA, on dev, on EC2.
#
# Usage:
#   ci.sh                                                  # all (default for dev/EC2)
#   ci.sh all                                              # full matrix + coverage
#   ci.sh builder <image>                                  # build a builder docker image
#   ci.sh test <image> <compiler> <preset>                 # configure + build + test
#   ci.sh coverage [image]                                 # coverage report (in CWD)
#   ci.sh patch-coverage <base_ref> [image]                # diff-cover gate vs base_ref
#   ci.sh coverage-regression <base_ref> [image]           # total-coverage non-regression vs base_ref
#   ci.sh dead-code [image] [compiler]                     # unused-* warnings + linker --print-gc-sections report
#
# Environment knobs:
#   PATCH_COVERAGE_MIN  minimum % required by `patch-coverage` (default 0)
#
# In GitHub Actions ($GITHUB_ACTIONS=true) the builder step uses GHA buildx
# layer cache; locally it falls back to the local docker layer cache.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGES=(ubuntu-24.04 ubuntu-26.04)
DOCKERFILES=(ubuntu-24.04-noble.Dockerfile ubuntu-26.04.Dockerfile)
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
    local image=$1 src=$2; shift 2
    docker run --rm \
        -v "$src:/src" \
        -w /src \
        "vc-builder:$image" bash -c "$*"
}

coverage_in_dir() {
    local image=$1 src=$2
    run_in_builder "$image" "$src" "
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

# Extract TOTAL line coverage % from a gcovr summary.txt.
total_coverage_pct() {
    awk '/^TOTAL/ { for (i=1;i<=NF;i++) if ($i ~ /%$/) { gsub("%","",$i); print $i; exit } }' "$1"
}

cmd_builder() {
    local image=$1
    local local_tag="vc-builder:$image"

    # Try pulling the published image from ghcr first. Skip the pull if
    # VC_BUILDER_FORCE_LOCAL=1 is set (the PR touched a Dockerfile, or
    # the user explicitly wants a from-scratch local build).
    if [[ "${VC_BUILDER_FORCE_LOCAL:-0}" != "1" ]]; then
        local owner
        owner=$(echo "${VC_BUILDER_REGISTRY_OWNER:-${GITHUB_REPOSITORY_OWNER:-scrollprize}}" | tr 'A-Z' 'a-z')
        local remote="ghcr.io/$owner/villa/volume-cartographer:builder-$image"
        if docker pull "$remote" 2>/dev/null; then
            docker tag "$remote" "$local_tag"
            return 0
        fi
        echo "ci.sh: ghcr pull of $remote failed; building locally" >&2
    fi

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
        --tag "$local_tag" \
        --file "$dockerfile" \
        --load \
        "${cache_args[@]}" \
        .
}

cmd_publish() {
    local image=$1
    local owner
    owner=$(echo "${VC_BUILDER_REGISTRY_OWNER:-${GITHUB_REPOSITORY_OWNER:-scrollprize}}" | tr 'A-Z' 'a-z')
    local dockerfile
    dockerfile="$(dockerfile_for "$image")"
    local repo="ghcr.io/$owner/villa/volume-cartographer"
    local sha=${GITHUB_SHA:-$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo local)}

    docker buildx build \
        --target builder \
        --tag "$repo:builder-$image" \
        --tag "$repo:builder-$image-$sha" \
        --file "$dockerfile" \
        --push \
        .
}

cmd_test() {
    local image=$1 compiler=$2 preset=$3
    run_in_builder "$image" "$REPO_ROOT" "
        cmake --preset $preset-$compiler &&
        cmake --build --preset $preset-$compiler &&
        ctest --preset $preset-$compiler"
}

cmd_compile() {
    local image=$1 compiler=$2 preset=$3
    run_in_builder "$image" "$REPO_ROOT" "
        cmake --preset $preset-$compiler &&
        cmake --build --preset $preset-$compiler"
}

cmd_coverage() {
    local image=${1:-ubuntu-24.04}
    coverage_in_dir "$image" "$REPO_ROOT"
}

cmd_patch_coverage() {
    local base_ref=$1
    # Second positional ("image") is accepted for backwards compat but ignored:
    # diff-cover only needs Python + git history, so we run it on the host.
    # When volume-cartographer lives as a subdir of a larger repo the .git is
    # outside the docker mount and diff-cover dies with "not a git repository".
    local min_pct=${PATCH_COVERAGE_MIN:-0}
    local cobertura="$REPO_ROOT/coverage/cobertura.xml"
    if [[ ! -f "$cobertura" ]]; then
        echo "patch-coverage: $cobertura missing — run 'ci.sh coverage' first" >&2
        return 1
    fi

    local git_root
    git_root=$(git -C "$REPO_ROOT" rev-parse --show-toplevel)
    git -C "$git_root" fetch --quiet origin "${base_ref#origin/}" || true

    # gcovr ran inside docker as root, so coverage/ is root-owned on the host.
    # We need to write patch.md / patch.html into it from the host runner.
    if [[ ! -w "$REPO_ROOT/coverage" ]]; then
        sudo chown -R "$(id -u):$(id -g)" "$REPO_ROOT/coverage" 2>/dev/null || true
    fi

    # gcovr emits paths relative to volume-cartographer with <source>.</source>.
    # When the git root is a parent dir, prepend the subdir so diff-cover's
    # path lookup matches what git reports as changed.
    local src_in_git fixed
    src_in_git=$(realpath --relative-to="$git_root" "$REPO_ROOT")
    fixed=$(mktemp -t cobertura-diffcov.XXXXXX.xml)
    trap 'rm -f "$fixed"' RETURN
    if [[ "$src_in_git" == "." ]]; then
        cp "$cobertura" "$fixed"
    else
        sed "s|<source>\.</source>|<source>${src_in_git}</source>|" "$cobertura" > "$fixed"
    fi

    python3 -m venv /tmp/diffcov >/dev/null
    /tmp/diffcov/bin/pip install --quiet diff-cover
    (
        cd "$git_root"
        /tmp/diffcov/bin/diff-cover "$fixed" \
            --compare-branch="$base_ref" \
            --fail-under="$min_pct" \
            --markdown-report "$REPO_ROOT/coverage/patch.md" \
            --html-report "$REPO_ROOT/coverage/patch.html"
    )
}

cmd_coverage_regression() {
    local base_ref=$1
    local image=${2:-ubuntu-24.04}
    local pr_summary="$REPO_ROOT/coverage/summary.txt"
    if [[ ! -f "$pr_summary" ]]; then
        echo "coverage-regression: $pr_summary missing — run 'ci.sh coverage' first" >&2
        return 1
    fi

    git -C "$REPO_ROOT" fetch --quiet origin "${base_ref#origin/}" || true

    local base_tree
    base_tree="$(mktemp -d)/base-tree"
    git -C "$REPO_ROOT" worktree add --detach "$base_tree" "$base_ref"
    trap "git -C '$REPO_ROOT' worktree remove --force '$base_tree' || true" RETURN

    # Base branch may not have the ci-coverage-gcc preset (e.g. before this
    # CI lands). In that case, skip the regression gate with a warning rather
    # than failing — there's no meaningful "base coverage" to compare to.
    if ! grep -q '"name": "ci-coverage-gcc"' "$base_tree/CMakePresets.json" 2>/dev/null; then
        echo "::warning::base ($base_ref) has no ci-coverage-gcc preset; skipping non-regression gate"
        return 0
    fi

    coverage_in_dir "$image" "$base_tree"

    local pr_cov base_cov
    pr_cov=$(total_coverage_pct "$pr_summary")
    base_cov=$(total_coverage_pct "$base_tree/coverage/summary.txt")
    echo "Total coverage — base ($base_ref): ${base_cov}%, PR head: ${pr_cov}%"
    if awk -v p="$pr_cov" -v b="$base_cov" 'BEGIN { exit !(p+0 < b+0) }'; then
        echo "::error::Coverage regressed: ${pr_cov}% < ${base_cov}%" >&2
        return 1
    fi
}

cmd_dead_code() {
    local image=${1:-ubuntu-24.04}
    local compiler=${2:-clang}
    mkdir -p "$REPO_ROOT/dead-code"

    # Build inside the container; capture full build log for compile-warning
    # extraction. Then run nm-based analysis (approach B): symbols defined
    # somewhere in our source .o files but absent from every final binary.
    run_in_builder "$image" "$REPO_ROOT" "
        rm -rf build/ci-dead-code-$compiler &&
        cmake --preset ci-dead-code-$compiler &&
        cmake --build --preset ci-dead-code-$compiler 2>&1 | tee dead-code/build.log
        scripts/dead-code-analysis.sh build/ci-dead-code-$compiler"
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
    all)                  cmd_all ;;
    builder)              shift; cmd_builder "$@" ;;
    test)                 shift; cmd_test "$@" ;;
    compile)              shift; cmd_compile "$@" ;;
    coverage)             shift; cmd_coverage "$@" ;;
    patch-coverage)       shift; cmd_patch_coverage "$@" ;;
    coverage-regression)  shift; cmd_coverage_regression "$@" ;;
    dead-code)            shift; cmd_dead_code "$@" ;;
    publish)              shift; cmd_publish "$@" ;;
    *)
        cat >&2 <<EOF
Usage: $0 [all
          | builder <image>
          | publish <image>
          | test <image> <compiler> <preset>
          | compile <image> <compiler> <preset>
          | coverage [image]
          | patch-coverage <base_ref> [image]
          | coverage-regression <base_ref> [image]
          | dead-code [image] [compiler]]
EOF
        exit 1
        ;;
esac
