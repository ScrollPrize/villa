#!/usr/bin/env bash
# grow_strips.sh MODE STEP : sheet strips of store fibers with gap/damaged spans, as patches for the spiral fit.
#   MODE  pieces = one strip per valid piece;  mirror = join close pieces, mirror the ends, trace, cut back per piece
#   STEP  fibers | trace | split | patches | all      (split does nothing in pieces mode; trace needs a GPU)
# Run it with the vesuvius environment active (or PY set to its python). Required environment:
#   SOURCE     fiber store (.vcobj)                 UMBILICUS  umbilicus.json in patch voxels
#   CT         the CT zarr (local path or URL)      OUT        output folder
# Optional: NAMES (text file of fiber UUIDs or names, one per line; default every fiber), PY=python, CT_LEVEL=2,
#   JOIN=75, EXT=80 (patch voxels), BATCH=8, THREADS=2, STRIDE=2, PREFIX=strip-net-.
# Result: $OUT/patches. See README.md beside this file.
set -euo pipefail
MODE=${1:?pieces or mirror}; STEP=${2:?fibers, trace, split, patches or all}
case $MODE in pieces|mirror) ;; *) echo "MODE must be pieces or mirror" >&2; exit 2;; esac
case $STEP in fibers|trace|split|patches|all) ;; *) echo "STEP must be fibers, trace, split, patches or all" >&2; exit 2;; esac
: "${SOURCE:?}" "${CT:?}" "${UMBILICUS:?}" "${OUT:?}"
PY=${PY:-python}; PKG=vesuvius.neural_tracing.tracers.fibers.sheet_strip
mkdir -p "$OUT"
run() { [ "$STEP" = "$1" ] || [ "$STEP" = all ]; }
if run fibers; then
  if [ "$MODE" = pieces ]; then
    "$PY" -m $PKG.gap_strips.pieces "$SOURCE" "$OUT/pieces.vcobj" ${NAMES:+--only "$NAMES"}
  else
    "$PY" -m $PKG.gap_strips.mirror "$SOURCE" "$OUT/pieces.vcobj" "$OUT/chains.vcobj" ${NAMES:+--only "$NAMES"} \
      --join "${JOIN:-75}" --ext "${EXT:-80}"
  fi
fi
TRACED=$OUT/pieces.vcobj; STRIPS=$OUT/strips_traced
if [ "$MODE" = mirror ]; then TRACED=$OUT/chains.vcobj; STRIPS=$OUT/strips_split; fi
if run trace; then
  "$PY" -m $PKG.infer_all --fibers "$TRACED" --out "$OUT/strips_traced" --ct "$CT" --ct-level "${CT_LEVEL:-2}" \
    --umbilicus "$UMBILICUS" --batch "${BATCH:-8}" --threads "${THREADS:-2}"
fi
if run split && [ "$MODE" = mirror ]; then
  "$PY" -m $PKG.gap_strips.split "$OUT/strips_traced" "$OUT/chains.vcobj" "$OUT/pieces.vcobj" "$STRIPS"
fi
if run patches; then
  "$PY" -m $PKG.gap_strips.fit_patches "$STRIPS" "$OUT/patches" --prefix "${PREFIX:-strip-net-}" \
    --stride "${STRIDE:-2}" --umbilicus "$UMBILICUS"
fi
