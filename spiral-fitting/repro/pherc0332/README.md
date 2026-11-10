# PHerc0332 spiral fit: exact code archive (frozen tree s1fh)

This branch is an ARCHIVE, not for merging. It holds the exact code the PHerc0332 (scan 20251211183505) spiral fits
ran from, so they can be reproduced byte for byte. Current villa main has moved on: since #2485, spiral-fitting reads
its inputs only from the object store, and these fits used directory datasets. Porting these features to main is
separate work.

## What this tree is
`spiral-fitting/` on this branch is the successor tree used from 2026-11-09 onward: s1_f327248db2c1f694 plus the config-gated `patch_sampling_reserved` draw share (off by default; with it off the sampling probabilities are identical to s1_f327). The four files that differ from s1_f327 are config.py, descriptions.json, fit_spiral.py and tests/test_config.py, plus the new tests/test_patch_sampling_reserved.py. See frozen_root/POST_FREEZE_NOTES.txt.

The rest of the monorepo is commit 30b4d409c (2026-10-30), which this frozen tree was taken from. The editable
sibling `vc_object_store/` and `vesuvius/src/vc3d_fiber_format/` come from that commit. The other tree is on
branch pherc0332-repro-s1_f327 (the tree every reference result came from).

## Verifying identity
`frozen_root/source.sha256` is the manifest captured when the tree was frozen (paths are relative to the frozen
root: `spiral-fitting/...`, `vesuvius/...`, `work/...`). Every `spiral-fitting/` and `vesuvius/` entry matches
this branch, and there are no other files under `spiral-fitting/` except this `repro/` directory. The `work/`
entries correspond to `frozen_root/work/` here (byte copies). Documented exceptions:
- `work/s1/s1_decide.py` is the post-freeze replacement (seed-2 pairing fix); this manifest hashes that replacement. The decision script is not used by the fit or scoring chain.

From the root of a checkout of this branch:

    # spiral-fitting/ and vesuvius/ entries (paths are repository-relative):
    grep -E '  (spiral-fitting|vesuvius)/' spiral-fitting/repro/pherc0332/frozen_root/source.sha256 | sha256sum -c
    # work/ entries against the byte copies here (the exceptions above are expected to fail):
    (cd spiral-fitting/repro/pherc0332/frozen_root && grep -E '  work/' source.sha256 \
        | sed 's#work/\.\./run_fit\.sh#work/run_fit.sh#' | sha256sum -c)

## Not in git
- `inputs.sha256` (189,701,348 bytes, sha256 e3b4ef4489fb8ae0f2255f597d56cc5c848dc2bdde2a7b57863e772b113332bb, over
  GitHub's file-size limit) is published as `eval/inputs.sha256` in the data bundle below. `frozen_root/inputs.json`
  is its small companion index.
- Data: the inputs as of the #81 reference, the scorer copies, DEPENDENCIES.md (portable scoring invocation and the
  hard-coded path constants in work/slice_eval.py) and the source archive of s1_f327248db2c1f694 are on
  dl.ash2txt.org under /home/data/public/datasets/spiral_datasets/PHerc0332/20251211183505 (published 2026-11-05,
  INVENTORY.json + UPLOAD_COMPLETE).
- Not published yet (in preparation): per-reference recipes (run manifests with their overrides and lineage),
  dataset-arm manifests and an arm assembler, launch scripts, a path-relocation wrapper, and the inputs added after
  2026-11-05 (needed for AW, SWA, M6 and M8). Until those land, only #81 has its data on the server.

## Environment
`frozen_root/venv.json` records the interpreter and `pip freeze` of the environment the fits ran in: Python 3.14,
torch 2.11.0+cu128, an RTX 4070 12 GB, Linux. `spiral-fitting/pyproject.toml` and `uv.lock` on this branch are the
frozen ones. A fresh-environment recipe (native prerequisites, `uv sync --frozen` and the known differences from
venv.json) is being validated and will be added.

## Hard-coded paths
The `work/` scripts carry workstation paths (`/media/djosey/nvme4/spiral_0332`, `/home/djosey/Documents/villa`):
run_fit.sh, s1/chain_s1.sh, slice_eval.py, slice_eval_frozen.py, holdout_score.py, s1/s1_manifest.py and
s1/s1_score.py. The `spiral-fitting/` runtime modules carry none. They are kept byte-identical here; relocation
happens in a separate wrapper, not by editing these copies.
