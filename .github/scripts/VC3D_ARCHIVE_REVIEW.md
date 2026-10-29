# VC3D release preservation — maintainer review notes

**Proposal for [#2049](https://github.com/ScrollPrize/villa/issues/2049), not a production migration.**
AI-assisted implementation; no native GITHUB_TOKEN deployment has been validated.

## Intended change

The official latest and stable jobs currently delete their preceding GitHub Release when publishing new packages. Each old release has six uploaded platform assets. This proposal instead:

1. Requires a unique archive alias referencing the **original annotated tag object**, not merely its peeled commit. Before first use, a maintainer must bootstrap one such alias for each current channel; **do not** overwrite or delete existing releases or binaries.
2. PATCHes the existing rolling release to use that archive tag, requiring the same numeric release ID and all six original numeric asset IDs to survive.
3. Moves the rolling channel tag with an atomic lease-protected push while pre-staging the next archive alias, then publishes the new six assets.
4. Verifies archived history/receipts and offers restricted same-commit/idempotent recovery. A broken, incomplete or ambiguous history fails closed.

Production changes are limited to the two workflow files and two scripts; historical binary bytes are never reuploaded in the candidate. The publication of new packages still uploads the *new* six files as before.

## Reproducible checks available in this PR

Run all public tests from the repository root with Python 3.9+, Git and Bash on a POSIX system. No third-party Python packages, credentials, downloads or Actions are needed:

    python3 -m unittest discover -s .github/scripts/tests -p 'test_vc3d_archive*.py' -v

- `test_vc3d_archive_contract.py`: eight synthetic metadata tests for expected asset identities, duplicates, mismatched source SHA, missing digests/files and release metadata.
- `test_vc3d_archive_local_git.py`: nine integration test methods, each exercising both `latest` and `stable`. The **actual Python and Bash helpers** run as subprocesses against disposable local Git repositories and a strict fake `gh` API. Git is restricted to the `file` protocol; no real GitHub client is called. New release creation is a fixture operation, **not** execution of the complete publication YAML or real uploads.

The local-Git suite checks two successive promotions, exact historical tag objects and original IDs, rejection of a changed historical asset/tag, atomic rejection of a two-ref push, a simulated denied PATCH, missing bootstrap, incomplete uploads, and the distinct interruption cases below.

On 29 October 2026, the corrected candidate passed all **17 distinct public test methods**, with no skips, run in batches because of the execution environment's per-call time limit. Channel subtests are not counted as extra methods. The targeted replacement-ID regression failed on the original published helper (`a6aba4c2e2b4c30b42f804b4a01b1b9ef2b24f36`) for both channels and passed after the correction.

The separate P135 harness remains historical evidence: 24 distinct local methods passed on the earlier candidate in its originating environment, using actual workflow Bash blocks, local Git and a simulated API. It has **not** been rerun on this correction and is not included in this PR. Its old/new comparison lost twelve prior IDs in the deletion-based baseline and retained twelve plus twelve new IDs in the candidate model. Do not add that historical count to the current 17 tests. An earlier private V5 test used a temporary PAT; neither it nor these local tests establishes native token behavior.

## Interruption boundary corrected on 29 October

The original helper accepted `ALREADY_ARCHIVED_ZERO_COPY` after an interruption between the release PATCH and the atomic tag push. At that point the old rolling tag contains no durable receipt of **its own** original asset IDs. Re-reading the archived release could therefore accept replacement IDs as original ones.

The regression first archives six synthetic asset IDs, leaves the rolling tag unchanged, replaces one ID in the fake service, and retries. The original helper returned success with that replacement ID. The corrected helper stops with `maintainer recovery required`, without another PATCH or tag push. Even an apparently intact archive stops in this state: current metadata alone cannot prove that its IDs are the originals.

| State on retry | Behavior |
| --- | --- |
| PATCH completed, old rolling tag still in place, rolling release absent | Stop for maintainer identity verification; do not adopt freshly observed IDs into a new receipt. |
| Atomic tag push completed with exact predecessor receipt, new release absent | `REPAIR_RELEASE_ONLY` remains available after the recorded archive IDs and tag objects are verified. |
| New release complete and predecessor history intact | `ALREADY_PUBLISHED`, without another PATCH or tag movement. |
| New upload incomplete, bootstrap missing, or history altered | Stop; no automatic deletion/reupload is attempted. |

For manual recovery, use independently retained pre-interruption evidence (for example the original run's receipt/logs) to establish the original release and asset identities. A fresh listing of the archive alone is insufficient. This document does not authorize any production mutation or claim a fully automated recovery procedure.

## Required decisions before adoption

- Maintainers: inspect and approve the historical alias bootstrap, effective GitHub release/tag permissions, release mutability, and possible external races. **Do not run this code against real releases without these checks.**
- GitHub release PATCH, annotated Git tag changes and release creation are **not one transaction**. Interrupted uploads, external concurrency, and the bounded archive-history traversal (maximum 2,000 links) remain operational risks.
- The patch **does not recover previously deleted binaries**. URLs containing the moving tag may change. Asset IDs and metadata alone are not proof of the binary build provenance.
- PHerc 343 in the issue is another researcher's experiment. A separate CPU renderer reproduced that particular result; we do not claim the render is impossible without old VC3D. We did not run VC3D on scroll CT data.

The upstream [CONTRIBUTING.md](../../CONTRIBUTING.md) asks for firsthand real-scroll use and direct before/after evidence. This proposal does **not** satisfy those requirements today. Feedback on whether a release-infrastructure PR can be reviewed with release-level evidence is requested; the author verification box must remain unchecked until personally verified.
