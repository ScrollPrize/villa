# VC3D release preservation — maintainer review notes

**Proposal for [#2049](https://github.com/ScrollPrize/villa/issues/2049), not a production migration.**
AI-assisted implementation; no native GITHUB_TOKEN deployment has been validated.

## Intended change

The official latest and stable jobs currently delete their preceding GitHub Release when publishing new packages. Each old release has six uploaded platform assets. This proposal instead:

1. Requires a unique archive alias referencing the **original annotated tag object**, not merely its peeled commit. Before first use, a maintainer must bootstrap one such alias for each current channel; **do not** overwrite or delete existing releases or binaries.
2. PATCHes the existing rolling release to use that archive tag, requiring the same numeric release ID and all six original numeric asset IDs to survive.
3. Moves the rolling channel tag with an atomic lease-protected push while pre-staging the next archive alias, then publishes the new six assets.
4. Verifies archived history/receipts and offers restricted same-commit/idempotent recovery. A broken, incomplete or ambiguous history fails closed.

Only the two workflow files and two scripts are changed; historical binary bytes are never reuploaded in the candidate. The publication of new packages still uploads the *new* six files as before.

## Reproducible checks available in this PR

Run the included, **synthetic metadata-only** standard-library unit tests locally:

    python -m unittest discover -s .github/scripts/tests -p 'test_vc3d_archive_contract.py'

These eight tests check the expected six-asset identities for both channels, duplicates, mismatched source SHA, missing SHA256 digest, missing files and release metadata. They do **not** simulate the full GitHub release API and do **not** establish token permissions.

The separate P135 local test harness, not included in this PR, ran 24 distinct methods successfully in its originating environment: actual workflow Bash publication blocks, local disposable Git repositories, and a simulated GitHub API. In that model, the original paths delete twelve old asset IDs, while the proposal preserves all twelve and adds twelve new IDs. This is **not** an upstream Actions run, real package execution, nor scroll voxel processing. A private earlier V5 synthetic proof used a temporary PAT and does not establish native token behavior.

## Required decisions before adoption

- Maintainers: inspect and approve the historical alias bootstrap, effective GitHub release/tag permissions, release mutability, and possible external races. **Do not run this code against real releases without these checks.**
- GitHub release PATCH, annotated Git tag changes and release creation are **not one transaction**. Interrupted uploads, external concurrency, and the bounded archive-history traversal (maximum 2,000 links) remain operational risks.
- The patch **does not recover previously deleted binaries**. URLs containing the moving tag may change. Asset IDs and metadata alone are not proof of the binary build provenance.
- PHerc 343 in the issue is another researcher's experiment. A separate CPU renderer reproduced that particular result; we do not claim the render is impossible without old VC3D. We did not run VC3D on scroll CT data.

The upstream [CONTRIBUTING.md](../../CONTRIBUTING.md) asks for firsthand real-scroll use and direct before/after evidence. This proposal does **not** satisfy those requirements today. Feedback on whether a release-infrastructure PR can be reviewed with release-level evidence is requested; the author verification box must remain unchecked until personally verified.
