# Multi-anchor appearance model

- Prior EMA/best-history state committed053f76eff. Reference frozen EMA99 update6500/832000; winner EMA512weight.25=69 (28+41), exact repeat hash6e590851a9a00555dbc14edc3aa6f1fdccfc57a47b4a8c3c85a22ba900050484. Training continues. Reuse prediction cache.
- Plan independently approved: explicit nearest-anchor/EMA convex mixture, actual observed medoids, stable ties and bounded history. Implemented shared native helper, per-beam state and CLI; native/Python regression build running. Grid: K2/4/8, bank32spacing128 vs64spacing256, anchor-only vs half-EMA. No default changes.
- Review clarified anchor score uses parent-window bank, expiry on selected-child update (one-step grace). Explicitly documented, not claiming candidate-time strict filtering like mean/best. Existing expiry/isolation unit test exercises retained parent versus pruned child.
- Approved test runner: QuickBuildO3 build,2native suites pass1.57s and113Python pass38.69s; logs cooperative-tests/20260913_211352_827829. Anchor grid started via approved embedding-trace-sweep manager nonexclusive20GiB/16CPU.
- All12 anchor settings completed: short bank K2/4/8 standalone76/84/76,halfEMA83/78/84;long bank standalone87/78/82,halfEMA78/82/74. EMA control69 exact. Best anchor longK8 halfEMA74 still worse. Continue finer EMA half-lives256/384/640/768/1024 and lighter anchor contributions; also refine longK8 blend. No early stop at failed initial grid.
- User added next LRU anchor experiment; scheduled after refinement instead of stopping. Capacity4/8, admission128/512, nearest/average/EMA blend. Finer EMA results256/384/640/768/1024=74/72/87/84/73; no improvement yet. ShortK2 mixes.75/.9=91/76.
# Latest continuation

- Mechanism implementation committed1425f53c5 after native2/Python116 tests; mechanism sweep running. Fine .05 weight neighbors now integrated into independent family coordinate cycles (opt-in), with up-front tick validation; Python116 passed31.81s (`20260913_221326_604149`). This checks neighboring weights even if a coarse axis winner lies between tested endpoints.

- Whole-path initial grid complete: K2spacing12881/78,51272/81; K4spacing12878/83,51275/84; K8spacing12884/84,51276/84 (nearest/halfEMA). EMA69 repeat exact. Committed initial implementations/harness844856518.
- Novelty LRU and support-softmin implemented and independently reviewed; native tests pass1.59s, Python suite running. Support-softmin uses stable expm1/log1p for large temperatures after review; normalization/split/limits tests included. Mechanism grid prepared; then all-family refinement remains required.

- Initial LRU complete: K4/spacing128 nearest/mean/halfEMA75/88/77; K4/51273/73/81; K8/12876/87/79; K8/51275/80/84. EMA69 repeat exact. Whole-path build fixed seed configuration scope; native2 tests1.95s/Python11545.24s pass (`20260913_215750_020249`). Whole-path12-setting grid launched next.

- LRU native2/Python113 tests passed; sweep running under nonexclusive manager, controls reproduced. Whole-path spherical-summary implementation independently reviewed, no blocking issues; tests added for support conservation, old appearance, cancellation/recovery, capacity and beam isolation. Build deferred until LRU finishes to keep its native binary fixed.
- Added finite per-family coordinate-refinement helper for subsequent sweeps; each family starts from its own best seed, strict improvement and memoized candidate evaluations. Tests exercise multiple cycles and retaining nonwinning families. User additionally requests algorithmic refinements, not only parameters; novelty-aware LRU admission and support-aware aggregation are next mechanism hypotheses.

- Refinement complete: EMA half-lives256/384/640/768/1024 scored74/72/87/84/73 restarts. Additional windowed-anchor mixtures scored91/76/80/88/85/83/73. Control EMA512 remains69 and repeated identically. No improvement; proceed to LRU.
- User correctly identified the rolling-window restriction as deviating from whole-path descriptive/diverse anchors. After LRU, remove that restriction using bounded streaming appearance support; do not present windowed results as fulfilling that requirement.
