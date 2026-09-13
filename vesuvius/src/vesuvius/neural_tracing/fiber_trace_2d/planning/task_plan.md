# Plan

1. Freeze current EMA99 hash/update, run off/original and previous best (.25,128/1024/count8/gaussian) before native changes.
2. Extend shared native helper and CLI; default weighted mean unchanged. Best history=min cosine cost over retained references, no age discount. EMA normalized per beam with distance half-lives128/512/2048; score against parent before updating. Fixed seed per restart. EMA+seed equally averages two costs. Gated variant only updates presence>=.5 and parent cosine>=.5; decay based on distance of current step even after rejected updates.
3. Native tests: formulas, best vs mean, gate, beam copy isolation, invalid options. Approved runner builds/tests; no geometry/default changes.
4. Rerun unchanged controls; sweep .25 weight with EMA/gated/EMA+seed/gated+seed at three half-lives, best history windows256/1024/4096 spacing128/count64. Reuse frozen cache, manager-gated extension. Repeat winner and refine +/- .05 weight to local minimum.
5. Report full results and timings, single-fiber limitation; keep defaults unchanged.

## Spec update
Document opt-in modes, formulas, ownership; preserve mean default.

## Docs updates
Add EMA99 report and embedding_tracing options; update changelog, status, task log.

## Validation
Independent review; approved experiment/test runners; native/Python regression tests; exact off/mean pre-change replay and winning repeat. Existing CMake build (documented build_dependencies.sh absent).
