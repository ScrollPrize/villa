# Status

- [x] Commit prior code053f76eff
- [x] Review plan
- [x] Implement/test windowed, LRU and whole-path appearance models (commit844856518)
- [x] Initial sweeps and exact controls: windowed74/LRU73/whole-path72 vs EMA69
- [x] Evaluate novelty LRU and support-softmin algorithmic refinements (both71 vs EMA69)
- [x] First independent refinement of all13 families and exact winner repeats (270 replays, best64)
- [ ] Cross-seed and admission-boundary refinements
- [ ] Repeat winners, examine further supported method improvements
- [x] Build/test seed-only, delayed EMA and hit-only LRU variants (2native suites,117Python tests)
- [x] Evaluate/refine seed/delay/hit variants (91 replays, all exact repeats pass; best remains64)
- [x] Build/test normalized-mean and coherence diagnostics (native2/Python118 pass)
- [x] Recheck optimized-path64 control and coherence counts (7,233,138, histogram matches)
- [ ] Compare normalized mean against64/68/66 controls, refine if useful
- [ ] Document full results
