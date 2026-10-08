# Per-fiber change: previous EMA999 u11800 versus refinement EMA999 u14400

Ranked by change in counted restarts (new minus old), combining both directions and all eligible spans of each original fiber. Negative is improvement. Same original-GT protocol and80-base-voxel restart filtering. Additional training updates confound attributing change to refinement itself.

User clarification: individual directed CASE extremes across the full cohort (not aggregated fibers): most helped, all −2 restarts:62(PHerc0332,3→1),160(Paris4,2→0),167(Paris4,8→6). Most worsened, all +2:4(0139,2→4),8(0139,5→7),9(0139,6→8),42(0139,3→5),86(Paris4,0→2),103(Paris4,0→2). All ties included.

Most helped: PHerc0139 kb_20260905T035141671_000093.json,16→13, cases16–31; improvements specifically23,25,27, one restart fewer each.

Most worsened: PHerc0139 kb_20260901T224443942_000068.json,12→16, cases8–15; regressions specifically8,9, two extra each.

Visualization root: `/ephemeral/hendrik/fiber3d_train2/evaluations/refinement_ema999_u14400_vis_20260915_v1/full/cases/`.

| Scroll | Fiber | Old restarts | New | Change | Old mean mm | New mean mm |
|---|---|---:|---:|---:|---:|---:|
| PHerc0139 | kb_20260905T035141671_000093.json | 16 | 13 | -3 | 10.647 | 12.928 |
| PHercParis4 | kb_20260729T173847941_000210.json | 20 | 18 | -2 | 10.709 | 11.836 |
| PHercParis4 | anon_20260827T134102921_000271.json | 2 | 0 | -2 | 6.604 | 19.813 |
| PHerc1299 | lt_20260825T075316065_000069.json | 2 | 0 | -2 | 10.781 | 32.344 |
| PHerc0332 | dj_20260810T113807458_000001.json | 4 | 2 | -2 | 72.185 | 120.309 |
| PHercParis4 | anon_20260907T141106877_000497.json | 2 | 0 | -2 | 51.126 | 153.378 |
| PHercParis4 | kb_20260729T182039882_000211.json | 1 | 0 | -1 | 5.578 | 11.156 |
| PHercParis4 | anon_20260815T025206849_000041.json | 4 | 3 | -1 | 12.247 | 15.309 |
| PHerc0139 | lt_20260909T082646148_000107.json | 1 | 0 | -1 | 16.568 | 33.137 |
| PHercParis4 | anon_20260902T181618881_000388.json | 3 | 2 | -1 | 60.023 | 80.031 |
| PHerc0139 | lt_20260909T071741261_000094.json | 0 | 0 | 0 | 4.102 | 4.102 |
| PHerc0139 | lt_20260909T084022344_000109.json | 1 | 1 | 0 | 12.746 | 12.746 |
| PHerc0139 | kb_20260827T204421830_000022.json | 2 | 2 | 0 | 14.945 | 14.945 |
| PHerc0139 | kb_20260908T172232906_000117.json | 1 | 1 | 0 | 18.885 | 18.885 |
| PHerc1451 | dj_20260903T200017539_000006.json | 0 | 0 | 0 | 22.258 | 22.258 |
| PHercParis4 | anon_20260905T163237868_000469.json | 7 | 7 | 0 | 23.017 | 23.017 |
| PHerc1299 | lt_20260824T091041867_000056.json | 5 | 5 | 0 | 23.821 | 23.821 |
| PHerc1451 | dj_20260903T195009387_000005.json | 0 | 0 | 0 | 26.112 | 26.112 |
| PHerc0332 | kb_20260822T033517433_000058.json | 0 | 0 | 0 | 30.679 | 30.679 |
| PHerc0332 | kb_20260822T164832186_000062.json | 0 | 0 | 0 | 33.555 | 33.555 |
| PHercParis4 | anon_20260901T201915394_000335.json | 0 | 0 | 0 | 35.826 | 35.826 |
| PHerc0139 | lt_20260904T062829219_000052.json | 5 | 5 | 0 | 45.601 | 45.601 |
| PHercParis4 | anon_20260902T125309620_000368.json | 1 | 1 | 0 | 48.058 | 48.058 |
| PHercParis4 | anon_20260815T020414697_000008.json | 0 | 0 | 0 | 49.090 | 49.090 |
| PHercParis4 | anon_20260821T221016194_000135.json | 3 | 3 | 0 | 52.937 | 52.937 |
| PHercParis4 | kb_20260711T043119415_000150.json | 3 | 3 | 0 | 55.061 | 55.061 |
| PHercParis4 | anon_20260816T224409952_000122.json | 0 | 0 | 0 | 65.095 | 65.095 |
| PHerc1299 | lt_20260819T081412107_000011.json | 0 | 0 | 0 | 72.265 | 72.265 |
| PHerc1299 | lt_20260824T083243313_000053.json | 0 | 0 | 0 | 78.743 | 78.743 |
| PHerc1299 | lt_20260819T050805459_000002.json | 0 | 0 | 0 | 80.825 | 80.825 |
| PHercParis4 | anon_20260824T032219305_000197.json | 0 | 0 | 0 | 81.787 | 81.787 |
| PHercParis4 | anon_20260901T165827570_000316.json | 0 | 0 | 0 | 128.956 | 128.956 |
| PHercParis4 | anon_20260902T131435457_000370.json | 0 | 0 | 0 | 145.611 | 145.611 |
| PHercParis4 | anon_20260902T103444019_000352.json | 0 | 0 | 0 | 148.276 | 148.276 |
| PHercParis4 | anon_20260901T170547855_000317.json | 0 | 0 | 0 | 171.571 | 171.571 |
| PHercParis4 | sm_20260707T184533000_000122.json | 0 | 0 | 0 | 174.961 | 174.961 |
| PHercParis4 | anon_20260904T210541536_000446.json | 0 | 0 | 0 | 175.857 | 175.857 |
| PHercParis4 | anon_20260822T103908823_000165.json | 0 | 0 | 0 | 224.740 | 224.740 |
| PHercParis4 | anon_20260902T215234337_000400.json | 0 | 0 | 0 | 261.841 | 261.841 |
| PHercParis4 | anon_20260904T145959056_000421.json | 0 | 0 | 0 | 262.868 | 262.868 |
| PHercParis4 | anon_20260822T121633375_000173.json | 0 | 0 | 0 | 263.041 | 263.041 |
| PHercParis4 | sm_20260706T204346695_000119.json | 0 | 0 | 0 | 290.724 | 290.724 |
| PHerc0139 | lt_20260831T061702750_000026.json | 8 | 9 | 1 | 7.697 | 6.928 |
| PHerc0139 | kb_20260828T155910844_000043.json | 4 | 5 | 1 | 52.761 | 43.968 |
| PHercParis4 | kb_20260803T040828806_000224.json | 1 | 2 | 1 | 117.687 | 78.458 |
| PHercParis4 | anon_20260903T214715058_000404.json | 0 | 1 | 1 | 345.753 | 172.877 |
| PHercParis4 | anon_20260815T135300158_000075.json | 0 | 2 | 2 | 42.166 | 14.055 |
| PHercParis4 | anon_20260902T214326941_000399.json | 4 | 6 | 2 | 21.602 | 15.430 |
| PHercParis4 | anon_20260826T172849645_000255.json | 1 | 4 | 3 | 89.039 | 35.616 |
| PHerc0139 | kb_20260901T224443942_000068.json | 12 | 16 | 4 | 18.580 | 14.208 |
