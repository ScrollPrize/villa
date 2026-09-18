# Short200 full-resolution fiberlet versus beam replay (2026-09-18)

The immutable 200-piece short-failure benchmark was evaluated with the best
full-resolution EMA999 prediction cache from
`5scroll_json_eligible_fullres_refinement_128_sd1_16x8_ema_20260916_213753`.
Both rows cover the same 200 directed pieces and 662.089848 mm.

| tracer | restart episodes | failures/mm | mean segment (mm) | failure-free pieces |
|---|---:|---:|---:|---:|
| native beam | 56 | 0.084581 | 11.6156 | 145/200 |
| bounded fiberlet, width 64 | 60 | 0.090622 | 10.8539 | 147/200 |

Fiberlets improved 20 pieces by 20 episodes and worsened 23 pieces by 24
episodes, leaving a net increase of four. The remaining 157 pieces had equal
episode counts. No fiberlet case hit the route-state limit.

## New fiberlet failures

These 17 pieces were failure-free with beam replay and had one restart episode
with bounded fiberlet replay. `render case` is local to the physical-scroll
artifact under the fiberlet result's `scrolls/NN/cases/` directory.

| source | piece | render case |
|---|---|---:|
| PHerc0139 | `041_c9a5ebd94ef89bb4f82e` | `scrolls/00/cases/19` |
| PHerc0139 | `052_98fad943d272459e1d92` | `scrolls/00/cases/30` |
| PHerc0139 | `068_72841b25864cc0bb45d3` | `scrolls/00/cases/46` |
| PHerc0139 | `069_589aad14b04c925de629` | `scrolls/00/cases/47` |
| PHerc0139 | `117_c3f367aa4c47b4e76ac4` | `scrolls/00/cases/75` |
| PHerc0139 | `118_e419bb6faf00f03fdc3b` | `scrolls/00/cases/76` |
| PHerc1229 | `023_70688f449967881a4cc2` | `scrolls/02/cases/4` |
| PHerc1229 | `123_40788538af8051f815e4` | `scrolls/02/cases/14` |
| PHerc1229 | `132_8ad7ddbc4d3538a249a1` | `scrolls/02/cases/23` |
| PHercParis4 eval | `027_f6ca60603e01b6f765b2` | `scrolls/04/cases/2` |
| PHercParis4 eval | `031_7896940a8b9dbf62041b` | `scrolls/04/cases/6` |
| PHercParis4 eval | `035_4084f12016174acc8ce2` | `scrolls/04/cases/10` |
| PHercParis4 eval | `036_c78d97fcf558a4dc51ae` | `scrolls/04/cases/11` |
| PHercParis4 eval | `076_c9be9415fd4a682ab535` | `scrolls/04/cases/15` |
| PHercParis4 eval | `090_1bb7965cb122613fa0aa` | `scrolls/04/cases/29` |
| PHercParis4 eval | `167_2001cc8f57030b876611` | `scrolls/04/cases/62` |
| PHercParis4 eval | `177_196e0e9e35a5212af8a6` | `scrolls/04/cases/72` |

Six pieces already failed under beam replay and gained another seven episodes:

| source | piece | render case | beam -> fiberlet |
|---|---|---:|---:|
| PHerc0139 | `013_741eab3a25953b8727d5` | `scrolls/00/cases/13` | 1 -> 3 |
| PHerc0139 | `017_e6ce98dd4ea614139d74` | `scrolls/00/cases/17` | 1 -> 2 |
| PHercParis4 eval | `030_f101d328e0211cebaeb5` | `scrolls/04/cases/5` | 1 -> 2 |
| PHercParis4 eval | `138_49905b47b5309f96a974` | `scrolls/04/cases/33` | 1 -> 2 |
| PHercParis4 eval | `139_73b538a69aa1abcfc637` | `scrolls/04/cases/34` | 1 -> 2 |
| PHercParis4 eval | `191_5a97fdc4d8c46fbca01d` | `scrolls/04/cases/86` | 1 -> 2 |

## Artifacts

- Beam result: `/ephemeral/hendrik/fiber3d_train2/evaluations/short200_fullres_refinement_213753_best_ema999_vis_20260918_v1`
- Fiberlet result and 200 complete galleries: `/ephemeral/hendrik/fiber3d_train2/evaluations/short200_fullres_fiberlet_bounded_w64_213753_best_ema999_20260918_v4`
- Render review index: `README.md` and `review_index.json` in the fiberlet result root.
- Original fiberlet wall time: 2450.282540 s, retained in `metrics.json`.

The visualization pass loaded the existing native trace artifacts and processed
zero inference tiles. It did not rerun either prediction inference or tracing.
