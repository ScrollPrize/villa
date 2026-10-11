# Joint optimization comparisons only

User explicitly excludes all frozen-geometry categorical results. Every row here
optimized winding/rejection probabilities, U/V anchors/offsets and winding widths.
The strong-refinement rows load earlier joint results but still update geometry;
they are not categorical/frozen runs. No optimizer rerun was needed for old rows.

| Run | Event-pass-equivalent budget | Retained length CCW / CW | Retained sign violations CCW / CW |
|---|---:|---:|---:|
|100, original penalty|6.20|91.4355% /91.3679%|6.4448% /6.4855%|
|100, then strong charge16|6.20+2.48|64.4886% /64.2597%|1.1317% /1.1100%|
|151, original penalty|6.20|87.5555% /87.5279%|6.0885% /6.1142%|
|151, then strong charge16|6.20+2.48|62.3967% /62.3517%|0.9999% /0.9853%|
|151, strong from start — in progress at3passes|3.00/10|38.4074% /38.4126%|0.0817% /0.0684%|

Sign violation percentage uses retained known-sign constraints, not all original
constraints. Retention uses length after upstream exclusions. Original-penalty
rows use joint-argmax broken classification and sign tolerance0.0001; strong rows
use broken probability>0.5 and strict zero sign tolerance. Thus those groups are
not identical readouts. All values use the full6,606,100-event input, not the
intermediate100k diagnostic sample. Budgets count sampled-event equivalents,
not guaranteed exhaustive passes. Lower error through rejection is not enough
to claim a better result. Current run unfinished; final row will be updated.

Sources (each directory has ccw/cw variants):

- `data/uv_anchor_v1/{ccw,cw}/summary.json`
- `data/uv_sign_charge_v1/{ccw16,cw16}/summary.json`
- `data/uv_fresh151_v1/{ccw,cw}_joint/summary.json`
- `data/uv_fresh151_v1/{ccw,cw}_sign/summary.json`
- `data/uv_joint151_strong_v1/{ccw,cw}/history.json` (current).
