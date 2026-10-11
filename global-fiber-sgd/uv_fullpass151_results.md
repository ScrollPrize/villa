# Full-pass joint151 results

Completed both10-pass runs (CCW1499.2s; CW1508.9s), no subsequent strong run.
Reference metric: equal original fibers, median global offset, retained matched
fibers only. Sign diagnostics use a fixed100k sample for passes0–9 and the full
graph at10. Loss is the mean online objective across each pass, not a frozen-state
global objective. Inherited rejection-logit warmup holds gradients in passes1–5.

## CCW

|Pass|L1 winding error|Within ±2 %|Retained length %|Sign violation %|Mean online loss|Matched fibers|Offset|
|---:|---:|---:|---:|---:|---:|---:|---:|
|0|12.242345|12.9912|100.0000|40.9860|—|382459|12|
|1|12.022044|12.9682|99.9893|15.3376|953.105591|382459|12|
|2|11.908604|13.0088|99.9869|13.0041|800.539856|382457|12|
|3|11.820130|13.0861|99.9829|11.3162|282.626282|382453|12|
|4|11.741293|13.0963|99.9811|10.0417|101.578636|382452|12|
|5|11.671729|13.1038|99.9764|9.2193|48.004284|382446|12|
|6|11.619284|13.1521|94.8535|8.4694|28.280231|373888|12|
|7|11.580048|13.1793|91.6497|7.0907|17.272625|369033|11|
|8|11.554746|13.1615|89.1748|6.2101|11.436736|364518|11|
|9|11.535806|13.1786|87.4787|5.5014|8.167250|361024|11|
|10|11.517027|13.1949|86.3351|5.0359|6.363428|358457|11|

Config: run_uv_fullpass151_ccw_joint.json; full output:
data/uv_fullpass151_v1/ccw_joint/.

## CW

|Pass|L1 winding error|Within ±2 %|Retained length %|Sign violation %|Mean online loss|Matched fibers|Offset|
|---:|---:|---:|---:|---:|---:|---:|---:|
|0|12.273619|12.6636|100.0000|40.8793|—|382459|12|
|1|12.094826|12.8618|99.9910|15.4033|953.530762|382459|11|
|2|11.979064|12.8587|99.9863|12.9744|800.902039|382457|11|
|3|11.887806|12.8782|99.9844|11.3960|282.831757|382454|11|
|4|11.806856|12.9245|99.9805|10.2050|101.652611|382451|11|
|5|11.737594|12.9399|99.9760|9.2410|48.064137|382446|11|
|6|11.683584|12.9816|94.8809|8.4178|28.328009|373967|11|
|7|11.642239|12.9705|91.6334|7.1211|17.313614|369023|11|
|8|11.616925|12.9666|89.1570|6.2565|11.472732|364490|11|
|9|11.598656|12.9823|87.4824|5.5536|8.174139|361054|11|
|10|11.579659|12.9902|86.3099|5.0365|6.374465|358433|11|

Config: run_uv_fullpass151_cw_joint.json; full output:
data/uv_fullpass151_v1/cw_joint/.

## Final robust ranges

Both directions: fitted q10/q90=48/121 after+11global offset (raw37/110).
Reference on the same matched cohorts:42/124. Quantiles were added after launch,
so only the final saved states have this diagnostic; intermediate quantiles are
not recoverable from the saved history. Range similarity does not imply local
assignment agreement: final mean absolute errors remain11.52/11.58windings.

Reference orientation:99.97443%CCW tangential weight across182808segments in
atan2(y-cy,x-cx) convention, increasing reference column.

## Reproduction

Use the approved runner for each joint config:

```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_fullpass151_ccw_joint.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_fullpass151_cw_joint.json
```

Completed output directories are protected; choose a fresh output root to rerun.
CPU-only quantiles: same runner with report_uv_fullpass151_reference.json.
