# Joint solve learning-rate and reference-width comparisons

State LR high=0.2078460969, low=0.02078460969. Width LR=2 in both.
Six GPUs,32768 constraints/GPU, sampled minibatch updates. Pass labels rounded;
336 updates=9.999892 dataset equivalents. All joint variables train immediately.
Reference-started runs initialize widths as well as assignments; radial width prior unchanged.
Loss is mean online objective over previous10passes, not frozen-state full objective.
Reference L1/within2 use retained matched original fibers after global translation.
Width factor is equal-interval mean fitted/reference, supported intervals only.
High runs cancelled at user request after fresh100 had completed. Low100/1000 continue.


## uv_joint1000_fresh_low_width_v2

In progress.

|Pass|Loss|Ref L1|Within ±2 %|Width factor|Retained length %|
|---:|---:|---:|---:|---:|---:|
|0|—|12.2423|12.991|0.86757|100.000|

## uv_joint100_fresh_high_width_v2

Complete.

|Pass|Loss|Ref L1|Within ±2 %|Width factor|Retained length %|
|---:|---:|---:|---:|---:|---:|
|0|—|12.2423|12.991|0.86757|100.000|
|10|14.8335|11.5217|13.068|0.88362|65.077|
|20|5.9551|11.4374|13.046|0.88321|65.815|
|30|8.7172|11.3706|13.006|0.88293|66.164|
|40|9.9300|11.3111|13.022|0.88268|66.378|
|50|9.7680|11.2528|13.050|0.88257|66.542|
|60|9.1792|11.1982|13.057|0.88246|66.675|
|70|8.5894|11.1439|13.088|0.88244|66.795|
|80|8.1751|11.0922|13.120|0.88236|66.898|
|90|7.6863|11.0391|13.186|0.88236|66.982|
|100|7.3176|10.9919|13.220|0.88233|67.054|

## uv_joint100_fresh_low_width_v2

Complete.

|Pass|Loss|Ref L1|Within ±2 %|Width factor|Retained length %|
|---:|---:|---:|---:|---:|---:|
|0|—|12.2423|12.991|0.86757|100.000|
|10|51.3169|11.7993|13.051|0.87086|79.884|
|20|6.2557|11.7397|13.171|0.87005|67.628|
|30|3.4045|11.7059|13.186|0.86917|66.678|
|40|2.7151|11.6823|13.234|0.86849|66.960|
|50|2.4010|11.6648|13.227|0.86797|67.353|
|60|2.2193|11.6480|13.257|0.86758|67.675|
|70|2.1023|11.6347|13.227|0.86729|67.964|
|80|2.0209|11.6207|13.238|0.86705|68.204|
|90|1.9604|11.6060|13.263|0.86690|68.411|
|100|1.9125|11.5911|13.259|0.86675|68.587|

## uv_joint100_reference_low_width_v2

Complete.

|Pass|Loss|Ref L1|Within ±2 %|Width factor|Retained length %|
|---:|---:|---:|---:|---:|---:|
|0|—|0.0005|100.000|1.00000|100.000|
|10|29.3388|0.6848|96.576|1.00079|96.349|
|20|3.8435|0.7763|94.647|1.00136|91.758|
|30|1.8533|0.8359|93.566|1.00312|91.207|
|40|1.3613|0.8821|92.789|1.00563|91.223|
|50|1.1543|0.9211|92.158|1.00862|91.298|
|60|1.0433|0.9541|91.592|1.01194|91.377|
|70|0.9761|0.9840|91.041|1.01532|91.446|
|80|0.9296|1.0115|90.557|1.01866|91.511|
|90|0.8986|1.0367|90.097|1.02195|91.556|
|100|0.8736|1.0606|89.648|1.02515|91.602|
