# Current EMA999 evaluation

Latest run:5scroll_json_eligible_128_sd1_16x8_ema_20260914_105436. Source current_ema_0p999.pt (not best). Protocol unchanged; eval only without rendering. Previous alignment log archived.

Frozen currentEMA999 at update10200/samples1305600, test_ema_0p999/loss_total=.29859120762052616. SHA4753ee9bee448fe3d5be1914aefc88ca1cc33799cc69195408ea6a8bc1be7d1b. Config updated to immutable content-addressed snapshot before preparation.
Independent review approved: only checkpoint/SHA,output root,manager label differ from prior matched EMA99 config. This is not a controlled same-age EMA decay comparison.

Prepared six source collections/50fibers; validate-full confirms checkpoint,locked membership/config/normal manifests. Programmatic config comparison confirms all evaluation settings identical apart from snapshot/output/manager label. Full run launched via approved command.

Manager taskd76738d9-9ec8-4e15-ab17-9fd791dc9ccc obtained all8GPUs exclusively. Model cache keye01285ed6edb280b1aa5832990e359c965467fd82049caca9037e58f941cd4bc; new model inference, reused inputs. At139s:8717tiles/2338published chunks. Driver utilization check outside sandbox confirms all8GPUs active; sandbox nvidia-smi cannot access driver.

Completed exit0:50fibers/90eligible runs/180directions,350.143913s;29213inferred tiles/9100published chunks,zero warm inference chunks. Matched directed coverage6867.627252833243mm for all4models. Metrics derived using shared complete_report and score functions, not manually entered.
CurrentEMA999u10200:119reported restarts,mean57.230227mm,median9.759005mm,P05.268266mm,P9590.082119mm; zero-length segments0. EMA99u5200:148,46.091458,8.202461,.143040,88.437892; zero4. Restarts-19.59%,mean+24.17%. Equal-scroll normalized score69.041084 vs55.845764. Per-scroll current restarts0139:55,0332:2,1299:5,1451:0,Paris4:57.
Durable full four-model/six-scope table and raw aggregate JSON: docs/benchmarks/eligible_ema999_u10200_20260914.{md,json}; includes Classic and EMA99u1800/u5200. Current run is later-trained, so no causal claim about EMA decay. No changes to default model/tracing, no rendering requested, no deferred work.
Output /ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_current_20260914_1555. Reuse approved run_classic_fiber_evaluation.py freeze-snapshot/prepare/validate-full/full with configs/eval_eligible_ema999_current_20260914_1555.json. No code changes; reuse build tested in preceding alignment task(23native/138Python).
