#194246 snapshot comparison

Continuation correction: user requested raw AND EMA selections, not raw alone. Add current/best-loss EMA99 (u23300/u19100) and EMA999 (u23300/u19300), freeze before inference, retain three completed raw evaluations. No separate EMA best-restarts checkpoints exist. Same approved runner, same GPU0–5 resources and policy. Commit current code after launch as requested.

Current task: current/best-loss/best-restarts with winning tracer. Raw filenames selected (EMA not requested); files exist. Freeze all before evaluation, then use exact winner config with model-specific namespace, GPU0–5 and unchanged resources. Previous work retained in durable reports/changelog. User asked commit status: current eval switches/runners/results are uncommitted,HEAD5d76a43ad; reported honestly, no commit requested yet.

Independent review approved. All3raw selections frozen before any evaluation: currentu23300 SHA f58ea7ce129b8f57a3882350f767e13987ed457ec4e07dfaa93582e35d28cbac; best-lossu20000 SHA b7da239165f7625181f8eefa11d3fe1f2af54e7f97b926a8445caee3a36c8315; best-restartsu17600 SHA f834bfc47f0e76e3166f4bc6ceacfb91f2936548e203cc11be007dc660fcfa00. Captured2026-10-07T11:47:44–49UTC; snapshots distinct. Selection best-restarts was26on32pieces64directions under training protocol, not a score for this full tuned evaluation.

Extracted shared collect_filtered_result from existing tuning runner and reused its report renderer (parameterized destination/title/notes), no duplicate inference/tracing/metrics. New reviewed_194246 approved-wrapper action.30config/selection/metric tests passed1.85s. Exactwinning args/cohort/nativepolicy checks retained; inference namespace modelhash-specific. Starting current thenbestloss thenbestrestarts.

Startup caught namespace length>64: NAME+fullSHA too long; no tracing ran. Fixed short label nw194246_+first32SHA (fullhash remains in actual modelidentity),added64limit regression. Preserve failedworkload as workload.invalid_namespace.json only when sole difference is invalidnamespace and no summary exists; frozen snapshots unchanged. Resume via sameapprovedcommand.

Preparedbindings also retain immutable namespace evidence; rather than weaken that check, removed the one-off workload repair and separate outputs into current_eval/best_loss_eval/best_restarts_eval under experimentroot. Originalfailedstartup evidence and frozen snapshots remain untouched. No inferred chunks existed from failedstartup.

Currentrawu23300 completed182directions with exactmatchednativepolicy/cohort. min512:4errors291pieces77.106850%,virtualmean6.362121mm,piecemean3.672977mm,median2.646842mm. PriorEMA999u12100:0errors233pieces82.086156%,13.078831/4.883512/3.814470mm. Reported interimcomparison; bestlossu20000 automatically next.

Bestlossrawu20000 completed182directions,matchedpolicy/cohort. min512:3errors270pieces78.988595%,virtualmean8.050887mm,piecemean4.055262mm,median2.710469mm.4errors beforelengthfilter;1discardederrorpiece. Interimtable reported; bestrestartsu17600 next.

All three evaluations completed successfully. Best-restarts raw u17600: 4 errors, 264 kept pieces, 76.844195% retained, virtual mean 8.321840 mm, piece mean 4.034831 mm, median 3.082824 mm. Before filtering: 4 errors, 647 pieces, 85.654% retained. Previous EMA999 u12100 remains better overall under this fixed policy (0 errors, 233 pieces, 82.086% retained); note its PHerc0332 retained coverage is zero after filtering, whereas the new models retain some coverage there. No new-model EMA comparison was performed or implied.

Final report: docs/benchmarks/normalwidth_194246_tuned_20261007_v1.md and JSON, including all min0/min512 and per-scroll metrics. All three cohorts/native policies and shared metric accounting passed validation. Changes remain uncommitted; user asked commit status, not to commit. No additional tuning performed.
