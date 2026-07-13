import { App, applyDocumentTheme, applyHostFonts, applyHostStyleVariables, type McpUiHostContext } from "@modelcontextprotocol/ext-apps";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import "./inspector.css";

type J = Record<string, any>;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const views = ["prediction-view", "surface-view", "segmentation-view", "registered-view", "geometry-view", "alignment-view", "grid-view", "comparison-view", "fold-view", "stability-view", "ranking-view", "review-view", "assessment-view", "evaluation-view", "artifact-view", "empty-view"];
let prediction: J | null = null;
let selected: J | null = null;
let reviewQueueJob: string | null = null;

function show(id: string) { views.forEach(v => $(v).classList.toggle("hidden", v !== id)); }
function value(id: string) { return Number(($<HTMLInputElement>(id)).value); }
function setStatus(text: string, active = false) { $("status").textContent = text; $("status").classList.toggle("active", active); }
function structured(result: CallToolResult): J { return (result.structuredContent as J) ?? {}; }
function compactUri(uri: string) { const parts = uri.replace(/\/$/, "").split("/"); return parts.at(-1) ?? uri; }

function renderPrediction(data: J) {
  prediction = data.prediction ?? data;
  show("prediction-view");
  $("volume-name").textContent = compactUri(prediction!.uri);
  $("shape").textContent = prediction!.shape_zyx.join(" × ");
  $("chunks").textContent = prediction!.chunks_zyx.join("³ · ");
  $("dtype").textContent = `${prediction!.dtype} / ${prediction!.compressor}`;
  $("space").textContent = prediction!.space;
  $<HTMLInputElement>("prediction-uri").value = prediction!.uri;
  const shape = prediction!.shape_zyx;
  $<HTMLInputElement>("cx").value = String(Math.floor(shape[2] / 2));
  $<HTMLInputElement>("cy").value = String(Math.floor(shape[1] / 2));
  $<HTMLInputElement>("cz").value = String(Math.floor(shape[0] / 2));
  setStatus("prediction loaded", true);
  if (data.candidates) renderCandidates(data);
}

function renderCandidates(data: J) {
  renderPredictionOnly(data.prediction);
  $("candidate-section").classList.remove("hidden");
  $("candidate-summary").textContent = `${data.candidates.length} retained · ${data.foreground_voxels.toLocaleString()} foreground voxels · ${data.chunks_read} chunks`;
  const list = $("candidate-list"); const map = $("candidate-map");
  list.innerHTML = ""; map.querySelectorAll(".dot").forEach(x => x.remove());
  const region = data.region; const minX = region.center.x - region.radius.x; const minY = region.center.y - region.radius.y;
  data.candidates.forEach((c: J, index: number) => {
    const button = document.createElement("button"); button.className = "candidate";
    button.innerHTML = `<b>${String(index + 1).padStart(2, "0")}</b><span>${c.coordinate.x}, ${c.coordinate.y}, ${c.coordinate.z}</span><i>${Math.round(c.combined_score * 100)}%</i>`;
    button.onclick = () => selectCandidate(c, button); list.appendChild(button);
    const dot = document.createElement("button"); dot.className = "dot"; dot.title = `Candidate ${index + 1}`;
    dot.style.left = `${Math.max(2, Math.min(98, (c.coordinate.x - minX) / (region.radius.x * 2) * 100))}%`;
    dot.style.top = `${Math.max(2, Math.min(98, (c.coordinate.y - minY) / (region.radius.y * 2) * 100))}%`;
    dot.style.setProperty("--score", String(c.combined_score)); dot.onclick = () => { button.click(); button.scrollIntoView({ block: "nearest" }); }; map.appendChild(dot);
  });
  if (data.candidates[0]) (list.firstElementChild as HTMLButtonElement).click();
  setStatus("candidate scan complete", true);
}
function renderPredictionOnly(data: J) { prediction = data; show("prediction-view"); $("volume-name").textContent = compactUri(data.uri); $("shape").textContent = data.shape_zyx.join(" × "); $("chunks").textContent = data.chunks_zyx.join("³ · "); $("dtype").textContent = `${data.dtype} / ${data.compressor}`; $("space").textContent = data.space; $<HTMLInputElement>("prediction-uri").value = data.uri; }

function selectCandidate(candidate: J, button: HTMLElement) {
  selected = candidate; document.querySelectorAll(".candidate.selected").forEach(x => x.classList.remove("selected")); button.classList.add("selected");
  $("seed-payload").textContent = JSON.stringify({ seed: candidate.coordinate, prediction_space: prediction?.space, surface_score: candidate.surface_score, ink_score: candidate.ink_score }, null, 2);
}

function renderSurface(data: J) {
  show("surface-view"); $("surface-job").textContent = data.job_id; const img = $<HTMLImageElement>("surface-preview");
  if (data.generation_preview) { img.src = data.generation_preview; img.style.display = "block"; } else img.style.display = "none";
  const keys: [string, any][] = [["Area", `${Number(data.meta.area_cm2 ?? 0).toFixed(4)} cm²`], ["Generation", data.meta.max_gen ?? "—"], ["Grid scale", (data.meta.scale ?? []).join(" × ")], ["Mode", data.meta.vc_gsfs_mode ?? "—"], ["Target", data.meta.target_volume ?? "—"]];
  $("surface-metrics").innerHTML = keys.map(([k,v]) => `<div><span>${k}</span><strong>${v}</strong></div>`).join(""); setStatus("surface rendered", true);
}

function renderSegmentation(data: J) {
  show("segmentation-view");
  $("segmentation-job").textContent = `${data.job_id} · ${data.state ?? "succeeded"}`;
  const progress = $("segmentation-progress"); progress.classList.add("done"); progress.querySelector("b")!.textContent = "complete";
  const probability = $<HTMLImageElement>("probability-preview"); const mask = $<HTMLImageElement>("mask-preview");
  probability.src = data.probability_preview ?? ""; mask.src = data.mask_preview ?? "";
  const m = data.manifest ?? {};
  const entries = [["Model",m.model],["Backend",m.backend],["Input ZYX",(m.input_shape_zyx??[]).join(" × ")],["Tile",m.tile_size],["Overlap",m.overlap],["Threshold",m.threshold],["Patches",m.patches]];
  $("segmentation-metrics").innerHTML = entries.map(([k,v]) => `<div><span>${k}</span><strong>${v ?? "—"}</strong></div>`).join("");
  setStatus("segmentation complete", true);
}

async function followSegmentation(job: J) {
  show("segmentation-view"); $("segmentation-job").textContent = job.job_id;
  const progress = $("segmentation-progress"); progress.classList.remove("done"); progress.querySelector("b")!.textContent = job.state ?? "queued";
  setStatus("segmenting volume…", true);
  for (let attempt=0; attempt<720; attempt++) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    const result = structured(await app.callServerTool({name:"vc_get_job",arguments:{job_id:job.job_id}}));
    progress.querySelector("b")!.textContent = result.state;
    const completed=Number(result.progress?.completed??5), total=Number(result.progress?.total??100); const percent=total>0?completed/total*100:5; (progress.querySelector("span") as HTMLElement).style.width = `${percent}%`;
    if (result.state === "succeeded") { renderSegmentation(structured(await app.callServerTool({name:"volume_inspect_segmentation",arguments:{job_id:job.job_id}}))); return; }
    if (["failed","cancelled"].includes(result.state)) { setStatus(`segmentation ${result.state}`); return; }
  }
  setStatus("segmentation polling timed out");
}

function renderRegistered(data: J) {
  show("registered-view");
  $("registered-job").textContent = `${data.job_id} · ${data.state ?? "succeeded"}`;
  $<HTMLImageElement>("registered-intensity").src = data.intensity_preview ?? "";
  $<HTMLImageElement>("registered-depth").src = data.depth_preview ?? "";
  $<HTMLImageElement>("registered-coverage").src = data.coverage_preview ?? "";
  const m = data.manifest ?? {}; const volume = m.volume ?? {};
  const entries = [
    ["Surface VU", (m.surface_shape_vu ?? []).join(" × ")],
    ["Valid pixels", Number(m.valid_surface_pixels ?? 0).toLocaleString()],
    ["Registered", Number(m.registered_pixels ?? 0).toLocaleString()],
    ["Coverage", `${(Number(m.coverage_fraction ?? 0) * 100).toFixed(2)}%`],
    ["Coordinate space", m.coordinate_space],
    ["Geometry axes", (m.geometry_axes ?? []).join(" · ")],
    ["Volume array", volume.array_path],
    ["Sampling", m.sampling],
  ];
  $("registered-metrics").innerHTML = entries.map(([k,v]) => `<div><span>${k}</span><strong>${v ?? "—"}</strong></div>`).join("");
  setStatus("surface registered", true);
}

async function followRegistered(job: J) {
  show("registered-view"); $("registered-job").textContent = `${job.job_id} · ${job.state ?? "queued"}`;
  setStatus("building registered surface bundle…", true);
  for (let attempt=0; attempt<720; attempt++) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    const result = structured(await app.callServerTool({name:"vc_get_job",arguments:{job_id:job.job_id}}));
    $("registered-job").textContent = `${job.job_id} · ${result.state}`;
    if (result.state === "succeeded") { renderRegistered(structured(await app.callServerTool({name:"surface_inspect_registered_render",arguments:{job_id:job.job_id}}))); return; }
    if (["failed","cancelled"].includes(result.state)) { setStatus(`registration ${result.state}`); return; }
  }
  setStatus("registration polling timed out");
}

function renderGeometry(data: J) {
  show("geometry-view"); const m=data.manifest??{}; $("geometry-job").textContent=`${data.job_id} · ${data.state??"succeeded"}`;
  $<HTMLImageElement>("geometry-stretch").src=data.stretch_preview??""; $<HTMLImageElement>("geometry-normal").src=data.normal_preview??""; $<HTMLImageElement>("geometry-fold").src=data.fold_preview??"";
  const entries=[["Surface VU",(m.surface_shape_vu??[]).join(" × ")],["Valid pixels",Number(m.valid_pixels??0).toLocaleString()],["Median edge U / V",`${Number(m.median_edge_length_u_voxels??0).toFixed(3)} / ${Number(m.median_edge_length_v_voxels??0).toFixed(3)}`],["P95 stretch",Number(m.p95_stretch_log_ratio??0).toFixed(4)],["P95 normal Δ",`${Number(m.p95_normal_change_degrees??0).toFixed(2)}°`],["Fold / degenerate",Number(m.fold_or_degenerate_cells??0).toLocaleString()],["Components",Number(m.connected_components??0).toLocaleString()],["Enclosed holes",Number(m.enclosed_holes??0).toLocaleString()],["Boundary pixels",Number(m.boundary_pixels??0).toLocaleString()],["Global intersections",m.global_self_intersection_tested?"tested":"not tested"]];
  $("geometry-metrics").innerHTML=entries.map(([k,v])=>`<div><span>${k}</span><strong>${v}</strong></div>`).join(""); setStatus("geometry measured",true);
}
function renderAlignment(data: J) {
  show("alignment-view"); const m=data.manifest??{}; $("alignment-job").textContent=`${data.job_id} · ${data.state??"succeeded"}`;
  $<HTMLImageElement>("alignment-gradient").src=data.gradient_preview??""; $<HTMLImageElement>("alignment-offset").src=data.offset_preview??""; $<HTMLImageElement>("alignment-confidence").src=data.confidence_preview??"";
  const entries=[["Surface VU",(m.surface_shape_vu??[]).join(" × ")],["Normal range",`±${m.maximum_offset_voxels??0} vox`],["Interpolation",m.interpolation],["Supported",Number(m.supported_pixels??0).toLocaleString()],["Support",`${(Number(m.support_fraction??0)*100).toFixed(2)}%`],["Median gradient",Number(m.median_peak_gradient??0).toFixed(3)],["Median offset",`${Number(m.median_peak_offset_voxels??0).toFixed(2)} vox`],["Median evidence",Number(m.median_confidence??0).toFixed(3)]];
  $("alignment-metrics").innerHTML=entries.map(([k,v])=>`<div><span>${k}</span><strong>${v}</strong></div>`).join(""); setStatus("CT alignment measured",true);
}
async function followEvidence(job:J, inspector:string, kind:"geometry"|"alignment") {
  show(`${kind}-view`); $(`${kind}-job`).textContent=`${job.job_id} · ${job.state??"queued"}`; setStatus(kind==="geometry"?"measuring surface geometry…":"sampling CT normal profiles…",true);
  for(let attempt=0;attempt<720;attempt++){await new Promise(resolve=>setTimeout(resolve,1000));const state=structured(await app.callServerTool({name:"vc_get_job",arguments:{job_id:job.job_id}}));$(`${kind}-job`).textContent=`${job.job_id} · ${state.state}`;if(state.state==="succeeded"){const evidence=structured(await app.callServerTool({name:inspector,arguments:{job_id:job.job_id}}));kind==="geometry"?renderGeometry(evidence):renderAlignment(evidence);return}if(["failed","cancelled"].includes(state.state)){setStatus(`${kind} ${state.state}`);return}}setStatus(`${kind} polling timed out`);
}

function renderGrid(data:J){show("grid-view");const m=data.manifest??{},p=m.periods_mm??{},s=m.significance??{};$("grid-job").textContent=`${data.job_id} · ${data.state??"succeeded"}`;$<HTMLImageElement>("grid-coherence").src=data.coherence_preview??"";const entries=[["Letter pitch",p.letters?`${Number(p.letters).toFixed(2)} mm`:"not found"],["Line spacing",p.lines?`${Number(p.lines).toFixed(2)} mm`:"not found"],["Column period",p.columns?`${Number(p.columns).toFixed(2)} mm`:"not found"],["Pixel U / V",(m.pixel_spacing_mm_uv??[]).map((x:number)=>Number(x).toFixed(4)).join(" / ")+" mm"],["Median score",Number(m.score_summary?.median??0).toFixed(3)],["Null trials",s.null_trials??"—"],["Empirical p",Number(s.empirical_p_value??1).toFixed(4)],["Polarity",m.polarity]];$("grid-metrics").innerHTML=entries.map(([k,v])=>`<div><span>${k}</span><strong>${v}</strong></div>`).join("");setStatus("grid coherence measured",true)}
function renderComparison(data:J){show("comparison-view");const m=data.manifest??{};$("comparison-job").textContent=`${data.job_id} · ${data.state??"succeeded"}`;$<HTMLImageElement>("comparison-agreement").src=data.agreement_preview??"";$<HTMLImageElement>("comparison-divergence").src=data.divergence_preview??"";$<HTMLImageElement>("comparison-priority").src=data.priority_preview??"";const entries=[["Map correlation",Number(m.map_correlation??0).toFixed(3)],["A stronger",`${(Number(m.a_greater_fraction??0)*100).toFixed(1)}%`],["B stronger",`${(Number(m.b_greater_fraction??0)*100).toFixed(1)}%`],["Max XYZ Δ",Number(m.maximum_xyz_difference??0).toExponential(2)],["Common letter",m.common_periods_mm?.letters?`${Number(m.common_periods_mm.letters).toFixed(2)} mm`:"—"],["Common line",m.common_periods_mm?.lines?`${Number(m.common_periods_mm.lines).toFixed(2)} mm`:"—"]];$("comparison-metrics").innerHTML=entries.map(([k,v])=>`<div><span>${k}</span><strong>${v}</strong></div>`).join("");setStatus("comparison mapped",true)}
function renderFold(data:J){show("fold-view");const m=data.manifest??{};$("fold-job").textContent=`${data.job_id} · ${data.state??"succeeded"}`;$<HTMLImageElement>("fold-profile").src=data.folded_preview??"";$<HTMLImageElement>("fold-search").src=data.search_preview??"";const entries=[["Input period",`${Number(m.input_line_period_mm??0).toFixed(3)} mm`],["Best period",`${Number(m.best_period_mm??0).toFixed(3)} mm`],["Fold statistic",Number(m.fold_statistic??0).toFixed(3)],["Corrected p",Number(m.look_elsewhere_corrected_empirical_p_value??1).toFixed(4)],["Search steps",m.period_steps],["Phase bins",m.phase_bins],["Null trials",m.null_trials],["Null kind",m.null_kind]];$("fold-metrics").innerHTML=entries.map(([k,v])=>`<div><span>${k}</span><strong>${v}</strong></div>`).join("");setStatus("epoch fold complete",true)}
async function followStructural(job:J,inspector:string,kind:"grid"|"comparison"|"fold"){show(`${kind}-view`);$(`${kind}-job`).textContent=`${job.job_id} · ${job.state??"queued"}`;setStatus(kind==="grid"?"measuring structural grid…":kind==="comparison"?"mapping divergence…":"searching fold period…",true);for(let attempt=0;attempt<720;attempt++){await new Promise(resolve=>setTimeout(resolve,1000));const state=structured(await app.callServerTool({name:"vc_get_job",arguments:{job_id:job.job_id}}));$(`${kind}-job`).textContent=`${job.job_id} · ${state.state}`;if(state.state==="succeeded"){const evidence=structured(await app.callServerTool({name:inspector,arguments:{job_id:job.job_id}}));if(kind==="grid")renderGrid(evidence);else if(kind==="comparison")renderComparison(evidence);else renderFold(evidence);return}if(["failed","cancelled"].includes(state.state)){setStatus(`${kind} ${state.state}`);return}}setStatus(`${kind} polling timed out`)}

function renderStability(data:J){show("stability-view");const m=data.manifest??{},comparisons=m.comparisons??[];$("stability-job").textContent=`${data.job_id} · ${data.state??"succeeded"}`;$<HTMLImageElement>("stability-displacement").src=data.displacement_preview??"";$<HTMLImageElement>("stability-angle").src=data.angle_preview??"";$<HTMLImageElement>("stability-local").src=data.stability_preview??"";const worst=comparisons.reduce((a:J,b:J)=>Number(a.median_local_stability??1)<Number(b.median_local_stability??1)?a:b,comparisons[0]??{});const entries=[["Variants",comparisons.length],["Overall stability",Number(m.overall_stability_score??0).toFixed(3)],["Worst valid IoU",Number(worst.valid_iou??0).toFixed(3)],["Worst median Δ",`${Number(worst.median_displacement_mm??0).toFixed(4)} mm`],["Worst normal Δ",`${Number(worst.median_normal_angle_degrees??0).toFixed(2)}°`],["Signal Δ",Number(worst.median_signal_difference??0).toFixed(3)]];$("stability-metrics").innerHTML=entries.map(([k,v])=>`<div><span>${k}</span><strong>${v}</strong></div>`).join("");setStatus("stability measured",true)}
function renderRanking(data:J){show("ranking-view");const m=data.manifest??{},ranked=m.ranked_candidates??[];$("ranking-job").textContent=`${data.job_id} · ${data.state??"succeeded"}`;$<HTMLImageElement>("ranking-preview").src=data.ranking_preview??"";const weights=m.weights??{};$("ranking-metrics").innerHTML=[["Candidates",ranked.length],["Formula","weighted geometric mean"],["Geometry weight",weights.geometry],["Alignment weight",weights.alignment],["Grid weight",weights.grid],["Stability weight",weights.stability]].map(([k,v])=>`<div><span>${k}</span><strong>${v??"—"}</strong></div>`).join("");$("ranking-candidates").innerHTML=ranked.map((c:J)=>`<article><div><b>${String(c.rank).padStart(2,"0")} · ${c.id}</b><span>${Number(c.combined_score).toFixed(3)}</span></div><code>geometry ${Number(c.components.geometry??0).toFixed(3)} · alignment ${Number(c.components.alignment??0).toFixed(3)} · grid ${Number(c.components.grid??0).toFixed(3)} · stability ${c.components.stability===undefined?"n/a":Number(c.components.stability).toFixed(3)}</code></article>`).join("");setStatus("evidence ranked",true)}
async function followFusion(job:J,inspector:string,kind:"stability"|"ranking"){show(`${kind}-view`);$(`${kind}-job`).textContent=`${job.job_id} · ${job.state??"queued"}`;setStatus(kind==="stability"?"comparing perturbation runs…":"fusing explicit evidence…",true);for(let attempt=0;attempt<720;attempt++){await new Promise(resolve=>setTimeout(resolve,1000));const state=structured(await app.callServerTool({name:"vc_get_job",arguments:{job_id:job.job_id}}));$(`${kind}-job`).textContent=`${job.job_id} · ${state.state}`;if(state.state==="succeeded"){const evidence=structured(await app.callServerTool({name:inspector,arguments:{job_id:job.job_id}}));kind==="stability"?renderStability(evidence):renderRanking(evidence);return}if(["failed","cancelled"].includes(state.state)){setStatus(`${kind} ${state.state}`);return}}setStatus(`${kind} polling timed out`)}

function renderReviewQueue(data:J){show("review-view");reviewQueueJob=data.job_id;const m=data.manifest??{},items=m.items??[];$("review-job").textContent=`${data.job_id} · ${items.length} items`;$<HTMLImageElement>("review-preview").src=data.queue_preview??"";$("review-items").innerHTML=items.map((item:J)=>`<article class="review-card" data-item-id="${item.item_id}"><span class="kind">${item.kind.replaceAll("_"," ")}</span><b>${item.item_id}</b><select aria-label="Decision for ${item.item_id}"><option value="">skip</option><option value="accept">accept</option><option value="reject">reject</option><option value="uncertain">uncertain</option><option value="defer">defer</option></select><input type="number" min="0" max="1" step="0.05" value="0.5" aria-label="Confidence"/><span class="priority">${Number(item.priority).toFixed(3)}</span></article>`).join("");setStatus("review queue ready",true)}
function renderAssessment(data:J){show("assessment-view");const m=data.manifest??{},records=m.records??[];$("assessment-job").textContent=`${data.job_id} · ${m.reviewer_id??"—"}`;$<HTMLImageElement>("assessment-preview").src=data.assessment_preview??"";$("assessment-metrics").innerHTML=[["Reviewer",m.reviewer_id],["Records",records.length],["Queue digest",String(m.queue_digest??"").slice(0,16)+"…"],["Created",m.created_at]].map(([k,v])=>`<div><span>${k}</span><strong>${v??"—"}</strong></div>`).join("");$("assessment-records").innerHTML=records.map((r:J)=>`<article><div><b>${r.item_id}</b><span>${r.decision} · ${Number(r.confidence).toFixed(2)}</span></div><code>${(r.reason_codes??[]).join(" · ")||"no reason code"}</code></article>`).join("");setStatus("assessment recorded",true)}
function renderEvaluation(data:J){show("evaluation-view");const m=data.manifest??{},bins=m.calibration_bins??[];$("evaluation-job").textContent=`${data.job_id} · ${m.record_count??0} labels`;$<HTMLImageElement>("evaluation-preview").src=data.evaluation_preview??"";$("evaluation-metrics").innerHTML=[["Records",m.record_count],["Binary labels",m.binary_record_count],["ROC AUC",m.roc_auc===null?"n/a":Number(m.roc_auc).toFixed(3)],["Average precision",m.average_precision===null?"n/a":Number(m.average_precision).toFixed(3)],["Pair agreement",m.overall_pair_agreement===null?"n/a":Number(m.overall_pair_agreement).toFixed(3)],["Queue digest",String(m.queue_digest??"").slice(0,16)+"…"]].map(([k,v])=>`<div><span>${k}</span><strong>${v??"—"}</strong></div>`).join("");$("evaluation-bins").innerHTML=bins.map((b:J)=>`<article><div><b>${Number(b.priority_min).toFixed(1)}–${Number(b.priority_max).toFixed(1)}</b><span>${b.count} labels</span></div><code>accept fraction ${b.accept_fraction===null?"n/a":Number(b.accept_fraction).toFixed(3)}</code></article>`).join("");setStatus("labels evaluated",true)}
async function followReview(job:J,inspector:string,kind:"review"|"assessment"|"evaluation"){show(`${kind}-view`);$(`${kind}-job`).textContent=`${job.job_id} · ${job.state??"queued"}`;setStatus(kind==="review"?"building review queue…":kind==="assessment"?"writing immutable assessment…":"evaluating supplied labels…",true);for(let attempt=0;attempt<720;attempt++){await new Promise(resolve=>setTimeout(resolve,1000));const state=structured(await app.callServerTool({name:"vc_get_job",arguments:{job_id:job.job_id}}));$(`${kind}-job`).textContent=`${job.job_id} · ${state.state}`;if(state.state==="succeeded"){const evidence=structured(await app.callServerTool({name:inspector,arguments:{job_id:job.job_id}}));if(kind==="review")renderReviewQueue(evidence);else if(kind==="assessment")renderAssessment(evidence);else renderEvaluation(evidence);return}if(["failed","cancelled"].includes(state.state)){setStatus(`${kind} ${state.state}`);return}}setStatus(`${kind} polling timed out`)}

function renderArtifacts(data: J) {
  show("artifact-view"); $("artifact-job").textContent = `${data.job_id} · ${data.state}`;
  const m = data.command_manifest ?? {}; $("manifest").innerHTML = `<div><span>Executable</span><code>${m.executable ?? "—"}</code></div><div><span>Working directory</span><code>${m.working_directory ?? "—"}</code></div><div><span>Arguments</span><code>${(m.arguments ?? []).join(" ")}</code></div>`;
  $("files").innerHTML = (data.files ?? []).map((f:J) => `<article><div><b>${f.name}</b><span>${Number(f.size_bytes).toLocaleString()} bytes</span></div><code>${f.sha256}</code></article>`).join(""); setStatus("checksums verified", true);
}

function route(data: J) { if(data.review_queue)renderReviewQueue(data);else if(data.review_assessment)renderAssessment(data);else if(data.label_evaluation)renderEvaluation(data);else if(data.operation==="review_create_queue"&&data.job_id)void followReview(data,"review_inspect_queue","review");else if(data.operation==="review_record_assessment"&&data.job_id)void followReview(data,"review_inspect_assessment","assessment");else if(data.operation==="metric_evaluate_labels"&&data.job_id)void followReview(data,"metric_inspect_label_evaluation","evaluation");else if(data.surface_stability)renderStability(data);else if(data.evidence_ranking)renderRanking(data);else if(data.operation==="surface_test_stability"&&data.job_id)void followFusion(data,"surface_inspect_stability","stability");else if(data.operation==="surface_rank_evidence"&&data.job_id)void followFusion(data,"surface_inspect_evidence_ranking","ranking");else if(data.structural_grid)renderGrid(data);else if(data.structural_comparison)renderComparison(data);else if(data.epoch_fold)renderFold(data);else if(data.operation==="text_measure_grid_coherence"&&data.job_id)void followStructural(data,"text_inspect_grid_coherence","grid");else if(data.operation==="ink_compare_registered_predictions"&&data.job_id)void followStructural(data,"ink_inspect_registered_comparison","comparison");else if(data.operation==="text_epoch_fold_structure"&&data.job_id)void followStructural(data,"text_inspect_epoch_fold","fold");else if (data.surface_geometry) renderGeometry(data); else if(data.surface_alignment) renderAlignment(data); else if(data.operation==="surface_validate_geometry"&&data.job_id) void followEvidence(data,"surface_inspect_geometry","geometry"); else if(data.operation==="surface_measure_volume_alignment"&&data.job_id) void followEvidence(data,"surface_inspect_volume_alignment","alignment"); else if (data.registered_surface) renderRegistered(data); else if (data.operation === "surface_render_registered_roi" && data.job_id) void followRegistered(data); else if (data.segmentation) renderSegmentation(data); else if (data.operation === "volume_run_segmentation" && data.job_id) void followSegmentation(data); else if (data.candidates) renderCandidates(data); else if (data.shape_zyx) renderPrediction(data); else if (data.generation_preview !== undefined) renderSurface(data); else if (data.files) renderArtifacts(data); }

const app = new App({ name: "Vellum Lens", version: "0.1.0" });
app.ontoolinput = () => setStatus("working…");
app.ontoolresult = result => route(structured(result));
app.onerror = error => { console.error(error); setStatus("tool error"); };
app.onhostcontextchanged = (ctx: McpUiHostContext) => { if (ctx.theme) applyDocumentTheme(ctx.theme); if (ctx.styles?.variables) applyHostStyleVariables(ctx.styles.variables); if (ctx.styles?.css?.fonts) applyHostFonts(ctx.styles.css.fonts); };
app.onteardown = async () => ({});

$("candidate-form").addEventListener("submit", async event => {
  event.preventDefault(); if (!prediction) return; setStatus("reading bounded chunks…");
  const args: J = { prediction_uri: $<HTMLInputElement>("prediction-uri").value, prediction_space: prediction.space, region: { center: { x:value("cx"), y:value("cy"), z:value("cz") }, radius: { x:value("rx"), y:value("ry"), z:value("rz") } }, surface_threshold:value("surface-threshold"), ink_threshold:value("ink-threshold"), ink_weight:value("ink-weight"), minimum_separation_voxels:value("separation"), max_candidates:50 };
  const ink = $<HTMLInputElement>("ink-uri").value.trim(); if (ink) args.ink_prediction_uri = ink;
  try { route(structured(await app.callServerTool({ name:"vc_find_seed_candidates", arguments:args }))); } catch (error) { console.error(error); setStatus("candidate scan failed"); }
});
$("copy-seed").onclick = async () => { if (!selected) return; await navigator.clipboard.writeText($("seed-payload").textContent ?? ""); setStatus("seed copied", true); };
$("submit-assessment").onclick = async () => {
  if (!reviewQueueJob) return;
  const reviewer=$<HTMLInputElement>("reviewer-id").value.trim(); const assessments:J[]=[];
  document.querySelectorAll<HTMLElement>(".review-card").forEach(card=>{const decision=(card.querySelector("select") as HTMLSelectElement).value;if(!decision)return;assessments.push({item_id:card.dataset.itemId,decision,confidence:Number((card.querySelector('input') as HTMLInputElement).value)});});
  if(!reviewer||!assessments.length){setStatus("choose a reviewer and at least one decision");return;}
  try{route(structured(await app.callServerTool({name:"review_record_assessment",arguments:{queue:{job_id:reviewQueueJob,artifact_id:"review-queue"},reviewer_id:reviewer,assessments,client_request_id:`widget-assessment-${Date.now()}`}})));}catch(error){console.error(error);setStatus("assessment submission failed");}
};

app.connect().then(() => { const ctx = app.getHostContext(); if (ctx) app.onhostcontextchanged?.(ctx); });
