from pathlib import Path
import argparse

import numpy as np
import tifffile
import cv2
import monotone_norm
import torch


def load_label_layer(path: Path, layer: int) -> np.ndarray:
	with tifffile.TiffFile(str(path)) as tif:
		series = tif.series[0]
		if len(series.shape) == 2:
			if layer != 0:
				raise ValueError(f"Requested layer {layer} but TIFF is single-layer with shape {series.shape}")
			arr = series.asarray()
		else:
			if layer < 0 or layer >= series.shape[0]:
				raise ValueError(f"Requested layer {layer} out of range for TIFF with {series.shape[0]} layers")
			arr = series.asarray(key=layer)
	if arr.dtype != np.uint8:
		arr = arr.astype(np.uint8)
	return arr


def compute_signed_distance(labels: np.ndarray) -> np.ndarray:
	if labels.ndim != 2:
		raise ValueError(f"Expected 2D labels, got shape {labels.shape}")

	valid = labels != 2
	fg = (labels == 1) & valid  # foreground / surface
	bg = (labels == 0) & valid  # background

	# Distance to foreground (surface): zeros at fg, non-zeros elsewhere.
	src_fg = np.where(fg, 0, 255).astype(np.uint8)
	dist_to_fg = cv2.distanceTransform(src_fg, distanceType=cv2.DIST_L2, maskSize=3)

	# Distance to background: zeros at bg, non-zeros elsewhere.
	src_bg = np.where(bg, 0, 255).astype(np.uint8)
	dist_to_bg = cv2.distanceTransform(src_bg, distanceType=cv2.DIST_L2, maskSize=3)

	signed = np.zeros_like(dist_to_fg, dtype=np.float32)
	signed[bg] = dist_to_fg[bg]      # positive outside foreground
	signed[fg] = -dist_to_bg[fg]     # negative inside foreground

	# Pixels with label == 2 remain at 0 in the signed map.

	if not np.any(valid):
		return signed

	max_abs = np.max(np.abs(signed[valid]))
	if max_abs == 0:
		return signed

	# Normalize to [-1, 1] over valid pixels.
	return signed / max_abs


def to_visual_uint8(norm_signed: np.ndarray) -> np.ndarray:
	# Map [-1,1] -> [0,255], with 0 distance around mid-gray (127/128).
	vis = (norm_signed + 1.0) * 0.5
	vis = np.clip(vis, 0.0, 1.0)
	vis = (vis * 255.0).round().astype(np.uint8)
	return vis


def compute_label_supervision(labels_in: np.ndarray, dbg: bool = False) -> dict:
	"""
	Core geometry/supervision pipeline.

	Takes a 2D label map with values {0,1,2} (2 = ignore) and computes:
	- signed distance + visualization
	- monotone-normalized distance fields and derived vis
	- connected-component structure
	- fractional position field frac_pos with same semantics as vis_frac_pos.tif
	- large outer CC masks eroded by 16 px with contiguous indices 1..K
	Returns all intermediate visualizations plus:
	- "frac_pos": float32 (H,W)
	- "outer_cc_idx": int32 (H,W)
	- "max_cc_idx": int
	"""

	if labels_in.ndim != 2:
		raise ValueError(f"expected 2D labels, got shape {labels_in.shape}")

	labels = labels_in.astype(np.uint8, copy=True)

	# Remove a 3-pixel border by marking it as ignore (label 2), same as CLI.
	labels[:3, :] = 2
	labels[-3:, :] = 2
	labels[:, :3] = 2
	labels[:, -3:] = 2

	# Connected components on original labels (everything except ignore=2).
	cc_mask = (labels != 2).astype(np.uint8)
	num_cc, cc = cv2.connectedComponents(cc_mask)
	if num_cc > 1:
		cc_vis = (cc.astype(np.float32) / float(num_cc - 1) * 255.0).astype(np.uint8)
	else:
		cc_vis = np.zeros_like(labels, dtype=np.uint8)

	# Second CC pass: label and background as separate component sets (ignore=2 stays background).
	label_mask = (labels == 1).astype(np.uint8)
	bg_mask = (labels == 0).astype(np.uint8)
	num_fg_cc, cc_fg = cv2.connectedComponents(label_mask)
	num_bg_cc, cc_bg = cv2.connectedComponents(bg_mask)
	combined_cc = np.zeros_like(labels, dtype=np.int32)
	if num_fg_cc > 1:
		combined_cc[labels == 1] = cc_fg[labels == 1]
	if num_bg_cc > 1:
		combined_cc[labels == 0] = cc_bg[labels == 0] + (num_fg_cc - 1)
	total_cc = (num_fg_cc - 1) + (num_bg_cc - 1)
	if total_cc > 0:
		cc_sep_vis = (combined_cc.astype(np.float32) / float(total_cc) * 255.0).astype(np.uint8)
	else:
		cc_sep_vis = np.zeros_like(labels, dtype=np.uint8)

	# Monotone-normalized field using one-sided distances from skeletons in label and inverse.
	# Forward skeleton: inside label (labels == 1).
	fg = labels == 1
	fg_u8 = (fg.astype(np.uint8) * 255)
	skel_fg = cv2.ximgproc.thinning(fg_u8)
	src_skel_fg = np.where(skel_fg > 0, 0, 255).astype(np.uint8)
	dist_fg = cv2.distanceTransform(
		src_skel_fg,
		distanceType=cv2.DIST_L2,
		maskSize=3,
	).astype(np.float32)

	# Inverse skeleton: inside background (labels == 0).
	bg = labels == 0
	bg_u8 = (bg.astype(np.uint8) * 255)
	skel_bg = cv2.ximgproc.thinning(bg_u8)
	src_skel_bg = np.where(skel_bg > 0, 0, 255).astype(np.uint8)
	dist_bg = cv2.distanceTransform(
		src_skel_bg,
		distanceType=cv2.DIST_L2,
		maskSize=3,
	).astype(np.float32)

	skel_fg_vis = (skel_fg > 0).astype(np.uint8) * 255
	skel_bg_vis = (skel_bg > 0).astype(np.uint8) * 255

	# Monotone normalization on both distance fields.
	mono_fg = monotone_norm.compute(dist_fg.astype(np.float32, copy=False)).astype(np.float32, copy=False)
	mono_bg_raw = monotone_norm.compute(dist_bg.astype(np.float32, copy=False)).astype(np.float32, copy=False)
	mono_bg = 1.0 - mono_bg_raw

	# Single-sided visualizations.
	mono_fg_vis = (np.clip(mono_fg, 0.0, 1.0) * 255.0).round().astype(np.uint8)
	mono_bg_vis = (np.clip(mono_bg, 0.0, 1.0) * 255.0).round().astype(np.uint8)

	# Iterative weighted averaging between mono_fg and mono_bg.
	combined = np.full_like(mono_fg, 0.5, dtype=np.float32)
	iter_vis = []
	for _ in range(5):
		w_fg = combined
		w_bg = 1.0 - combined
		combined = w_fg * mono_fg + w_bg * mono_bg
		iter_vis.append((np.clip(combined, 0.0, 1.0) * 255.0).round().astype(np.uint8))

	mono = combined
	mono_vis = (np.clip(mono, 0.0, 1.0) * 255.0).round().astype(np.uint8)
	
	# Analyze label-only components inside each first-pass CC:
	# 1) require each label-CC to touch ignore (2) in at least two disjoint outside regions;
	# 2) build connectivity graph between inner components; print neighbors per component;
	# 3) derive per-pixel fractional order along the chain.
	valid_first_cc = np.zeros(num_cc, dtype=np.bool_)
	label_order_map = np.zeros_like(labels, dtype=np.int32)
	frac_step = np.full_like(labels, -1.0, dtype=np.float32)
	frac_pos = np.full_like(labels, -1.0, dtype=np.float32)
	offset_base = 0.0

	# Outer CC index map (eroded by 16 px per valid outer CC), contiguous ids 1..K.
	# Stored as uint8: 0 = background, 1..K = eroded outer CC ids.
	outer_cc_idx = np.zeros_like(labels, dtype=np.uint8)
	outer_counter = 0
	outer_kernel = np.ones((33, 33), np.uint8)

	H, W = labels.shape

	if dbg:
		dbg_print = print
	else:
		def dbg_print(*args, **kwargs):
			return
	
	for big_id in range(1, num_cc):
		mask_big = cc == big_id
		if not np.any(mask_big & (labels == 1)):
			continue

		label_ids = np.unique(combined_cc[mask_big & (labels == 1)])
		label_ids = label_ids[label_ids > 0]
		if label_ids.size == 0:
			continue

		ok = True
		for lid in label_ids:
			mask_label = combined_cc == lid

			# Ignore-neighbor regions: ignore label (2) adjacent to this label region.
			dil = cv2.dilate(mask_label.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
			touch = (dil.astype(bool) & (labels == 2))
			num_touch_cc, _ = cv2.connectedComponents(touch.astype(np.uint8))

			# Need at least two disjoint outside-contact regions.
			if num_touch_cc - 1 < 2:
				ok = False
				break

		if not ok:
			continue

		valid_first_cc[big_id] = True

		# Build adjacency between all inner components (label & bg) within this first-pass CC.
		big_comp_ids = np.unique(combined_cc[mask_big])
		big_comp_ids = big_comp_ids[big_comp_ids > 0]
		neighbors = {int(cid): set() for cid in big_comp_ids}

		# 4-connected adjacency within this big CC, using full-size arrays.
		for y in range(H - 1):
			for x in range(W - 1):
				if not mask_big[y, x]:
					continue

				id00 = int(combined_cc[y, x])
				id10 = int(combined_cc[y + 1, x])
				id01 = int(combined_cc[y, x + 1])

				if id00 > 0 and id10 > 0 and id00 != id10:
					if id00 in neighbors and id10 in neighbors:
						neighbors[id00].add(id10)
						neighbors[id10].add(id00)

				if id00 > 0 and id01 > 0 and id00 != id01:
					if id00 in neighbors and id01 in neighbors:
						neighbors[id00].add(id01)
						neighbors[id01].add(id00)

		dbg_print(f"First CC {big_id}:")
		for cid in sorted(neighbors.keys()):
			nbs = sorted(neighbors[cid])
			dbg_print(f"  inner CC {cid}: neighbors {nbs}")
	
		# Connectivity-based ordering: follow a simple chain if it exists.
		comp_ids = sorted(neighbors.keys())
		deg1 = [cid for cid in comp_ids if len(neighbors[cid]) == 1]
		if not deg1:
			dbg_print("  no endpoint with single neighbor, skipping ordering")
			continue

		start = deg1[0]
		order = []
		visited = set()
		current = start

		while True:
			order.append(current)
			visited.add(current)
			unvisited_nbs = [n for n in neighbors[current] if n not in visited]
			if not unvisited_nbs:
				break
			if len(unvisited_nbs) != 1:
				# Branching or ambiguity: invalid chain.
				order = []
				break
			current = unvisited_nbs[0]

		if not order or len(order) != len(comp_ids):
			dbg_print("  invalid or incomplete chain, skipping this first CC")
			continue
	
		dbg_print(f"  order: {order}")
	
		chain = list(order)
		if not chain:
			dbg_print("  empty chain, skipping")
			continue
	
		L = len(chain)
		if L <= 3:
			dbg_print("  chain too short for segment skipping rules, skipping")
			continue

		# FG/BG flag per component in chain (majority vote).
		is_fg_comp = {}
		for cid in chain:
			mask_c = combined_cc == cid
			fg_count = np.count_nonzero(mask_c & (labels == 1))
			bg_count = np.count_nonzero(mask_c & (labels == 0))
			is_fg_comp[cid] = fg_count >= bg_count

		# Per-pixel fractional order along chain with segment-level skipping:
		# - Each inner CC normally contributes two zones (lower/upper).
		# - First and last CC only have one zone and are fully skipped.
		# - For the 2nd CC, we skip the first half (towards the start).
		# - For the second-last CC, we skip the second half (towards the end).
		eps = 1e-6
		zone_counter = 0
		for idx, cid in enumerate(chain):
			# Skip first and last CC entirely.
			if idx == 0 or idx == L - 1:
				continue

			mask_c = (combined_cc == cid) & mask_big
			if not np.any(mask_c):
				continue

			# Decide lower vs upper half via distance to prev/next components (only for splitting).
			prev_cid = chain[idx - 1] if idx > 0 else None
			next_cid = chain[idx + 1] if idx + 1 < L else None

			lower_mask = np.zeros_like(mask_c, dtype=bool)
			upper_mask = np.zeros_like(mask_c, dtype=bool)

			if prev_cid is not None and next_cid is not None:
				src_prev = np.where(combined_cc == prev_cid, 0, 255).astype(np.uint8)
				d_prev_full = cv2.distanceTransform(src_prev, cv2.DIST_L2, 3)
				src_next = np.where(combined_cc == next_cid, 0, 255).astype(np.uint8)
				d_next_full = cv2.distanceTransform(src_next, cv2.DIST_L2, 3)
				lower_mask[mask_c] = d_prev_full[mask_c] <= d_next_full[mask_c]
				upper_mask[mask_c] = ~lower_mask[mask_c]
			elif prev_cid is not None:
				lower_mask[mask_c] = True
				upper_mask[mask_c] = False
			elif next_cid is not None:
				lower_mask[mask_c] = False
				upper_mask[mask_c] = True
			else:
				lower_mask[mask_c] = True
				upper_mask[mask_c] = False

			use_lower = True
			use_upper = True

			# Skip first half of the second CC (towards the start).
			if idx == 1:
				use_lower = False

			# Skip second half of the second-last CC (towards the end).
			if idx == L - 2:
				use_upper = False

			base = offset_base + 0.5 * zone_counter
			zone_counter += 1

			# Four cases for smooth ramp:
			# (fg, lower), (fg, upper), (bg, lower), (bg, upper)
			# upper half of fg and lower half of bg use an inverted ramp compared to the others.
			if use_lower and np.any(lower_mask & mask_c):
				mask_zone = lower_mask & mask_c
				mono_zone = mono[mask_zone].astype(np.float32, copy=False)
				if is_fg_comp[cid]:
					# fg lower: normal ramp
					t_zone = mono_zone * 0.5 - 0.5
				else:
					# bg lower: inverted ramp
					t_zone = -mono_zone * 0.5
				frac_pos[mask_zone] = base + t_zone

			if use_upper and np.any(upper_mask & mask_c):
				base = base + 0.25
				mask_zone = upper_mask & mask_c
				mono_zone = mono[mask_zone].astype(np.float32, copy=False)
				if is_fg_comp[cid]:
					# fg upper: inverted ramp
					t_zone = 0.25 - mono_zone * 0.5
				else:
					# bg upper: normal ramp
					t_zone = mono_zone * 0.5 - 0.25
				frac_pos[mask_zone] = base + t_zone

		# Store discrete order index (1..N) per component id for debugging,
		# counting only CCs that participate in winding (exclude first and last).
		order_idx = 1
		for idx, cid in enumerate(chain):
			if idx == 0 or idx == L - 1:
				continue
			label_order_map[combined_cc == cid] = order_idx
			order_idx += 1

		# Outer CC mask for this valid chain: erode by 16 px then assign next index.
		eroded = cv2.erode(mask_big.astype(np.uint8), outer_kernel, iterations=1) > 0
		if np.any(eroded):
			outer_counter += 1
			outer_cc_idx[eroded] = outer_counter

	max_cc_idx = outer_counter

	# Mask frac_pos & frac_step outside eroded outer CCs so supervision is only defined
	# inside outer_cc_idx>0.
	mask_outer = outer_cc_idx > 0
	frac_pos[~mask_outer] = -1.0
	frac_step[~mask_outer] = -1.0

	# Signed distance + visualization.
	signed = compute_signed_distance(labels)
	signed_vis = to_visual_uint8(signed)

	# Debug image: keep only pixels near half-step boundaries (multiples of 0.5),
	# computed from the stepped field.
	bins = np.floor(frac_step * 2.0 + 1e-6).astype(np.int32)
	bins[frac_step < 0.0] = -1
	neigh_min = np.full_like(bins, 1000000, dtype=np.int32)
	neigh_max = np.full_like(bins, -1000000, dtype=np.int32)

	for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
		shifted = np.full_like(bins, -1, dtype=np.int32)
		if dy == 0 and dx == 1:
			shifted[:, 1:] = bins[:, :-1]
		elif dy == 0 and dx == -1:
			shifted[:, :-1] = bins[:, 1:]
		elif dy == 1 and dx == 0:
			shifted[1:, :] = bins[:-1, :]
		elif dy == -1 and dx == 0:
			shifted[:-1, :] = bins[1:, :]

		valid = shifted != -1
		neigh_min[valid] = np.minimum(neigh_min[valid], shifted[valid])
		neigh_max[valid] = np.maximum(neigh_max[valid], shifted[valid])

	mask = (bins != -1) & ((neigh_max - neigh_min) > 0)
	frac_dbg = np.where(mask, frac_step, 0.0).astype(np.float32)

	iters_stack = np.stack(iter_vis, axis=0) if iter_vis else None

	return {
		"labels": labels,
		"signed": signed,
		"signed_vis": signed_vis,
		"mono": mono,
		"mono_vis": mono_vis,
		"mono_fg_vis": mono_fg_vis,
		"mono_bg_vis": mono_bg_vis,
		"mono_iters": iters_stack,
		"skel_fg_vis": skel_fg_vis,
		"skel_bg_vis": skel_bg_vis,
		"cc_vis": cc_vis,
		"cc_sep_vis": cc_sep_vis,
		"frac_step": frac_step,
		"frac_pos": frac_pos,
		"frac_dbg": frac_dbg,
		"outer_cc_idx": outer_cc_idx,
		"max_cc_idx": max_cc_idx,
	}


def compute_frac_pos_batch_from_labels(labels_tensor: "torch.Tensor"):
	"""
	Batch API for external callers (e.g. training code).
 
	Accepts a PyTorch tensor of labels:
	- shape (N,H,W) or (N,1,H,W)
	- values {0,1,2}, with 2 = ignore
 
	Returns:
	- frac_pos_batch: (N,H,W) float32 tensor
	- outer_cc_idx_batch: (N,H,W) uint8 tensor (0 outside outer CCs, 1..K inside, per-image)
	- max_cc_idx: int, maximum CC index used over the entire batch
	- mono_batch: (N,H,W) float32 tensor, combined monotone field
	"""
 
	if not isinstance(labels_tensor, torch.Tensor):
		raise TypeError("labels_tensor must be a torch.Tensor")
 
	if labels_tensor.dim() == 4:
		if labels_tensor.size(1) != 1:
			raise ValueError(f"expected (N,1,H,W) labels, got {labels_tensor.shape}")
		labels_np = labels_tensor[:, 0].detach().cpu().numpy()
	elif labels_tensor.dim() == 3:
		labels_np = labels_tensor.detach().cpu().numpy()
	else:
		raise ValueError(f"expected labels tensor of shape (N,H,W) or (N,1,H,W), got {labels_tensor.shape}")
 
	N = labels_np.shape[0]
	frac_pos_list = []
	outer_idx_list = []
	max_idx_list = []
	mono_list = []
 
	for i in range(N):
		sup = compute_label_supervision(labels_np[i])
		frac_pos_list.append(sup["frac_pos"])
		outer_idx_list.append(sup["outer_cc_idx"])
		max_idx_list.append(int(sup["max_cc_idx"]))
		mono_list.append(sup["mono"])
 
	if frac_pos_list:
		frac_pos_arr = np.stack(frac_pos_list, axis=0).astype(np.float32, copy=False)
		outer_arr = np.stack(outer_idx_list, axis=0).astype(np.uint8, copy=False)
		mono_arr = np.stack(mono_list, axis=0).astype(np.float32, copy=False)
		batch_max_idx = max(max_idx_list)
	else:
		frac_pos_arr = np.zeros((0, 0, 0), dtype=np.float32)
		outer_arr = np.zeros((0, 0, 0), dtype=np.uint8)
		mono_arr = np.zeros((0, 0, 0), dtype=np.float32)
		batch_max_idx = 0
 
	frac_pos_batch = torch.from_numpy(frac_pos_arr)
	outer_cc_idx_batch = torch.from_numpy(outer_arr)
	mono_batch = torch.from_numpy(mono_arr)
 
	return frac_pos_batch, outer_cc_idx_batch, batch_max_idx, mono_batch


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Generate signed-distance supervision visualization from a label TIFF layer.",
	)
	parser.add_argument("label_tif", type=str, help="Path to label TIFF file.")
	parser.add_argument("--layer", type=int, default=0, help="Layer index to read (0-based).")
	parser.add_argument(
		"--out",
		type=str,
		default=None,
		help="Output TIFF path for visualization (default: vis.tif next to input).",
	)

	args = parser.parse_args()

	label_path = Path(args.label_tif)
	if not label_path.is_file():
		raise SystemExit(f"Label TIFF does not exist: {label_path}")

	labels = load_label_layer(label_path, args.layer)
	results = compute_label_supervision(labels, dbg=True)

	# Keep original lbl.tif dump for debugging.
	tifffile.imwrite("lbl.tif", results["labels"])

	if args.out is not None:
		out_path = Path(args.out)
	else:
		out_path = label_path.with_name("vis.tif")

	# Signed-distance visualization.
	tifffile.imwrite(str(out_path), results["signed_vis"])

	# Also write the monotone-normalized visualization next to the main vis.
	mono_out_path = out_path.with_name("vis_monotone.tif")
	tifffile.imwrite(str(mono_out_path), results["mono_vis"])

	# Write single-sided and iterative visualizations.
	mono_fg_out_path = out_path.with_name("vis_monotone_fg.tif")
	tifffile.imwrite(str(mono_fg_out_path), results["mono_fg_vis"])

	mono_bg_out_path = out_path.with_name("vis_monotone_bg.tif")
	tifffile.imwrite(str(mono_bg_out_path), results["mono_bg_vis"])

	if results["mono_iters"] is not None:
		mono_iters_out_path = out_path.with_name("vis_monotone_iters.tif")
		tifffile.imwrite(str(mono_iters_out_path), results["mono_iters"])

	# Write skeletons.
	skel_fg_out_path = out_path.with_name("vis_skeleton_fg.tif")
	tifffile.imwrite(str(skel_fg_out_path), results["skel_fg_vis"])

	skel_bg_out_path = out_path.with_name("vis_skeleton_bg.tif")
	tifffile.imwrite(str(skel_bg_out_path), results["skel_bg_vis"])

	# Write connected components on original labels.
	cc_out_path = out_path.with_name("vis_labels_cc.tif")
	tifffile.imwrite(str(cc_out_path), results["cc_vis"])

	# Write second CC pass (label/bg separated, ignore=2 as background).
	cc_sep_out_path = out_path.with_name("vis_labels_cc_sep.tif")
	tifffile.imwrite(str(cc_sep_out_path), results["cc_sep_vis"])

	# Write stepped & fractional order fields as float32 TIFFs.
	frac_step_out_path = out_path.with_name("vis_frac_step.tif")
	tifffile.imwrite(str(frac_step_out_path), results["frac_step"].astype(np.float32))

	frac_out_path = out_path.with_name("vis_frac_pos.tif")
	tifffile.imwrite(str(frac_out_path), results["frac_pos"].astype(np.float32))

	# Debug image near half-step boundaries.
	frac_dbg_out_path = out_path.with_name("vis_frac_pos_dbg.tif")
	tifffile.imwrite(str(frac_dbg_out_path), results["frac_dbg"].astype(np.float32))


if __name__ == "__main__":
	main()
