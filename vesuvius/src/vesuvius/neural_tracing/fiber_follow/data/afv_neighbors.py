"""Neighboring annotated fibers of an AFV source: foreign masks, certified wrong continuations and switch detection.

These are supervision-only paths, never image inputs.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.fiber_follow.data.components import crop_indices
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, exact_nearest, interp_at


class AFVBank:
    """Nearby annotations of the same AFV catalog, cleared of the intended annotation."""
    def __init__(self, fibers):
        self.fibers = fibers
        self.exclusion = 1.5
        self._trees = OrderedDict()
        self.root = Path(fibers.path)
        self.run = {'mining': {'min_distance': self.exclusion}, 'digest': fibers.metadata['uuid']}
        self._cdf = np.cumsum(fibers.lengths)/fibers.lengths.sum()

    def draw_path(self, rng, *, min_length=0., hard_fraction=0.):
        """Draw a nearby, target-clear continuous path with arc correspondence.

        Parent locations are length weighted; neighbor IDs are sampled without
        duplicate RTree blocks. Whole-target clearance rejects overlapping
        duplicates. Unknown/short/nonmatching geometry is retried or skipped.
        """
        if not 0 <= hard_fraction <= 1 or min_length < 0:
            raise ValueError('Invalid AFV neighbor sampling options')
        from vesuvius.neural_tracing.fiber_follow.data.bank_geometry import difficulty_scores, DIFFICULTY_KINDS
        proposals = []
        hard = rng.random() < hard_fraction
        for _ in range(16):
            fi = int(np.searchsorted(self._cdf, rng.random(), side='right'))
            parent = self.fibers[fi]
            at = float(rng.uniform(0., parent.length))
            anchor = interp_at(parent.points, parent.s, [at])[0]
            radius = 6. if rng.random() < .5 else 32.
            ids = self.fibers.nearby_fiber_ids(anchor, radius, self.fibers.catalog[fi][0])
            if not ids:
                continue
            tree, gap = self._target_tree(fi)
            for neighbor_id in rng.permutation(ids)[:8]:
                neighbor = self.fibers[self.fibers.id_to_index[int(neighbor_id)]]
                j = int(np.argmin(np.linalg.norm(neighbor.points-anchor, axis=1)))
                span = max(160., min_length+32.)
                a, b = max(0., neighbor.s[j]-span), min(neighbor.length, neighbor.s[j]+span)
                line = interp_at(neighbor.points, neighbor.s, np.arange(a,b+1e-9,.25))
                if len(line) < 2:
                    continue
                distance, nearest = tree.query(line)
                supported = ((distance-gap > self.exclusion) & (distance <= 32.)
                             & (np.abs(parent.s[nearest]-at) <= span))
                edges = np.diff(np.r_[False,supported,False].astype(np.int8))
                starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
                choices = [(a,b) for a,b in zip(starts,stops) if (b-a-1)*.25 >= max(32.,min_length)]
                if not choices:
                    continue
                begin,end = choices[int(rng.integers(len(choices)))]
                curve = line[begin:end]
                if arclength(curve)[-1] < min_length:
                    continue
                # Match against the local parent window, then leave stricter
                # monotonicity/seed observability checks to the existing tasks.
                near = nearest[begin:end]
                lo,hi = max(0,int(near.min())-2),min(len(parent.s),int(near.max())+3)
                _,_,segment,u = exact_nearest(curve,parent.points[lo:hi])
                matched = parent.s[lo+segment]+u*np.diff(parent.s[lo:hi])[segment]
                if matched[-1] < matched[0]:
                    curve,matched = curve[::-1].copy(),matched[::-1]
                if np.any(np.diff(matched) < -1e-5):
                    continue
                proposals.append((fi,curve,(float(matched.min()),float(matched.max()))))
                break
            if proposals and (not hard or len(proposals) >= 4):
                break
        if not proposals:
            return None
        if not hard:
            return proposals[0]
        kind = int(rng.integers(len(DIFFICULTY_KINDS)))
        scores = [difficulty_scores(line,self.fibers[fi])[kind] for fi,line,_ in proposals]
        return proposals[int(np.argmax(scores))]

    def provenance(self):
        return dict(kind='afv', path=self.fibers.path, manifest=self.fibers.manifest_entries())

    def spatial_records(self, fi, world, radius=0.):
        bounds = np.stack((np.min(world, axis=0)-radius, np.max(world, axis=0)+radius))
        own = self.fibers.catalog[fi][0]
        return list(self.fibers.nearby_blocks(bounds, own, records=True))

    def _target_tree(self, fi):
        if fi not in self._trees:
            p = self.fibers[fi].points
            self._trees[fi] = (cKDTree(p),float(np.linalg.norm(np.diff(p,axis=0),axis=1).max()/2))
            if len(self._trees) > 8:
                self._trees.popitem(last=False)
        self._trees.move_to_end(fi)
        return self._trees[fi]

    def clear_of_target(self, fi, world):
        tree,gap = self._target_tree(fi)
        return tree.query(np.asarray(world).reshape(-1,3))[0]-gap > self.exclusion

    def candidates(self, item, crop, rule, *, mask_crop=None, rasterize=True):
        """Exact centerline queries, with whole-annotation clearance.

        Confidence labels rasterize only cells containing these line samples;
        their entire cell must clear the target exclusion tube. Query locations
        are never rasterized, expanded, jittered or snapped to the crop grid.
        Observation-only rows use the same geometry for sampling feedback but
        do not need a dense confidence-label mask.
        """
        fi,t,reverse = item['fiber_ref']
        lateral, forward = crop.lateral_coords[[0,-1]], crop.forward_coords[[0,-1]]
        corners = np.array([[a,b,z] for a in lateral for b in lateral for z in forward])
        world = corners @ np.asarray(item['frame']).T+item['pos']
        lines = [r['samples'] for r in self.spatial_records(fi, world)]
        mask_crop = crop if mask_crop is None else mask_crop
        shape = (mask_crop.depth,mask_crop.width,mask_crop.width)
        mask = np.zeros(shape,bool) if rasterize else None
        empty = dict(foreign=mask, local=np.empty((0,3)), nearest=np.empty(0,np.int64), path_ids=np.empty(0,np.int64),
                     counts=dict(foreign_components=0))
        if not lines or len(item['identity_curve']) < 3:
            return empty
        pos, frame = np.asarray(item['pos']), np.asarray(item['frame'])
        local = np.concatenate([(p-pos) @ frame for p in lines])
        path_ids = np.repeat(np.arange(len(lines)),[len(p) for p in lines])
        indices = crop_indices(crop,local)
        near = np.all((indices >= 0) & (indices <= np.array([crop.depth,crop.width,crop.width])-1),axis=1)
        local,indices,path_ids = local[near],indices[near],path_ids[near]
        if not len(local):
            return empty
        tree, gap = self._target_tree(fi)
        keep = tree.query(local @ frame.T+pos)[0]-gap > max(self.exclusion,rule.own_radius)
        nearest = cKDTree(item['identity_curve']).query(local)[1]
        keep &= (nearest > 0) & (nearest < len(item['identity_curve'])-1)
        local,nearest,path_ids = local[keep],nearest[keep],path_ids[keep]
        if rasterize:
            voxels = np.unique(np.rint(crop_indices(mask_crop,local)).astype(int),axis=0)
            voxels = voxels[np.all((voxels >= 0) & (voxels < np.asarray(shape)),axis=1)]
            # Only these occupied cells are queried; avoid materializing the
            # full million-voxel coordinate grid for every decision.
            lateral, forward = mask_crop.lateral_coords, mask_crop.forward_coords
            points = np.column_stack((lateral[voxels[:, 2]], lateral[voxels[:, 1]],
                                      forward[voxels[:, 0]]))
            half_cell = np.sqrt(3)*mask_crop.spacing/2
            keep = tree.query(points @ frame.T+pos)[0]-gap-half_cell > max(self.exclusion,rule.own_radius)
            mask[tuple(voxels[keep].T)] = True
        return dict(foreign=mask, local=local, nearest=nearest, path_ids=path_ids,
                    counts=dict(foreign_components=int(bool(len(local)))))
