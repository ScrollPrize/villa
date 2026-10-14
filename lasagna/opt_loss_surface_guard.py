"""Opt-in orientation/area guard relative to healthy post-repair quads."""
import torch


def triangle_vectors(x):
    p, q, r, s = x[:-1, :-1], x[:-1, 1:], x[1:, :-1], x[1:, 1:]
    return torch.linalg.cross(q-p, r-p), torch.linalg.cross(r-s, q-s)


def corner_vectors(x):
    p, q, r, s = x[:-1, :-1], x[:-1, 1:], x[1:, :-1], x[1:, 1:]
    return [torch.linalg.cross(u, v) for u, v in
            [(q-p,r-p), (q-p,s-q), (s-r,r-p), (s-r,s-q)]]


class SurfaceGuard:
    def __init__(self, reference, weight=100., step=100.):
        corners = corner_vectors(reference.detach())
        mean = sum(corners)
        self.valid = torch.stack([((t*mean).sum(-1)>0) & (t.norm(dim=-1)>=step**2*1e-4) for t in corners]).all(0)
        a,b = triangle_vectors(reference.detach())
        self.valid &= (a*b).sum(-1)>0
        horizontal = (mean[:,1:]*mean[:,:-1]).sum(-1)<0
        vertical = (mean[1:]*mean[:-1]).sum(-1)<0
        self.valid[:,1:] &= ~horizontal; self.valid[:,:-1] &= ~horizontal
        self.valid[1:] &= ~vertical; self.valid[:-1] &= ~vertical
        self.normals = [torch.nn.functional.normalize(t, dim=-1) for t in corners]
        self.areas = [t.norm(dim=-1).clamp_min(1e-4) for t in corners]
        self.weight = weight

    def __call__(self, xyz):
        vectors = corner_vectors(xyz[0])
        if not bool(self.valid.any()):
            return xyz.sum()*0
        # All four corners are encouraged to retain 10% of reference signed area.
        return self.weight*sum(torch.relu(.1-(t*n).sum(-1)/area)[self.valid].square().mean()
                               for t, n, area in zip(vectors, self.normals, self.areas))
