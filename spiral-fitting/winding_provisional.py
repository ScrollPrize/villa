"""Reversible trial admissions over an immutable, already-audited region.

This evaluation helper does not persist or alter a growth workspace. Evidence
is append-only; quarantining geometry never erases the constraint that exposed
it. All dependent admissions are withdrawn, even if another path exists: that
path must be assessed afresh rather than automatically certifying a survivor.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from numbers import Integral
from types import MappingProxyType

from grow_winding_region import audit_accepted_topology


REQUIRED_CHECKS = {
    'track': {'anchored_theta', 'opposite_object_brackets', 'incident_complete'},
    'patch': {'both_track_arms', 'theta_transport', 'incident_complete'},
    'loop_track': {'anchored_theta', 'fiber_patch_loop', 'separated_contacts', 'incident_complete'},
    'loop_patch': {'fiber_track_loop', 'separated_contacts', 'theta_transport', 'incident_complete'},
    'fiber': {'component_cycles', 'theta_transport', 'incident_complete'},
    'growth_patch': {'component_cycles', 'theta_transport', 'incident_complete', 'ordinary_support'},
}


@dataclass(frozen=True)
class Admission:
    node: str
    offset: int
    kind: str
    parents: frozenset[str]
    checks: dict


class ProvisionalRegion:
    """In-memory trial ledger; caller supplies completed geometric evidence.

    A contradictory cycle proves incompatibility, not which endpoint is wrong.
    Both provisional endpoints' upstream trial roots and their descendants are
    therefore quarantined: a bad leaf may expose a wrong bridge assignment.
    A trusted/trusted contradiction halts the trial instead of rewriting the
    baseline. A caller must stop ordinary growth when either occurs and rebuild
    any external frontier caches from the returned surviving offsets.
    """
    def __init__(self, seed, trusted, edges, guard):
        self.seed = seed
        self.trusted = MappingProxyType(dict(trusted))
        self.edges = list(edges)
        audit_accepted_topology(seed, self.trusted, self.edges)
        self.base_guard = guard.fork()
        self.guard = guard.fork()
        self.records = {}
        self.quarantined = set()
        self.events = []
        self.halted = False
        self.joint_groups = {}

    @property
    def offsets(self):
        return {**self.trusted, **{n:r.offset for n,r in self.records.items()}}

    def observe(self, edges):
        """Reveal independently verified eligible constraints, including zero support."""
        if self.halted:
            raise RuntimeError('Trial halted by a trusted-region contradiction')
        self.edges.extend(edges)
        active = self.offsets
        conflicts = []
        roots = set()
        trusted_conflict = False
        for eid, edge in enumerate(self.edges):
            if edge.a not in active or edge.b not in active:
                continue
            residual = active[edge.b]-active[edge.a]-edge.delta
            if residual:
                conflicts.append(dict(edge=eid, a=edge.a, b=edge.b, residual=int(residual)))
                roots.update({edge.a,edge.b} & self.records.keys())
                trusted_conflict |= edge.a in self.trusted and edge.b in self.trusted
        if trusted_conflict:
            self.halted = True
            self.events.append(dict(action='halt',reason='trusted_theta_conflict',conflicts=conflicts))
            raise RuntimeError('New evidence contradicts the trusted region; trial halted')
        if roots:
            ancestors=set(roots)
            while True:
                providers={p for n in ancestors for p in self.records[n].parents if p in self.records}
                if providers <= ancestors: break
                ancestors.update(providers)
            roots={n for n in ancestors if not (self.records[n].parents & self.records.keys())}
            return self.invalidate(roots, dict(reason='late_theta_conflict',conflicts=conflicts,
                                              scope='all_implicated_provisional_ancestry'))
        return []

    def admit(self, node, offset, kind, parents, checks, edges=(), *, _joint=False):
        """Stage a single addition; failures preserve the current geometry index.

        Checks are named audit results, not scores to average. This function
        manages their provenance and rollback; it cannot establish their truth.
        """
        if self.halted:
            raise RuntimeError('Trial halted by a trusted-region contradiction')
        if node in self.offsets or node in self.quarantined:
            raise ValueError('Node already active or quarantined; use a fresh reviewed trial')
        if kind not in REQUIRED_CHECKS:
            raise ValueError('Unknown admission kind')
        if kind in ('loop_track','loop_patch') and not _joint:
            raise ValueError('Fiber/track/patch loops require atomic admission')
        if isinstance(offset,bool) or not isinstance(offset,Integral):
            raise ValueError('Theta offsets must be exact integers')
        parents = frozenset(parents)
        # Retain new eligible evidence even if the geometric proposal fails.
        withdrawn = self.observe(edges)
        if withdrawn:
            return dict(accepted=False,reason='late_conflict_requires_restart',withdrawn=withdrawn)
        missing = sorted(REQUIRED_CHECKS[kind]-checks.keys())
        failed = sorted(k for k,v in checks.items() if v is not True)
        if missing or failed:
            return dict(accepted=False,reason='incomplete_or_failed_checks',missing=missing,failed=failed)
        if not parents or not parents <= self.offsets.keys():
            return dict(accepted=False,reason='missing_live_dependencies')
        active = self.offsets
        # Every provisional provider whose edge supports this admission is a
        # dependency, not just a handpicked spanning-tree parent.
        providers = {e.b if e.a == node else e.a for e in self.edges
                     if (e.a == node and e.b in active) or (e.b == node and e.a in active)}
        if not providers <= parents:
            return dict(accepted=False,reason='unrecorded_evidence_dependencies',providers=sorted(providers-parents))
        active[node] = int(offset)
        try:
            audit_accepted_topology(self.seed,active,self.edges)
        except RuntimeError as error:
            return dict(accepted=False,reason='theta_or_connectivity_conflict',detail=str(error))
        staged = self.guard.fork()
        if kind in ('patch','loop_patch','growth_patch') and node not in staged.patch_nodes:
            return dict(accepted=False,reason='missing_patch_geometry')
        conflict = staged.check(node,int(offset))
        if conflict:
            return dict(accepted=False,reason='geometry_conflict',conflict=conflict)
        staged.accept(node,int(offset))
        self.records[node] = Admission(node,int(offset),kind,parents,dict(checks))
        self.guard = staged
        self.events.append(dict(action='admit',node=node,offset=int(offset),kind=kind,
                                parents=sorted(parents),checks=dict(checks)))
        return dict(accepted=True,reason='provisional')

    def admit_fiber_track_patch(self, track, track_offset, patch, patch_offset,
                               fiber, checks, edges):
        """Atomically stage the mutually supported track and intact patch.

        The accepted fiber is the external anchor. The loop never promotes
        its own participants to trusted objects. Either participant's later
        invalidation removes the entire joint hypothesis and its descendants.
        """
        required={'fiber_track_contact','fiber_patch_contact','track_patch_contact',
                  'separated_contacts','theta_transport','incident_complete'}
        if any(isinstance(value,bool) or not isinstance(value,Integral)
               for value in (track_offset,patch_offset)):
            raise ValueError('Theta offsets must be exact integers')
        if required-checks.keys() or any(v is not True for v in checks.values()):
            return dict(accepted=False,reason='incomplete_or_failed_loop_checks')
        if fiber not in self.trusted or len({track,patch,fiber})!=3:
            return dict(accepted=False,reason='missing_trusted_fiber_anchor')
        reuse_track=track in self.records
        if reuse_track and (self.records[track].kind not in ('track','loop_track')
                            or self.records[track].offset!=track_offset):
            return dict(accepted=False,reason='existing_track_theta_or_kind_conflict')
        edges=list(edges)
        pairs={frozenset((e.a,e.b)) for e in self.edges+edges}
        if any(frozenset(pair) not in pairs for pair in ((fiber,track),(fiber,patch),(track,patch))):
            return dict(accepted=False,reason='missing_loop_constraint')
        withdrawn=self.observe(edges)
        if withdrawn:return dict(accepted=False,reason='late_conflict_requires_restart',withdrawn=withdrawn)
        trial=copy.copy(self)
        trial.records=self.records.copy();trial.events=self.events.copy()
        trial.edges=self.edges.copy();trial.guard=self.guard.fork()
        trial.quarantined=self.quarantined.copy();trial.joint_groups=self.joint_groups.copy()
        for node,offset,kind in ((track,track_offset,'loop_track'),(patch,patch_offset,'loop_patch')):
            if node==track and reuse_track:
                continue
            active=trial.offsets
            parents={e.b if e.a==node else e.a for e in trial.edges
                     if (e.a==node and e.b in active) or (e.b==node and e.a in active)}
            result=trial.admit(node,offset,kind,parents,{k:True for k in REQUIRED_CHECKS[kind]},_joint=True)
            if not result['accepted']:return result
        trial.joint_groups[track]=trial.joint_groups.get(track,frozenset((track,)))|{patch}
        self.records=trial.records;self.events=trial.events;self.edges=trial.edges
        self.guard=trial.guard;self.joint_groups=trial.joint_groups
        return dict(accepted=True,reason='joint_provisional_loop',nodes=[track,patch],reused_track=reuse_track)

    def invalidate(self, roots, witness):
        """Withdraw root proposals and all downstream validation dependencies."""
        roots = set(roots)
        if not roots or not roots <= self.records.keys():
            raise ValueError('Invalidation roots must be active provisional nodes')
        removed = set(roots)
        while True:
            peers={n for group in self.joint_groups.values() if group & removed for n in group}
            descendants = {n for n,r in self.records.items() if r.parents & removed}|peers
            if descendants <= removed: break
            removed.update(descendants)
        surviving = {n:r for n,r in self.records.items() if n not in removed}
        # Rebuild instead of leaving stale surfaces/bins in the monotone guard.
        staged = self.base_guard.fork()
        staged.patch_nodes = self.guard.patch_nodes.copy()
        for node, record in surviving.items():
            conflict = staged.check(node,record.offset)
            if conflict:
                raise RuntimeError(f'Unchanged surviving geometry failed replay: {conflict}')
            staged.accept(node,record.offset)
        active = {**self.trusted, **{n:r.offset for n,r in surviving.items()}}
        audit_accepted_topology(self.seed,active,self.edges)
        self.records = surviving
        self.joint_groups={k:g for k,g in self.joint_groups.items() if not g & removed}
        self.guard = staged
        self.quarantined.update(removed)
        self.events.append(dict(action='quarantine',roots=sorted(roots),removed=sorted(removed),witness=witness))
        return sorted(removed)
