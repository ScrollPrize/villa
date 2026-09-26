"""Contiguous acceptance ledger and conservative stop/retract policy."""
from dataclasses import dataclass, asdict
import numpy as np
from ..events import truncate_path
from ..geometry import arclength


@dataclass(frozen=True)
class JudgePolicyConfig:
    accept: float = .9
    alarm: float = .5
    delay: float = 8.
    provisional: float = 32.

    def __post_init__(self):
        if not 0 <= self.alarm < self.accept <= 1 or self.delay < 0 or self.provisional <= 0:
            raise ValueError('Invalid judge thresholds or tail settings')

    def to_dict(self):
        return asdict(self)


class JudgePolicy:
    def __init__(self, cfg=None, path_step=4.):
        self.cfg = cfg or JudgePolicyConfig()
        self.accepted = 0.
        self.ledger = []
        self.alarm = None
        self.unresolved = []
        self.audit = []
        self.reason = None
        self.path_step = path_step
        self.sample_arcs = {0.}
        self.evidence_gaps = []

    def covers(self, arc):
        end = 0.
        for interval in self.ledger:
            if interval['start'] > end+1e-7:
                break
            end = max(end, interval['end'])
        return arc <= end+1e-7 and not any(a < arc and b > a for a,b in self.evidence_gaps)

    def anchor(self, records):
        return next((r['arc'] for r in records if r['query'] and r['regular'] and r['arc'] <= self.accepted+1e-7
                     and self.covers(r['arc'])), None)

    def decide(self, records, scores, final=False):
        previous_accepted = self.accepted
        endpoint = records[-1]['arc']
        self.sample_arcs.update(r['arc'] for r in records if r['regular'])
        anchor = self.anchor(records)
        queries = [i for i, r in enumerate(records) if r['query'] and anchor is not None and r['arc'] >= anchor]
        seed = next((r for r in records if r['arc'] == 0), None)
        ready = (seed is not None and seed['support'] and anchor is not None and len(queries) >= 4
                 and records[queries[-1]]['arc']-records[queries[0]]['arc'] >= 12-1e-7)
        # Seed-only reference entries do not bridge the gap to the overlap anchor.
        boundary = self.accepted
        running = 1.
        gap = False
        eligible = []
        alarm_boundary = None
        previous_regular = None
        for i in queries:
            r = records[i]
            if r['regular']:
                if previous_regular is not None and r['arc']-previous_regular > self.path_step+1e-7:
                    gap = True
                    self.evidence_gaps.append((previous_regular, r['arc']))
                previous_regular = r['arc']
            gap = gap or not r['support'] or not np.isfinite(scores[i])
            eligible.append(bool(ready and not gap))
            if not ready or gap:
                continue
            running = min(running, float(scores[i]))
            if running <= self.cfg.alarm:
                preceding = sorted(s for s in self.sample_arcs if s < r['arc'] and s <= self.accepted)
                alarm_boundary = preceding[-1] if preceding else 0.
                self.alarm = dict(call=len(self.audit), interval=(alarm_boundary, r['arc']))
                self.reason = 'judge_alarm'
                break
            if running >= self.cfg.accept and (final or r['arc'] <= endpoint-self.cfg.delay+1e-7) and (final or r['regular']):
                boundary = max(boundary, r['arc'])
        if alarm_boundary is not None:
            self.accepted = alarm_boundary
            self.ledger = [dict(e, end=min(e['end'], self.accepted)) for e in self.ledger if e['start'] < self.accepted]
        elif self.alarm is None and boundary > self.accepted:
            self.ledger.append(dict(start=self.accepted, end=boundary, call=len(self.audit)))
            self.accepted = boundary
        self.unresolved = [(self.accepted, endpoint)] if endpoint > self.accepted else []
        if self.alarm is None and (anchor is None or endpoint-self.accepted > self.cfg.provisional+1e-7):
            self.reason = 'judge_unverified'
        self.audit.append(dict(call=len(self.audit), endpoint=endpoint, anchor=anchor, eligible=eligible,
                               arcs=[r['arc'] for r in records], scores=list(map(float, scores)),
                               support=[r['support'] for r in records], accepted=self.accepted,
                               previous_accepted=previous_accepted,
                               final=bool(final), reason=self.reason))
        return self.reason

    def export(self, path, reason):
        length = arclength(path)[-1]
        if self.accepted < length-1e-7 and self.reason is None:
            self.reason = 'judge_unverified'
        return truncate_path(path, self.accepted), self.reason or reason
