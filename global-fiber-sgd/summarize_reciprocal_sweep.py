"""Summarize full-volume parallel-off / perpendicular-iteration experiments."""
import argparse
import json
from pathlib import Path


def summarize(moving=False):
    root=Path(__file__).parent
    baseline=json.loads((root/"data/rejection_audit_v1/full.json").read_text())["runs"][0]
    rows=[]
    for iterations in (4,16,64):
        experiment="moving_projection_v1" if moving else "parallel_reciprocal_v1"
        path=root/f"data/{experiment}/full_i{iterations}.json"
        d=json.loads(path.read_text());r=d["runs"][0]
        assert d["settings"]["parallel_reciprocal"] is False
        assert d["settings"].get("moving_projection_windows",False)==moving
        assert r["first_failure_checks"].get("probe_reciprocal",0)==0
        n=r["candidates"]+r["refined_candidates"]
        failures=r["first_failure_checks"].get("central_reciprocal",0)
        evaluated=r["perpendicular_reciprocal_evaluated"]
        assert sum(r["first_failure_checks"].values())+r["events"]+r["deduplicated_candidates"]==n
        assert r["perpendicular_reciprocal_window_failures"]<=failures
        rows.append(dict(iterations=iterations,initial_candidates=r["candidates"],
            refinement_candidates=r["refined_candidates"],attempts=n,
            perpendicular_reciprocal_evaluated=evaluated,perpendicular_reciprocal_failures=failures,
            reciprocal_percent_all=100*failures/n,reciprocal_percent_evaluated=100*failures/evaluated if evaluated else 0,
            failures_at_projection_window=r["perpendicular_reciprocal_window_failures"],
            failures_at_fiber_endpoint=r["perpendicular_reciprocal_endpoint_failures"],
            accepted=r["events"],parallel=r["parallel"],perpendicular=r["orthogonal"],
            accepted_change_vs_legacy_percent=100*(r["events"]/baseline["events"]-1),
            extraction_seconds=r["seconds"],orientation_seconds=r["orientation_seconds"],
            wall_seconds=d["wall_seconds"],first_failure_checks=r["first_failure_checks"]))
    return dict(moving_projection_windows=moving,legacy=dict(accepted=baseline["events"],parallel=baseline["parallel"],perpendicular=baseline["orthogonal"]),
        rows=rows,note="Unsigned full-volume runs, no DB overwrite. Reciprocal failures now exclusively perpendicular. Denominators vary because adaptive refinement changes. Boundary counters overlap and do not prove causation individually.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--moving",action="store_true")
    args=parser.parse_args()
    result=summarize(args.moving)
    name="moving_projection_results.json" if args.moving else "reciprocal_sweep_results.json"
    (Path(__file__).parent/name).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
