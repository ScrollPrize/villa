"""Validate the audit ledger and calculate candidate-attempt percentages."""
import argparse
import json
from pathlib import Path


def summarize(path):
    data=json.loads(path.read_text())
    # New extraction modes intentionally differ from the historical database.
    # A comparison, when requested by the native runner, must still have passed.
    if "all_measurements_match_database" in data:
        assert data["all_measurements_match_database"]
    run=data["runs"][0]
    n=run["candidates"]+run["refined_candidates"]
    failures=run["first_failure_checks"]
    assert sum(failures.values())+run["deduplicated_candidates"]+run["events"]==n
    nonconsistency=("invalid_central_winding","invalid_all_windings","winding_cutoff")
    if data.get("settings",{}).get("certify_winding") is False:
        nonconsistency+=("winding_invalid_probe",)
    assert sum(v for k,v in failures.items() if k not in nonconsistency)==run["rejected_consistency"]
    rows=[dict(check=k,count=v,percent_all_attempts=100*v/n) for k,v in failures.items()]
    overlaps=[dict(check=k,count=v,percent_all_attempts=100*v/n,
                   percent_reaching_winding_check=100*v/run["winding_checks_evaluated"])
              for k,v in run["winding_overlap_checks"].items()]
    result=dict(attempts=n,initial_candidates=run["candidates"],refinement_candidates=run["refined_candidates"],
        first_failure_checks=rows,overlapping_winding_checks=overlaps,
        winding_checks_evaluated=run["winding_checks_evaluated"],
        accepted=dict(count=run["events"],percent=100*run["events"]/n),
        deduplicated=dict(count=run["deduplicated_candidates"],percent=100*run["deduplicated_candidates"]/n),
        all_saved_measurements_unchanged=data.get("all_measurements_match_database"),
        note="First-failure geometry attribution, not independent filter ablation. Denominator is candidate attempts, not unique fiber pairs. Winding overlaps count evaluable candidates once per predicate; percentages do not add. Signed checks are not applied.")
    if "outcomes_by_initial_relation" in run:
        typed=run["outcomes_by_initial_relation"]
        assert sum(sum(v.values()) for v in typed.values())==n
        for outcome,count in dict(failures,accepted=run["events"],deduplicated=run["deduplicated_candidates"]).items():
            assert sum(v.get(outcome,0) for v in typed.values())==count
        assert typed.get("unclassified",{}).keys() <= {"seed_tangent_support","seed_orientation"}
        assert typed["parallel"]["accepted"]==run["parallel"]
        assert typed["perpendicular"]["accepted"]==run["orthogonal"]
        result["by_initial_relation"]={}
        for kind,outcomes in typed.items():
            total=sum(outcomes.values())
            rejected=total-outcomes.get("accepted",0)-outcomes.get("deduplicated",0)
            result["by_initial_relation"][kind]=dict(attempts=total,
                percent_all_attempts=100*total/n,
                rejected=rejected,rejected_percent=100*rejected/total,
                outcomes=[dict(outcome=k,count=v,percent_within_relation=100*v/total,
                    percent_all_attempts=100*v/n) for k,v in outcomes.items()])
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("input",type=Path);p.add_argument("--output",type=Path)
    a=p.parse_args();text=json.dumps(summarize(a.input),indent=2)+"\n"
    if a.output:a.output.write_text(text)
    else:print(text,end="")
