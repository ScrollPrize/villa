"""Report event and distinct-neighbor degrees from a saved constraint graph."""
import argparse
import json
from pathlib import Path
import sqlite3
import statistics


def summarize(path):
    db=sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True)
    meta={k:json.loads(v) for k,v in db.execute("select key,value from metadata")}
    assert meta["complete"]
    n=db.execute("select count(*) from fibers").fetchone()[0]
    e=db.execute("select count(*) from constraints").fetchone()[0]
    pairs=db.execute("select count(*) from (select fiber_a,fiber_b from constraints group by fiber_a,fiber_b)").fetchone()[0]
    neighbors={r[0]:r[1] for r in db.execute("select id,count(*) from (select fiber_a id,fiber_b neighbor from constraints union select fiber_b id,fiber_a neighbor from constraints) group by id")}
    degrees=sorted(neighbors.get(r[0],0) for r in db.execute("select id from fibers"))
    distance=[]
    for kind in ("parallel","orthogonal"):
        op=">" if kind=="parallel" else "<="
        for bucket,count,mean in db.execute(f"select cast(round(winding_distance*2) as integer),count(*),avg(winding_distance) from constraints where parallel_score {op} perpendicular_score group by 1 order by 1"):
            distance.append(dict(kind=kind,nearest_half_winding=bucket/2,events=count,mean_distance=mean))
    stats=meta["statistics"]
    return dict(database=str(path.resolve()),database_bytes=path.stat().st_size,fibers=n,events=e,
        event_statistics=stats,unique_fiber_pairs=pairs,
        neighbor_degree=dict(mean=2*pairs/n,median=statistics.median(degrees),p95=degrees[int(.95*(n-1))],maximum=max(degrees)),
        fibers_with_constraints=len(neighbors),coverage_percent=100*len(neighbors)/n,
        parallel_percent=100*stats["parallel"]/e,orthogonal_percent=100*stats["orthogonal"]/e,
        winding_distance_bins=distance,
        note="Degree counts both endpoints; local events repeat between pairs. Bins are nearest half-winding measurements, not ground-truth winding labels.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("database",type=Path);p.add_argument("--output",type=Path)
    a=p.parse_args();text=json.dumps(summarize(a.database),indent=2)+"\n"
    if a.output:a.output.write_text(text)
    else:print(text,end="")
