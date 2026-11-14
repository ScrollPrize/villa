"""Report event and distinct-neighbor degrees from a saved constraint graph."""
import argparse
import json
from pathlib import Path
import numpy as np
from .fiber_graph_store import FiberGraph


def summarize(path):
    path=Path(path)
    graph=FiberGraph(path);n=len(graph.ids);e=len(graph.event_ids)
    pair_rows=np.unique(np.sort(np.column_stack((graph.i,graph.j)),axis=1),axis=0)
    pairs=len(pair_rows)
    degrees=np.sort(np.bincount(pair_rows.ravel(),minlength=n))
    connected=int(np.count_nonzero(degrees))
    distance=[]
    for kind in ("parallel","orthogonal"):
        values=graph.distance[graph.parallel if kind=='parallel' else ~graph.parallel]
        # SQLite ROUND uses half-away-from-zero, unlike NumPy's ties-to-even.
        buckets=np.copysign(np.floor(abs(values)*2+.5),values).astype(np.int64)
        for bucket in np.unique(buckets):
            selected=values[buckets==bucket]
            distance.append(dict(kind=kind,nearest_half_winding=float(bucket/2),events=len(selected),mean_distance=float(selected.mean())))
    stats=dict(fibers=n,constraints=e,isolated_fibers=n-connected,
               parallel=int(graph.parallel.sum()),orthogonal=int((~graph.parallel).sum()))
    return dict(database=str(path.resolve()),database_bytes=path.stat().st_size,fibers=n,events=e,
        event_statistics=stats,unique_fiber_pairs=pairs,
        neighbor_degree=dict(mean=2*pairs/n if n else 0,median=float(np.median(degrees)) if n else 0,
                             p95=int(degrees[int(.95*(n-1))]) if n else 0,maximum=int(degrees[-1]) if n else 0),
        fibers_with_constraints=connected,coverage_percent=100*connected/n if n else 0,
        parallel_percent=100*stats["parallel"]/e if e else 0,orthogonal_percent=100*stats["orthogonal"]/e if e else 0,
        winding_distance_bins=distance,
        note="Degree counts both endpoints; local events repeat between pairs. Bins are nearest half-winding measurements, not ground-truth winding labels. database_bytes is only the root container; referenced stores are not counted.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("database",type=Path);p.add_argument("--output",type=Path)
    a=p.parse_args();text=json.dumps(summarize(a.database),indent=2)+"\n"
    if a.output:a.output.write_text(text)
    else:print(text,end="")
