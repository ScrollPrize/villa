"""Read-only integrity, measurement round-trip and AFV geometry checks."""
import argparse
import json
import math
from pathlib import Path
import re
import sqlite3
import struct


def validate(path, afv, all_measurements=False):
    db = sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)
    source = sqlite3.connect(afv.resolve().as_uri()+"?mode=ro", uri=True)
    assert db.execute("pragma integrity_check").fetchall() == [("ok",)]
    assert not db.execute("pragma foreign_key_check").fetchall()
    meta = {k: json.loads(v) for k,v in db.execute("select key,value from metadata")}
    assert meta["complete"] and meta["format"] == "fiber-constraints"
    count = db.execute("select count(*) from constraints").fetchone()[0]
    assert count == meta["statistics"]["constraints"]
    assert db.execute("select count(*) from fibers").fetchone()[0] == meta["statistics"]["fibers"]
    source_ids = {r[0] for r in source.execute("select id from fibers")}
    ids = {r[0] for r in db.execute("select id from fibers")}
    assert ids <= source_ids
    if meta["provenance"]["config"].get("full_volume"):
        assert ids == source_ids
    header = (Path(__file__).resolve().parents[1] / "volume-cartographer/core/include/vc/fiber_tracer/FiberTraceConstraints.hpp").read_text()
    struct_body = header.split("struct FiberTraceConstraint {",1)[1].split("\n};",1)[0]
    # Every declared field must be serialized, including optional/null fields.
    expected = set(re.findall(r"(?:std::\w+(?:<[^;]+?>)?|double|bool|int|cv::Vec3d)\s+(\w+)\s*(?:=|\{|;)", struct_body))
    expected |= {"supportAEnd", "supportBEnd"}
    step = 1 if all_measurements else max(1,count//1000)
    rows = db.execute("select id,fiber_a,fiber_b,arc_a,arc_b,ax,ay,az,bx,by,bz,parallel_score,perpendicular_score,winding_distance,parallel_winding_distance,evidence_weight,normal_a,normal_b,measurement from constraints where id % ?=0", (step,))
    checked=0
    for row in rows:
        m = json.loads(row[-1]); assert set(m) == expected, (set(m)^expected)
        assert m["localEvent"] and not m["hardContinuity"] and m["signedWindingDelta"] is None
        assert [m[k] for k in ("arcABaseVoxels","arcBBaseVoxels")] == list(row[3:5])
        assert m["pointABaseXYZ"] == list(row[5:8]) and m["pointBBaseXYZ"] == list(row[8:11])
        assert [m[k] for k in ("parallelScore","perpendicularScore","windingDistance","parallelWindingDistance","evidenceWeight")] == list(row[11:16])
        for side in range(2):
            n=json.loads(row[16+side])
            if n["valid"]:
                assert all(math.isfinite(x) for x in n["xyz"])
                assert abs(math.sqrt(sum(x*x for x in n["xyz"]))-1)<1e-5
            else:
                assert n["xyz"] is None
            # Check a bounded deterministic sample against original unsimplified geometry.
            if checked < 100:
                points=[]
                for (blob,) in source.execute("select points from blocks where fiber_id=? order by first_segment",(row[1+side],)):
                    block=list(struct.iter_unpack("<ddd",blob));points.extend(block[1:] if points else block)
                target=row[3+side];remaining=target; actual=None
                for a,b in zip(points,points[1:]):
                    length=math.dist(a,b)
                    if remaining<=length+1e-7:
                        f=min(1,max(0,remaining/length)) if length else 0
                        actual=[x+f*(y-x) for x,y in zip(a,b)];break
                    remaining-=length
                assert actual is not None and math.dist(actual,row[5+3*side:8+3*side])<1e-5
        checked+=1
    degrees=dict(db.execute("select f.id,coalesce(a.n,0)+coalesce(b.n,0) from fibers f left join (select fiber_a id,count(*) n from constraints group by fiber_a) a on a.id=f.id left join (select fiber_b id,count(*) n from constraints group by fiber_b) b on b.id=f.id"))
    assert sum(degrees.values())==2*count
    assert sum(v==0 for v in degrees.values())==meta["statistics"]["isolated_fibers"]
    result={"integrity":"ok","fibers":len(ids),"constraints":count,"measurement_rows_checked":checked,"geometry_endpoints_checked":2*min(checked,100),"degree_sum":sum(degrees.values())}
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("database",type=Path)
    p.add_argument("--afv",type=Path,default=Path(__file__).parent/"data/Paris4_v1.afv")
    p.add_argument("--all-measurements",action="store_true")
    a=p.parse_args();validate(a.database,a.afv,a.all_measurements)
