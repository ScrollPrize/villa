"""Read-only schema2 validation against the original schema1 extraction."""
import argparse
import json
from pathlib import Path
import sqlite3
import time


def validate(original, oriented):
    started = time.monotonic()
    db = sqlite3.connect(oriented.resolve().as_uri() + '?mode=ro', uri=True)
    db.execute('ATTACH DATABASE ? AS original', (original.resolve().as_uri() + '?mode=ro',))
    assert db.execute('PRAGMA integrity_check').fetchall() == [('ok',)]
    assert not db.execute('PRAGMA foreign_key_check').fetchall()
    assert db.execute('PRAGMA user_version').fetchone() == (2,)
    meta = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
    assert meta['schema_version'] == 2 and meta['complete']
    assert not db.execute('SELECT count(*) FROM fibers WHERE normal_sign!=1').fetchone()[0]
    # Every original fiber and every event field except endpoint normals is exact.
    for table, excluded in [('fibers', set()), ('constraints', {'normal_a', 'normal_b'})]:
        cols = [r[1] for r in db.execute(f'PRAGMA original.table_info({table})') if r[1] not in excluded]
        mismatch = ' OR '.join(f'a."{c}" IS NOT b."{c}"' for c in cols)
        assert db.execute(f'SELECT count(*) FROM {table}').fetchone() == db.execute(f'SELECT count(*) FROM original.{table}').fetchone()
        assert db.execute(f'SELECT count(*) FROM {table} a LEFT JOIN original.{table} b ON a.id=b.id WHERE {mismatch}').fetchone()[0] == 0
    print('All original fibers and non-normal constraint fields match exactly.', flush=True)
    flips = 0
    valid_count = 0
    invalid_count = 0
    for side in ('a', 'b'):
        col = f'normal_{side}'
        expressions = []
        for prefix in ('a', 'b'):
            expressions.extend(f"json_extract({prefix}.{col},'$.xyz[{d}]') AS {prefix}{d}" for d in range(3))
            expressions.extend(f"json_extract({prefix}.{col},'$.{field}') AS {prefix}_{field}" for field in ('valid', 'reason'))
        sql = f'SELECT {",".join(expressions)} FROM constraints a JOIN original.constraints b ON a.id=b.id'
        # SQLite parses JSON in native code; no Python per-event deserialization.
        diff_plus = '+'.join(f'abs(a{d}-b{d})' for d in range(3))
        diff_minus = '+'.join(f'abs(a{d}+b{d})' for d in range(3))
        bad = f'a_valid IS NOT b_valid OR a_reason IS NOT b_reason OR (a_valid AND min({diff_plus},{diff_minus})>1e-12) OR (NOT a_valid AND (a0 IS NOT NULL OR a1 IS NOT NULL OR a2 IS NOT NULL))'
        result = db.execute(f'SELECT sum({bad}), sum(a_valid), sum(NOT a_valid), sum(a_valid AND ({diff_plus})>1e-12) FROM ({sql})').fetchone()
        assert result[0] == 0, (side, result)
        valid_count += result[1]; invalid_count += result[2]; flips += result[3]
        print(f'Endpoint {side}: {result[3]:,} sign flips; all axes preserved.', flush=True)
    assert flips == meta['normal_orientation']['flipped_endpoints']
    assert invalid_count == meta['normal_orientation']['invalid_endpoints']
    result = dict(integrity='ok', exact_constraint_fields=True, valid_endpoints=valid_count,
                  invalid_endpoints=invalid_count, flipped_endpoints=flips,
                  seconds=time.monotonic()-started)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('oriented', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate(args.original, args.oriented)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
