"""Validate completed fiber comparison runs and summarize the paired effects."""
import argparse
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    runs = json.loads((args.directory / 'runs.json').read_text())
    assert len(runs) == 8, 'Expected two repetitions of all four configurations'
    groups, identities, sequences = {}, set(), []
    for run in runs:
        folder = args.directory / run['name']
        report = json.loads((folder / 'summary.json').read_text())
        rows = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
        measured = [row for row in rows if row['measured']]
        assert len(rows) == 520 and len(measured) == 500
        assert sum(row['observed_states'] for row in measured) == 16000
        assert all(math.isfinite(row['loss']) and math.isfinite(row['error_mean']) for row in measured)
        assert report['remote_prefetch_lookahead'] == 0 and report['batch'] == 32 and report['grad_steps'] == 1
        assert report['cuda_augmentation'] == run['augmentation'] and report['cuda_cache'] == run['cache']
        assert report['heading'] == 'tensor'
        identities.add(report['native_build']['sha256'])
        sequences.append([row['dataset_counts'] for row in rows])
        readers = report['reader_stats']
        cache = readers['chunk_cache']
        assert cache['layout'] == run['cache']
        assert readers['augmentation_batch_calls_including_warmup'] == (520 if run['augmentation'] == 'batch' else 16640)
        assert len(cache['readers']) == (4 if run['cache'] == 'shared' else 8)
        assert sum(r['cache_budget'] for r in cache['readers'].values()) == 4 * (768 << 20)
        assert all(r['cpu_routes'] == 0 for r in [*cache['readers'].values(), *cache['startup_readers'].values()])
        assert all(len(c) == (2 if run['cache'] == 'shared' else 1) for c in cache['consumers'].values())
        run['uploads_gib_including_warmup_prefetch'] = sum(r['uploaded_bytes'] for r in cache['readers'].values()) / 2**30
        run['augmentation_seconds_including_warmup'] = readers['stage_seconds_including_warmup']['augmentation']
        run['sampling_seconds_including_warmup'] = readers['stage_seconds_including_warmup']['read_sample']
        run['unique_positions'] = sum(c['unique_positions'] for c in report['coverage'].values())
        run['source_chunk_counts'] = {k: c['unique_crop_footprint_chunks'] for k, c in report['coverage'].items()}
        groups.setdefault((run['augmentation'], run['cache']), []).append(run)
    assert len(identities) == 1, 'Native binary changed during the experiment'
    assert all(s == sequences[0] for s in sequences), 'Source draw sequences differed'
    summaries = {}
    for (augmentation, cache), values in groups.items():
        assert {v['repetition'] for v in values} == {1, 2}
        summaries[f'{augmentation}_{cache}'] = {
            key: dict(mean=statistics.mean(v[key] for v in values),
                      min=min(v[key] for v in values), max=max(v[key] for v in values))
            for key in ('wall_seconds', 'loader_seconds', 'preparation_seconds', 'update_seconds',
                        'uploads_gib_including_warmup_prefetch', 'augmentation_seconds_including_warmup',
                        'sampling_seconds_including_warmup')}
    effects = {}
    for cache in ('separate', 'shared'):
        effects[f'batching_with_{cache}_caches_seconds_saved'] = (
            summaries[f'sample_{cache}']['wall_seconds']['mean'] - summaries[f'batch_{cache}']['wall_seconds']['mean'])
    for augmentation in ('sample', 'batch'):
        effects[f'cache_sharing_with_{augmentation}_augmentation_seconds_saved'] = (
            summaries[f'{augmentation}_separate']['wall_seconds']['mean'] - summaries[f'{augmentation}_shared']['wall_seconds']['mean'])
    result = dict(runs=runs, configurations=summaries, effects=effects, native_sha256=next(iter(identities)),
                  same_source_draw_sequence=True,
                  limitation='Two runs per configuration, reverse order in round two. Live feedback changes individual crops. No statistical significance claim.')
    (args.directory / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(configurations=summaries, effects=effects), indent=2))


if __name__ == '__main__':
    main()
