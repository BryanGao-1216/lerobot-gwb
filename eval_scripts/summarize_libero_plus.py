#!/usr/bin/env python
"""Aggregate official LIBERO-plus results by suite, perturbation category and difficulty."""
import argparse
import contextlib
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

SUITES = {'libero_spatial': 2402, 'libero_object': 2518, 'libero_goal': 2591, 'libero_10': 2519}


def summarize(records, fields):
    acc = defaultdict(lambda: [0, 0])
    for r in records:
        key = tuple(r[f] for f in fields)
        acc[key][0] += int(r['success'])
        acc[key][1] += 1
    ans = []
    for key, (good, n) in sorted(acc.items(), key=lambda kv: str(kv[0])):
        ans.append({**dict(zip(fields, key)), 'successes': good, 'episodes': n,
                    'success_rate_pct': round(100 * good / n, 3)})
    return ans


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--classification', required=True)
    args = p.parse_args()
    root = Path(args.run_dir)
    classes = json.loads(Path(args.classification).read_text())

    # Resolve task indices against the installed fork; classification file is 1-indexed.
    from wand.api import library as _magickwand  # noqa: F401
    from libero.libero import benchmark
    records = []
    for suite, expected in SUITES.items():
        info_path = root / suite / 'eval_info.json'
        if not info_path.exists():
            raise SystemExit(f'Missing {info_path}. All four suites are required for a full benchmark.')
        info = json.loads(info_path.read_text())
        task_results = [x for x in info.get('per_task', []) if x.get('task_group') == suite]
        if len(task_results) != expected:
            raise SystemExit(f'{suite}: got {len(task_results)} tasks, expected {expected}')
        with contextlib.redirect_stdout(io.StringIO()):
            b = benchmark.get_benchmark_dict()[suite]()
        assert b.get_num_tasks() == expected, (suite, b.get_num_tasks())
        lookup = {r['name']: r for r in classes[suite]}
        if len(lookup) != expected:
            raise SystemExit(f'{suite}: classification has duplicate or missing names')
        seen = set()
        for x in task_results:
            tid = int(x['task_id'])
            if tid in seen or not 0 <= tid < expected:
                raise SystemExit(f'{suite}: repeated/out-of-range task index {tid}')
            seen.add(tid)
            name = b.get_task(tid).name
            if name not in lookup:
                raise SystemExit(f'{suite}: no classification for task_id={tid}, name={name!r}')
            meta = lookup[name]
            outcomes = x['metrics']['successes']
            if len(outcomes) != 1:
                raise SystemExit(f'{suite}:{tid}: expected exactly 1 episode, got {len(outcomes)}')
            records.append(dict(suite=suite, task_id=tid, task_name=name,
                                category=meta['category'], difficulty_level=int(meta['difficulty_level']),
                                success=int(bool(outcomes[0]))))
    assert len(records) == 10030

    summary = {
        'overall': summarize(records, []),
        'by_suite': summarize(records, ['suite']),
        'by_category': summarize(records, ['category']),
        'by_difficulty': summarize(records, ['difficulty_level']),
        'by_suite_category': summarize(records, ['suite', 'category']),
        'by_category_difficulty': summarize(records, ['category', 'difficulty_level']),
    }
    for label, data in summary.items():
        write_csv(root / f'{label}.csv', data)
    write_csv(root / 'per_variant_results.csv', records)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))

    print('\n=== LIBERO-plus full benchmark ===')
    for label in ('overall', 'by_suite', 'by_category', 'by_difficulty'):
        print('\n' + label)
        for item in summary[label]:
            print(item)
    print(f'\nSaved all tables in {root}')


if __name__ == '__main__':
    main()
