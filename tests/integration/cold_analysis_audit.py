"""Opt-in before/after audit with NO application analysis cache.

Alternate execution order on each capture; time analysis only. Compare every
raw frame field, candidate, RT decision and projected sample, not just counts.
The optional synthetic mode varies sensor pose/FOV/noise and includes absence.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
import cv2
import numpy as np
import scipy
import nsl_color_calibration as nc
from cold_analysis_reference import assert_exact, legacy_reduction
from fuse_color_pointcloud import capture_pairs, load_rt, write_binary_ply
from synthetic_board_audit import render


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')


def captures(args):
    pairs = capture_pairs(args.capture_root)
    assert pairs, 'No capture pairs'
    if args.samples:
        indices = set(np.linspace(0, len(pairs)-1, min(args.samples, len(pairs))).astype(int))
        requested = set(args.include_stamp)
        assert requested <= {p['timestamp'] for p in pairs}, 'Unknown requested capture'
        indices.update(i for i, p in enumerate(pairs) if p['timestamp'] in requested)
        pairs = [pairs[i] for i in sorted(indices)]
    rt = load_rt(args.rt)[2]
    rows, states = [], Counter()
    total_points = 0
    for repeat in range(args.repeats):
        for index, pair in enumerate(pairs):
            frames, seconds = {}, {}
            order = ('legacy', 'optimized') if (index+repeat) % 2 == 0 else ('optimized', 'legacy')
            for mode in order:
                with legacy_reduction() if mode == 'legacy' else nullcontext():
                    started = perf_counter()
                    frames[mode] = nc.analyze_pair(pair, args.color_cal, .1, 20., 'wf_range')
                    seconds[mode] = perf_counter()-started
            assert_exact(frames['legacy'], frames['optimized'], pair['timestamp'])
            for mode, frame in frames.items():
                with legacy_reduction() if mode == 'legacy' else nullcontext():
                    nc.select_plane_for_rt(frame, rt, args.color_cal)
            assert_exact(frames['legacy'], frames['optimized'], 'association')
            projections = {}
            for mode, frame in frames.items():
                with legacy_reduction() if mode == 'legacy' else nullcontext():
                    projections[mode] = nc.project_to_color(frame, rt, args.color_cal, return_indices=True)
            assert_exact(projections['legacy'], projections['optimized'], 'projection')
            # Exercise serialization as well as the in-memory RGB/XYZ comparison.
            ply = []
            for mode in ('legacy', 'optimized'):
                points, rgb = projections[mode][:2]
                path = args.output/f'{mode}.ply'
                write_binary_ply(path, points, rgb)
                ply.append(path.read_bytes())
            assert ply[0] == ply[1], 'PLY payload changed'
            state = frames['optimized']['board_identity']['state']
            states[state] += 1
            total_points += len(projections['optimized'][0])
            row = dict(stamp=pair['timestamp'], repeat=repeat, seconds=seconds, state=state,
                       candidates=len(frames['optimized']['source_candidates']),
                       points=len(projections['optimized'][0]), all_fields_exact=True, ply_exact=True)
            rows.append(row)
            save_report(args.output/'capture_progress.json', dict(rows=rows))
            print(json.dumps(row), flush=True)
    timings = {m: np.array([r['seconds'][m] for r in rows]) for m in ('legacy', 'optimized')}
    summary = dict(cache_used=False, scope='Existing captures analyzed afresh; NOT new physical camera validation',
                   input_frames=len(pairs), repeats=args.repeats, states=dict(states), exact_projected_points=total_points,
                   python=sys.version, numpy=np.__version__, opencv=cv2.__version__, scipy=scipy.__version__,
                   opencv_threads=cv2.getNumThreads(),
                   timing={m: dict(total=float(v.sum()), mean=float(v.mean()), median=float(np.median(v)),
                                   p95=float(np.percentile(v, 95))) for m, v in timings.items()},
                   paired_reduction_median_percent=float(np.median(1-timings['optimized']/timings['legacy'])*100),
                   slower_frames=int(np.sum(timings['optimized'] > timings['legacy'])), rows=rows)
    save_report(args.output/'capture_summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}), flush=True)


def synthetic(args):
    rng = np.random.default_rng(args.seed)
    rows = []
    for index in range(40):
        width = int(rng.choice([160, 320]))
        center = [*rng.uniform(-.16, .16, 2), rng.uniform(.55, 1.4)]
        parameters = (width, width*3//4, float(rng.uniform(50, 110)), center,
                      rng.uniform(-.7, .7, 3).tolist(), float(rng.uniform(.002, .009)),
                      float(rng.uniform(.05, .40)))
        grid, truth = render(*parameters, index+900, board_present=index < 32)
        candidates = {}
        for mode in ('legacy', 'optimized'):
            with legacy_reduction() if mode == 'legacy' else nullcontext():
                candidates[mode] = nc.nsl_plane_candidates(grid.reshape(-1, 3)*1000, grid.shape[:2])
        assert_exact(candidates['legacy'], candidates['optimized'], f'synthetic[{index}]')
        eligible = [p for p in candidates['optimized'] if nc.support_quality(grid[p['mask']], p)['eligible']]
        scored = [(np.count_nonzero(p['mask'] & truth)/max(1, truth.sum()),
                   np.count_nonzero(p['mask'] & truth)/p['mask'].sum()) for p in eligible]
        recall = max((r for r, purity in scored if purity >= .95), default=0.)
        row = dict(case=index, parameters=parameters, board_present=index < 32,
                   exact=True, eligible=len(eligible), detected=bool(recall >= .70))
        rows.append(row)
        save_report(args.output/'synthetic_progress.json', dict(rows=rows))
        print(json.dumps(row), flush=True)
    save_report(args.output/'synthetic_summary.json', dict(seed=args.seed, cases=len(rows), all_exact=True,
                board_detected=sum(r['detected'] for r in rows[:32]), board_cases=32, no_board_cases=8,
                no_board_with_candidates=sum(r['eligible'] > 0 for r in rows[32:]), rows=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-root', type=Path)
    parser.add_argument('--rt', type=Path)
    parser.add_argument('--color-cal', type=Path, default=ROOT/'cal/c')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=0)
    parser.add_argument('--include-stamp', action='append', default=[])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--seed', type=int, default=51021)
    args = parser.parse_args()
    if args.repeats < 1 or args.samples < 0:
        parser.error('--repeats must be positive and --samples must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        synthetic(args)
    else:
        if not args.capture_root or not args.rt:
            parser.error('--capture-root and --rt are required without --synthetic')
        captures(args)
