"""Opt-in cold/warm timing and exact candidate/RT-input parity, no accuracy claim."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nsl_analysis_cache import AnalysisCache
from nsl_color_calibration import analyze_pair, select_plane_for_rt, project_to_color, board_candidates
from fuse_color_pointcloud import capture_pairs, load_rt


def signature(frame, training=False):
    digest = sha256(frame['xyz_mm'].tobytes())
    planes = board_candidates(frame, for_calibration=True) if training else frame['source_candidates']
    for plane in planes:
        for field in ('mask', 'normal', 'h', 'board_center', 'size'):
            digest.update(np.asarray(plane[field]).tobytes())
    return digest.hexdigest()


def main(args):
    cache = AnalysisCache()
    pairs = capture_pairs(args.capture_root)
    rt = load_rt(args.rt)[2]
    result = []
    # Uniform positions in the supplied dataset, not hand-picked successes.
    for index in np.linspace(0, len(pairs)-1, args.samples).astype(int):
        pair = pairs[index]
        inputs = pair, ROOT/'cal/c', .1, 20., 'wf_range'
        start = perf_counter()
        cold = analyze_pair(*inputs)
        cold_seconds = perf_counter()-start
        # Fill the cache with the analysis just measured. Include its cost as a
        # separate first-use overhead, never as part of a disk-hit speedup.
        start = perf_counter()
        cache.write(cache.folder/f'{cache.key(*inputs)}.npz', cold)
        write_seconds = perf_counter()-start
        cache.memory.clear()
        start = perf_counter()
        warm = cache.analyze(*inputs, analyze_pair)
        warm_seconds = perf_counter()-start
        assert signature(cold) == signature(warm)
        assert signature(cold, True) == signature(warm, True)
        select_plane_for_rt(cold, rt, ROOT/'cal/c')
        select_plane_for_rt(warm, rt, ROOT/'cal/c')
        assert cold['board_identity']['state'] == warm['board_identity']['state']
        for a, b in zip(project_to_color(cold, rt, ROOT/'cal/c')[:2],
                        project_to_color(warm, rt, ROOT/'cal/c')[:2]):
            np.testing.assert_array_equal(a, b)
        row = dict(stamp=pair['timestamp'], uncached_seconds=cold_seconds, disk_cached_seconds=warm_seconds,
                   first_use_cache_write_seconds=write_seconds,
                   source_and_training_exact=True, rgb_and_xyz_exact=True)
        print(row, flush=True)
        result.append(row)
    assert cache.misses == 0, 'Disk-hit timing must not include fresh analysis'
    summary = dict(rows=result, samples=len(result),
                   uncached_seconds=sum(r['uncached_seconds'] for r in result),
                   disk_cached_seconds=sum(r['disk_cached_seconds'] for r in result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--rt', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT/'output/validation/cache_timing.json')
    main(parser.parse_args())
