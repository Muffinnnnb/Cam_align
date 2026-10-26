"""Before/after structural refactor parity. Existing RT read-only; no raw writes."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nsl_analysis_cache import AnalysisCache
import nsl_color_calibration as nc
import calibrate_d_to_r as dc
from fuse_color_pointcloud import capture_pairs, load_rt


def digest(value):
    h = sha256()
    def visit(v):
        if isinstance(v, np.ndarray):
            h.update(str((v.dtype.str, v.shape)).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for key in sorted(v):
                visit(key); visit(v[key])
        elif isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
        else:
            h.update(repr(v).encode())
        h.update(b'\0')
    visit(value)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    rt = load_rt(ROOT/'output/calibration/nsl_to_color_wf_range_auto_v10_rt.json')[2]
    cache, records, states = AnalysisCache(), {}, Counter()
    expected = json.loads(args.reference.read_text(encoding='utf-8')) if args.compare else None
    for i, pair in enumerate(capture_pairs(args.capture_root)):
        inputs = pair, ROOT/'cal/c', .1, 20., 'wf_range'
        frame = nc.analyze_pair(*inputs) if args.compare else cache.analyze(*inputs, nc.analyze_pair)
        raw = digest(frame)
        nc.select_plane_for_rt(frame, rt, ROOT/'cal/c')
        association = digest(frame)
        projected = nc.project_to_color(frame, rt, ROOT/'cal/c', return_indices=True)
        records[pair['timestamp']] = [raw, association, digest(projected)]
        states[frame['board_identity']['state']] += 1
        if expected:
            original = expected['nsl'][pair['timestamp']]
            assert raw == original[0], f'Source analysis changed: {pair["timestamp"]}'
            assert digest(projected) == original[2], f'XYZ/RGB/UV changed: {pair["timestamp"]}'
            assert frame['board_identity']['state'] == 'matched', pair['timestamp']
            if expected.get('version', 1) >= 2:
                assert association == original[1], f'Association changed: {pair["timestamp"]}'
            # v1 reference's second hash included _color_projection's Python id()
            # cache keys. Those are not cross-process measurement values. Compare
            # every raw candidate field and every projection field above instead.
        print(f'{i+1}: {pair["timestamp"]}', flush=True)
    depth = {}
    pairs = dc.frame_pairs()
    for stamp in list(pairs)[::max(1, len(pairs)//4)][:4]:
        depth[stamp] = digest(dc.analyze_frame(stamp))
    report = dict(version=2, nsl=records, depth=depth)
    if args.compare:
        assert records.keys() == expected['nsl'].keys()
        assert depth == expected['depth'], 'Depth analysis changed'
        args.reference.with_name('after.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    else:
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        args.reference.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(compared=args.compare, nsl=len(records), depth=len(depth),
                         states=dict(states), cache_hits=cache.hits, cache_misses=cache.misses)), flush=True)


if __name__ == '__main__':
    main()
