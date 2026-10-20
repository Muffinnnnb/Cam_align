"""Re-read captures and check fallback against the unchanged detector path."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import nsl_color_calibration as nc
from nsl_board_identity import identify_board
from nsl_plane_fit import support_quality
from fuse_color_pointcloud import capture_pairs


def signature(planes):
    digest = sha256()
    for p in planes:
        for key in ('mask', 'normal', 'h', 'board_center'):
            digest.update(np.asarray(p[key]).tobytes())
    return digest.hexdigest()


def audit(args):
    rt_hash = sha256(args.rt.read_bytes()).hexdigest()
    rt = json.loads(args.rt.read_text(encoding='utf-8'))
    K, D = nc.camera(args.color_cal)
    rows = []
    for pair in capture_pairs(args.capture_root):
        frame = nc.analyze_pair(pair, args.color_cal, .1, 20., 'wf_range')
        raw_hash = sha256(frame['xyz_mm'].tobytes()).hexdigest()
        grid = frame['xyz_mm'].reshape(*frame['pcd_shape'], 3)*.001
        with patch.object(nc, 'metric_support', return_value=[]), patch.object(nc, 'plane_conditioned_support', return_value=[]), patch.object(nc, 'local_support', return_value=[]):
            old_planes = nc.nsl_plane_candidates(frame['xyz_mm'], frame['pcd_shape'], .1, 20.)
        for p in old_planes:
            p['support_quality'] = support_quality(grid[p['mask']], p)
        old_frame = dict(frame, source_candidates=old_planes)
        before = identify_board(old_frame, rt, K, D)
        after = identify_board(frame, rt, K, D)
        identical = signature(old_planes) == signature(frame['source_candidates'])
        assert sha256(frame['xyz_mm'].tobytes()).hexdigest() == raw_hash
        assert '제외하고' in after['basis'], 'Require exact-input time-block-excluded RT'
        assert before['state'] != 'matched' or after['state'] == 'matched'
        row = dict(timestamp=pair['timestamp'], before=before['state'], after=after['state'],
                   candidate_supports_identical=identical,
                   extra_candidates=len(frame['source_candidates'])-len(old_planes))
        rows.append(row); print(row, flush=True)
    assert sha256(args.rt.read_bytes()).hexdigest() == rt_hash
    summary = dict(denominator=len(rows), before=dict(Counter(r['before'] for r in rows)),
        after=dict(Counter(r['after'] for r in rows)), partial_included=False,
        all_source_supports_identical=all(r['candidate_supports_identical'] for r in rows),
        rt_sha256=rt_hash, rt_unchanged=True, raw_xyz_preserved=True,
        external_dataset=False, rows=rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'actual_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--capture-root', type=Path, required=True)
    p.add_argument('--rt', type=Path, required=True)
    p.add_argument('--color-cal', type=Path, default=ROOT/'cal/c')
    p.add_argument('--output', type=Path, default=ROOT/'output/validation/metric_support')
    audit(p.parse_args())
