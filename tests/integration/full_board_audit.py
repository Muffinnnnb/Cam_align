"""Real capture regression: matched boards / ALL frames, never including partials."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import nsl_color_calibration as nc
from nsl_board_identity import identify_board, association_model
from nsl_plane_fit import support_quality
from nsl_projection_fit import observation, outline_errors
from nsl_geometry import corrected_points
from fuse_color_pointcloud import capture_pairs, write_image


def historical_module(ref, filename):
    source = subprocess.check_output(['git', 'show', f'{ref}:{filename}'], cwd=ROOT)
    module = types.ModuleType('historical_'+Path(filename).stem)
    module.__file__ = str(ROOT/filename)
    exec(compile(source, f'{ref}:{filename}', 'exec'), module.__dict__)
    return module


def audit(args):
    args.output.mkdir(parents=True, exist_ok=True)
    current = json.loads(args.rt.read_text(encoding='utf-8'))
    previous = json.loads(args.baseline_rt.read_text(encoding='utf-8'))
    baseline = historical_module(args.baseline, 'nsl_color_calibration.py')
    identity = historical_module(args.baseline, 'nsl_board_identity.py')
    K, D = nc.camera(args.color_cal)
    rows, tiles, common = [], [], []
    for number, pair in enumerate(capture_pairs(args.capture_root), 1):
        frame = nc.analyze_pair(pair, args.color_cal, .1, 20., 'wf_range')
        original_hash = sha256(frame['xyz_mm'].tobytes()).hexdigest()
        grid = frame['xyz_mm'].reshape(*frame['pcd_shape'], 3)*.001
        old = dict(frame)
        old['source_candidates'] = baseline.nsl_plane_candidates(frame['xyz_mm'], frame['pcd_shape'], .1, 20.)
        for p in old['source_candidates']:
            p['support_quality'] = support_quality(grid[p['mask']], p)
        before = identity.identify_board(old, previous, K, D)
        after = identify_board(frame, current, K, D)
        assert sha256(frame['xyz_mm'].tobytes()).hexdigest() == original_hash
        assert '제외하고' in after['basis'], 'Require time-block-excluded model for every frame'
        plane = after['plane']
        if after['state'] == 'matched':
            assert plane['support_quality']['eligible'] and after['evidence']['coverage'] >= .60
            assert np.isfinite(grid[plane['mask']]).all()
        row = dict(timestamp=pair['timestamp'], before=before['state'], after=after['state'],
                   basis=after['basis'], evidence=after.get('evidence'),
                   source_points=int(plane['mask'].sum()) if plane else 0,
                   quality=plane['support_quality'] if plane else None,
                   measured_size_mm=(np.array(plane['size'])*1000).tolist() if plane else None,
                   reason=after.get('reason'))
        rows.append(row)
        # Same v9-complete support for BOTH models: do not compare different masks.
        if before['state'] == 'matched':
            item = observation(old, before['plane'])
            metrics = {'timestamp': pair['timestamp']}
            for tag, rt in [('before', previous), ('after', current)]:
                model, _ = association_model(frame, rt, K, D)
                transformed = corrected_points(grid[before['plane']['mask']], model)@np.array(model['R']).T+model['t']
                error = transformed@frame['target_plane']['normal']-frame['target_plane']['h']
                metrics[tag] = dict(point_rms_mm=float(np.sqrt(np.mean(error**2))*1000),
                    boundary_px=outline_errors(item, model, K, D)['boundary_mean_px'],
                    outside=nc.plane_projection_metrics(old, before['plane'], model, args.color_cal)['outside_fraction'])
            common.append(metrics)
        left = nc.pointcloud_view(frame['xyz_mm'], frame['pcd_shape'], before['plane'], plane_color=(255, 0, 255))
        middle = nc.pointcloud_view(frame['xyz_mm'], frame['pcd_shape'], plane,
                                   plane_color=(0,255,0) if after['state']=='matched' else (255,190,0))
        tile = np.pad(np.hstack([left, middle, cv2.resize(frame['color'], (320,240))]), ((27,0),(0,0),(0,0)))
        cv2.putText(tile, f'{number} {pair["timestamp"]}  v9:{before["state"]}  v10:{after["state"]}',
                    (4,18), cv2.FONT_HERSHEY_SIMPLEX, .46, (255,255,255), 1)
        tiles.append(tile)
        if before['state'] != after['state']:
            write_image(args.output/f'{pair["timestamp"]}.png',tile)
        print(number, pair['timestamp'], before['state'], '->', after['state'], flush=True)
    for i in range(0,len(tiles),10):
        page=tiles[i:i+10]
        if len(page)%2:page.append(np.zeros_like(page[0]))
        write_image(args.output/f'review_{i//10+1}.png',np.vstack([np.hstack(page[j:j+2]) for j in range(0,len(page),2)]))
    counts=Counter(r['after'] for r in rows)
    summary=dict(denominator=len(rows), before=dict(Counter(r['before'] for r in rows)),after=dict(counts),
                 matched_rate=counts['matched']/len(rows), partial_included=False,
                 goal_met=counts['matched']/len(rows)>=.90,
                 metric='matched boards / ALL captures; partial, ambiguous, unconfirmed and failures are NOT successes',
                 baseline_commit=args.baseline, rt_sha256=sha256(args.rt.read_bytes()).hexdigest(),
                 all_models_time_block_excluded=True, raw_xyz_preserved=True,
                 visual_review='pending; numeric gates are not a visual ground truth',
                 external_dataset=False)
    summary['common_support_comparison']={'count':len(common), 'description':'same v9 matched supports, time-block-excluded models',
        **{tag:{key:float(np.median([r[tag][key] for r in common])) for key in ('point_rms_mm','boundary_px','outside')}
           for tag in ('before','after')}}
    for filename,data in [('summary.json',summary),('frames.json',rows),('common_support.json',common)]:
        (args.output/filename).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if not summary['goal_met']:
        raise SystemExit('Goal NOT met: do not add partials to the numerator')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--capture-root',type=Path,required=True)
    p.add_argument('--rt',type=Path,required=True)
    p.add_argument('--baseline-rt',type=Path,default=ROOT/'output/calibration/nsl_to_color_wf_range_auto_v9_rt.json')
    p.add_argument('--baseline',default='34de5bf')
    p.add_argument('--color-cal',type=Path,default=ROOT/'cal/c')
    p.add_argument('--output',type=Path,default=ROOT/'output/validation/full_board_20261020/verified')
    audit(p.parse_args())
