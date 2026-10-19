"""Opt-in real capture audit. Visual labels are evaluation only, never model inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from fuse_color_pointcloud import capture_pairs, write_image
from nsl_color_calibration import analyze_pair, camera, pointcloud_view
from nsl_board_identity import identify_board


def audit(root, rt_path, output, baseline_ref):
    output.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(['git', 'show', f'{baseline_ref}:nsl_board_identity.py'], cwd=ROOT)
    baseline = types.ModuleType('baseline_identity')
    exec(compile(source, '<recorded baseline>', 'exec'), baseline.__dict__)
    rt = json.loads(rt_path.read_text(encoding='utf-8'))
    K, D = camera(ROOT/'cal/c')
    rows, tiles, changed = [], [], []
    for number, pair in enumerate(capture_pairs(root), 1):
        f = analyze_pair(pair, ROOT/'cal/c', .1, 20., 'wf_range')
        def fingerprint():
            digest = hashlib.sha256(f['xyz_mm'].tobytes())
            for p in f['source_candidates']: digest.update(p['mask'].tobytes())
            return digest.hexdigest()
        before_data = fingerprint()
        old, new = baseline.identify_board(f, rt, K, D), identify_board(f, rt, K, D)
        assert fingerprint() == before_data, 'Identity must not modify source points or masks'
        if old['plane'] is not None:
            assert new['plane'] is old['plane'], 'Previously recognized boards must keep their original hypothesis'
        assert '제외하고' in new['basis'], 'Every test frame must use its time-block-excluded model'
        row = {'timestamp': pair['timestamp'], 'before': old['state'],
               **{k: v for k, v in new.items() if k != 'plane'}}
        rows.append(row)
        view = pointcloud_view(f['xyz_mm'], f['pcd_shape'], new['plane'], plane_color=(255, 0, 255))
        rgb = cv2.resize(f['color'], (view.shape[1], view.shape[0]))
        tile = np.pad(np.hstack([view, rgb]), ((25, 0), (0, 0), (0, 0)))
        cv2.putText(tile, f'{number} {pair["timestamp"]} {new["state"]}', (3, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
        tiles.append(tile)
        if old['state'] != new['state']:
            previous = pointcloud_view(f['xyz_mm'], f['pcd_shape'], old['plane'], plane_color=(255, 0, 255))
            detail = np.pad(np.hstack([previous, view, rgb]), ((30, 0), (0, 0), (0, 0)))
            cv2.putText(detail, f'{pair["timestamp"]}  BEFORE: {old["state"]}  |  AFTER: {new["state"]}  |  RGB',
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            write_image(output/f'{pair["timestamp"]}.png', detail)
            changed.append(pair['timestamp'])
        print(number, pair['timestamp'], old['state'], '->', new['state'], flush=True)
    for start in range(0, len(tiles), 10):
        page = tiles[start:start+10]
        if len(page) % 2: page.append(np.zeros_like(page[0]))
        write_image(output/f'review_{start//10+1}.png', np.vstack([np.hstack(page[i:i+2]) for i in range(0, len(page), 2)]))
    counts = lambda key: {s: sum(r[key] == s for r in rows) for s in ('matched', 'partial', 'unconfirmed', 'ambiguous')}
    summary = dict(input=len(rows), before=counts('before'), after=counts('state'), changed=changed,
                   baseline_commit=baseline_ref, baseline_source_sha256=hashlib.sha256(source).hexdigest(),
                   rt_sha256=hashlib.sha256(rt_path.read_bytes()).hexdigest(),
                   full_source_preserved=True, time_block_excluded_models=True,
                   metric='board location association including partial; not full-board recovery or RT precision',
                   external_camera_dataset=False, visual_review='pending')
    summary['association_rate'] = (summary['after']['matched']+summary['after']['partial'])/len(rows)
    (output/'frames.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--rt', type=Path, default=ROOT/'output/calibration/nsl_to_color_wf_range_auto_v8_rt.json')
    parser.add_argument('--output', type=Path, default=ROOT/'output/validation/board_identity_20261020')
    parser.add_argument('--baseline', default='56b6a91')
    args = parser.parse_args()
    audit(args.capture_root, args.rt, args.output, args.baseline)
