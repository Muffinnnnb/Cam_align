"""Full detection under synthetic poses, FOVs, densities, noise and dropout.

Not an external-camera accuracy benchmark. Truth masks are used ONLY to score
the outputs; neither detector receives them. RGB/RT association is tested in
test_nsl_board_identity.py, independently of this source detection audit.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import nsl_color_calibration as nc
from nsl_plane_fit import support_quality


def render(width, height, fov, center, pose, noise, dropout, seed, board_present=True):
    rng = np.random.default_rng(seed)
    focal = width/(2*np.tan(np.deg2rad(fov)/2))
    yy, xx = np.mgrid[:height, :width]
    rays = np.stack([(xx-(width-1)/2)/focal, (yy-(height-1)/2)/focal, np.ones_like(xx)], -1)
    basis = cv2.Rodrigues(np.array(pose, float))[0]
    normal, center = basis[:, 2], np.array(center, float)
    depth = (center@normal)/(rays@normal)
    local = (rays*depth[..., None]-center)@basis
    truth = (abs(local[..., 0]) <= .125) & (abs(local[..., 1]) <= .14) & (depth > 0) & board_present
    background = np.full((height, width), center[2]+.50)
    ellipse = ((xx-width*.57)/(width*.23))**2+((yy-height*.70)/(height*.38))**2
    body = ellipse < 1
    background[body] = center[2]+.12+.09*ellipse[body]
    # Correct visibility: a board behind another surface is not visible.
    truth &= depth < background
    z = np.where(truth, depth, background)+rng.normal(0, noise, (height, width))
    grid = rays*z[..., None]
    grid[rng.random((height, width)) < dropout] = np.nan
    return grid, truth & np.isfinite(grid).all(2)


def audit():
    fixed = [
        (320, 240, 70, (0, 0, .8), (0, 0, 0), .003, .1),
        (160, 120, 70, (0, 0, .8), (0, 0, 0), .003, .1),
        (320, 240, 110, (0, 0, .55), (.2, .5, .1), .005, .35),
        (320, 240, 50, (.17, -.12, 1.2), (-.4, .4, -.6), .005, .2),
        (320, 240, 90, (-.18, .05, .75), (.1, -.7, .5), .008, .2),
        (160, 120, 90, (-.10, .05, .6), (.1, -.5, .5), .004, .25),
        (320, 240, 70, (.25, .1, 1.3), (.4, .7, -.4), .006, .3),
        (320, 240, 70, (-.3, -.1, 1.5), (-.3, -.5, .2), .006, .1),
    ]
    cases = [(args, i+31, True) for i, args in enumerate(fixed)]
    rng = np.random.default_rng(8127)
    for i in range(32):
        width = int(rng.choice([160, 320]))
        center = [*rng.uniform(-.16, .16, 2), rng.uniform(.55, 1.4)]
        args = (width, width*3//4, float(rng.uniform(50, 110)), center,
                rng.uniform(-.7, .7, 3).tolist(), float(rng.uniform(.002, .009)), float(rng.uniform(.05, .40)))
        cases.append((args, i+900, i < 24))
    rows = []
    for case, (args, seed, present) in enumerate(cases):
        grid, truth = render(*args, seed, board_present=present)
        original = grid.copy()
        variants = {}
        for variant in ('baseline', 'metric_fallback'):
            context = patch.object(nc, 'metric_support', return_value=[]) if variant == 'baseline' else patch.object(nc, 'metric_support', wraps=nc.metric_support)
            with context:
                planes = nc.nsl_plane_candidates(grid.reshape(-1, 3)*1000, grid.shape[:2])
            eligible = [p for p in planes if support_quality(grid[p['mask']], p)['eligible']]
            scored = [(np.count_nonzero(p['mask'] & truth)/max(1, truth.sum()),
                       np.count_nonzero(p['mask'] & truth)/p['mask'].sum()) for p in eligible]
            recall = float(max([r for r, purity in scored if purity >= .95], default=0.))
            variants[variant] = dict(recall=recall, detected=recall >= .70, eligible_candidates=len(eligible))
        np.testing.assert_array_equal(grid, original)
        row = dict(case=case, parameters=args, seed=seed, board_present=present,
                   visible_points=int(truth.sum()), **variants)
        rows.append(row); print(json.dumps(row), flush=True)
    board_rows = [r for r in rows if r['board_present']]
    summary = dict(description='Synthetic source detection, NOT measured cross-camera performance',
        criterion='>=70% recall of measured visible board samples, >=95% purity; all existing structural gates',
        board_cases=len(board_rows), no_board_cases=len(rows)-len(board_rows),
        baseline_detected=sum(int(r['baseline']['detected']) for r in board_rows),
        fallback_detected=sum(int(r['metric_fallback']['detected']) for r in board_rows),
        regressions=[r['case'] for r in rows if r['baseline']['detected'] and not r['metric_fallback']['detected']],
        new_no_board_candidates=sum(max(0, r['metric_fallback']['eligible_candidates']-r['baseline']['eligible_candidates']) for r in rows if not r['board_present']),
        rows=rows)
    output = ROOT/'output/validation/metric_support'
    output.mkdir(parents=True, exist_ok=True)
    (output/'synthetic_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    assert not summary['regressions'] and not summary['new_no_board_candidates'], summary
    assert summary['fallback_detected'] > summary['baseline_detected'], summary
    print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    audit()
