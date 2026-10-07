"""Historical pre-v6 experiment; not the current detector's acceptance test."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import nsl_color_calibration as nc
from app_paths import OUTPUT_DIR, calibration_input
from fuse_color_pointcloud import DEFAULT_CAPTURE_ROOT, DEFAULT_COLOR_CAL, write_image


def load_frames(folder):
    # Recompute from the original files; never validate against stale masks.
    return nc.analyze_capture()


def plane_errors(frames, result):
    rotation, translation = np.array(result["R"]), np.array(result["t"])
    angles, distances = [], []
    for frame in frames.values():
        source, target = frame["source_plane"], frame["target_plane"]
        predicted = rotation @ source["normal"]
        angles.append(np.degrees(np.arccos(np.clip(predicted @ target["normal"], -1, 1))))
        distances.append(source["h"] + predicted @ translation - target["h"])
    return {"normal_rmse_deg": float(np.sqrt(np.mean(np.square(angles)))),
            "plane_rmse_mm": float(np.sqrt(np.mean(np.square(distances))) * 1000),
            "score": float(np.mean((np.array(angles) / 3)**2 + (np.array(distances) / .020)**2))}


def check_prior_refinement(frames, baseline):
    """Reproduce the proposed tight-slab loop without modifying the saved RT.

    Freeze validation masks before any iterations. This is a sensitivity test,
    NOT an independent accuracy benchmark: the seed RT used all input captures.
    """
    def extract(result):
        selected = {}
        for name, frame in frames.items():
            plane = nc.guided_plane_from_color(frame, result, plane_tolerance=.012,
                                               min_coverage=.25, max_angle_deg=6,
                                               max_plane_error_m=.040)
            if plane is not None:
                selected[name] = {**frame, "source_plane": plane}
        return selected

    selected = extract(baseline)
    holdout = {name: selected[name] for name in sorted(selected)[::5]}
    if len(holdout) < 3 or len(selected) - len(holdout) < 6:
        return {"status": "insufficient validation planes", "automatically_applied": False}
    report = {"validation_frames": list(holdout), "baseline": plane_errors(holdout, baseline),
              "note": "Baseline-conditioned fixed-mask sensitivity check; seed RT uses all frames. Not independent accuracy.",
              "automatically_applied": False, "iterations": []}
    current = baseline
    for index in range(3):
        if index:
            selected = extract(current)
        training = {name: frame for name, frame in selected.items() if name not in holdout}
        try:
            candidate = nc.calibrate(training, None, source_plane_key="source_plane",
                                     target_plane_key="target_plane", source_name="nsl", target_name="color",
                                     ransac_angle_deg=3, ransac_distance_m=.020)
        except RuntimeError as exc:
            report["iterations"].append({"iteration": index + 1, "error": str(exc)})
            break
        rotation_step = cv2.Rodrigues(np.array(candidate["R"]) @ np.array(current["R"]).T)[0]
        report["iterations"].append({
            "iteration": index + 1, "training_inliers": candidate["ransac_inliers"],
            "validation": plane_errors(holdout, candidate),
            "rotation_step_deg": float(np.rad2deg(np.linalg.norm(rotation_step))),
            "translation_step_mm": float(np.linalg.norm(np.array(candidate["t"]) - current["t"]) * 1000),
        })
        current = candidate
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "validation" / "nsl")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--check-refinement", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = load_frames(args.output)
    if args.baseline:
        result = json.loads(calibration_input(nc.RESULT_PATH).read_text(encoding="utf-8"))
    else:
        result = nc.calibrate_nsl_color(frames, args.output / "candidate_rt.json", progress=lambda msg: print(msg, flush=True))
    if args.check_refinement:
        report = check_prior_refinement(frames, result)
        (args.output / "refinement_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report, flush=True)
    rows = []
    for stamp, frame in frames.items():
        plane = nc.select_plane_for_rt(frame, result)
        rows.append({"stamp": stamp, "selected": plane is not None,
                     "angle_deg": frame.get("rt_normal_error_deg"),
                     "plane_mm": None if plane is None else frame["rt_plane_error_m"] * 1000})
        if plane is not None:
            assert np.count_nonzero(plane["mask"]) == plane["inliers"]
            points = frame["xyz_mm"][np.asarray(plane["mask"], bool).ravel()] * .001
            assert np.allclose(points.mean(0), plane["center"], atol=1e-8)
        if stamp == "20260922_093112_477":
            frame["inspection_plane"] = nc.guided_plane_from_color(frame, result)
            inspection = frame["inspection_plane"]
            assert inspection is not None
            pts = frame["xyz_mm"][np.asarray(inspection["mask"], bool).ravel()] * .001
            assert len(pts) == inspection["inliers"]
            assert np.allclose(pts.mean(0), inspection["center"])
            left = nc.pointcloud_view(frame["xyz_mm"], frame["pcd_shape"], inspection, plane_color=(255, 255, 0))
            right = nc.color_plane_view(frame["color"], frame["target_plane"])
            left = cv2.resize(left, (right.shape[1], right.shape[0]), interpolation=cv2.INTER_NEAREST)
            write_image(args.output / "first_frame.png", np.hstack((left, right)))
            nc.save_source_cloud(args.output / "first_frame_inspection.ply", frame, inspection=True)
            nc.save_fused_cloud(args.output / "first_frame_color.ply", frame, result)
            nc.save_projection(args.output / "first_frame_projection.png", frame, result)
        if frame.get("rt_inlier"):
            assert frame["rt_normal_error_deg"] < result["ransac_angle_threshold_deg"]
            assert frame["rt_plane_error_m"] < result["ransac_plane_threshold_m"]
            assert frame["projection_overlap"] >= result["projection_overlap_threshold"]
    if result.get("fitted_plane_selection"):
        assert sum(f.get("rt_inlier", False) for f in frames.values()) == len(result["final_validated_frames"])
    (args.output / "frame_checks.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print({key:result[key] for key in ("plane_pairs", "ransac_inliers", "normal_rmse_deg", "plane_rmse_m")}, flush=True)


if __name__ == "__main__":
    main()
