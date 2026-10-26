import argparse
import json
from pathlib import Path

import cv2
from app_paths import CAL_DIR


DEFAULT_DIR = CAL_DIR / "d"


def main():
    parser = argparse.ArgumentParser(description="16-bit depth PNG viewer")
    parser.add_argument("image", nargs="?", type=Path, help="depth_raw PNG path")
    args = parser.parse_args()

    path = args.image or max(DEFAULT_DIR.glob("*depth_raw*.png"), key=lambda p: p.stat().st_mtime)
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != "uint16" or depth.ndim != 2:
        raise ValueError(f"16-bit single-channel depth image required: {path}")

    try:
        json_path = path.with_name(path.name.replace("_raw_", "_")).with_suffix(".json")
        scale = json.loads(json_path.read_text(encoding="utf-8"))["depth_scale_m_per_unit"]
    except Exception:
        scale = 0.001

    display = cv2.applyColorMap(
        cv2.convertScaleAbs(depth, alpha=255 / max(1, int(depth.max()))),
        cv2.COLORMAP_JET,
    )

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        raw = int(depth[y, x])
        meters = raw * scale
        view = display.copy()
        text = f"({x}, {y}) raw={raw}, depth={meters:.3f} m"
        cv2.circle(view, (x, y), 4, (255, 255, 255), -1)
        cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Depth (ESC to quit)", view)
        print(text)

    print(f"Loaded: {path.name} | {depth.shape[1]}x{depth.shape[0]} | {depth.dtype}")
    cv2.imshow("Depth (ESC to quit)", display)
    cv2.setMouseCallback("Depth (ESC to quit)", on_mouse)
    while cv2.waitKey(0) != 27:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
