# Cam_align project workflow

- Preserve raw captures, active calibration files, local settings and unrelated
  edits. Keep generated outputs under `output/`; do not commit private captures.
- Keep detection, full board correspondence, RT training membership and alignment
  error separate. Never count partial matches as full matches or relax criteria
  merely to improve a displayed percentage.
- For every implemented project update, append the change, reason, measured
  validation, limitations and commit reference to the existing Notion notebook:
  https://app.notion.com/p/3ffbc9ba81a881129714eaaefd626c42
  Read before editing; preserve the historical entries and images. If access is
  unavailable, report the pending notebook update rather than claiming it saved.
- Extend `docs/algorithm-evolution.html` in place when documenting algorithm
  changes; do not replace its previous explanations with a new document.
