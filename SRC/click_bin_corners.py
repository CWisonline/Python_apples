"""
Manually mark (or correct) the four outer rim corners of the apple bin.

Writes/updates DATA/bin_corners.csv, which apple_counter_v2.py uses in
preference to automatic rim detection for any image listed in it.

Usage
-----
    python SRC/click_bin_corners.py                 # all images in RAW_DIR
    python SRC/click_bin_corners.py DSCN9917.JPG    # just these files
    python SRC/click_bin_corners.py --flagged       # only images whose bin_qc
                                                    # in image_summary.csv is not auto_ok

In the window
-------------
  * The automatic estimate is drawn in yellow.
  * Click the 4 OUTER rim corners in order: top-left, top-right,
    bottom-right, bottom-left. Where a corner is hidden under a pole, click
    where the two rim edges would meet.
  * Enter / Space : accept (your 4 clicks, or the yellow estimate if you
                    made no clicks)
  * Backspace     : undo last click
  * s             : skip this image (no change to the CSV)
  * Esc           : save and quit
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apple_counter_v2 as ac  # noqa: E402

DISPLAY_W = 1600
FIELDS = ["filename", "tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"]
NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]


def load(path):
    out = {}
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                out[r["filename"]] = r
    return out


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])


def annotate(img, auto):
    s = DISPLAY_W / img.shape[1]
    base = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    clicks = []

    def redraw():
        v = base.copy()
        cv2.polylines(v, [np.round(auto * s).astype(np.int32)], True, (0, 255, 255), 1)
        for i, p in enumerate(clicks):
            cv2.circle(v, p, 6, (0, 0, 255), 2)
            cv2.putText(v, NAMES[i], (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(clicks) > 1:
            cv2.polylines(v, [np.array(clicks, np.int32)], len(clicks) == 4, (0, 0, 255), 2)
        msg = (f"click {NAMES[len(clicks)]}" if len(clicks) < 4 else "Enter = accept") + \
              "   |  Enter=accept  Backspace=undo  s=skip  Esc=quit"
        cv2.putText(v, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("bin corners", v)

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            redraw()

    cv2.namedWindow("bin corners", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("bin corners", on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 32):                       # Enter / Space
            if len(clicks) == 4:
                return np.array(clicks, float) / s
            if not clicks:
                return auto
        elif k == 8 and clicks:                 # Backspace
            clicks.pop()
            redraw()
        elif k == ord("s"):
            return "skip"
        elif k == 27:                           # Esc
            return "quit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--raw", type=Path, default=ac.RAW_DIR)
    ap.add_argument("--corners", type=Path, default=ac.CORNERS_CSV)
    ap.add_argument("--flagged", action="store_true",
                    help="only images not flagged auto_ok in OUT_DIR/image_summary.csv")
    args = ap.parse_args()

    paths = [args.raw / f for f in args.files] if args.files else ac.list_images(args.raw)
    if args.flagged:
        summ = pd.read_csv(ac.OUT_DIR / "image_summary.csv")
        bad = set(summ.loc[summ.bin_qc != "auto_ok", "filename"])
        paths = [p for p in paths if p.name in bad]
    rows = load(args.corners)
    print(f"{len(paths)} images to review; corners file: {args.corners}")
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  cannot read {p}")
            continue
        if p.name in rows:
            r = rows[p.name]
            auto = np.array([[float(r[f"{c}_x"]), float(r[f"{c}_y"])] for c in ("tl", "tr", "br", "bl")])
        else:
            auto, _, _ = ac.detect_bin_corners(img)
        res = annotate(img, auto)
        if isinstance(res, str) and res == "quit":
            break
        if isinstance(res, str) and res == "skip":
            continue
        rows[p.name] = {"filename": p.name, **{f"{c}_{ax}": f"{res[i][j]:.1f}"
                        for i, c in enumerate(("tl", "tr", "br", "bl")) for j, ax in enumerate("xy")}}
        save(args.corners, rows)
        print(f"  saved {p.name}")
    cv2.destroyAllWindows()
    save(args.corners, rows)


if __name__ == "__main__":
    main()
