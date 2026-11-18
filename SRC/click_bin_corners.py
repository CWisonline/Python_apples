r"""
Manually mark (or correct) the four outer rim corners of the apple bin.

Writes/updates DATA/bin_corners_<raw folder>.csv, which apple_counter_v2.py
uses (for the same --raw folder) in preference to automatic rim detection for
any image listed in it.

Usage (from the repository folder; Windows paths shown)
-----
  All images in a folder:
    python SRC\click_bin_corners.py --raw DATA\raw_test_bin

  Only images flagged by a previous counter run (bin_qc not auto_ok), using
  the image_summary.csv in that run's output folder:
    python SRC\click_bin_corners.py --raw DATA\raw_test_bin --out DATA\processed_v2 --flagged

  Particular images:
    python SRC\click_bin_corners.py --raw DATA\raw_test_bin DSCN9917.JPG DSCN9939.JPG

  Then re-run the counter with the same --raw folder; it picks the corners up.

In the window
-------------
  * The window is sized to fit your screen (use --scale 0.2, say, if it is
    still too big or small). A magnifier near the cursor shows the photo at
    full resolution: aim the cross hair at the corner, then click.
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

SCREEN_FILL_W, SCREEN_FILL_H = 0.90, 0.85   # fraction of the screen the window may use
MAG_SIZE = 220        # magnifier box on screen (px)
MAG_ZOOM_FULLRES = 1  # magnifier shows the photo at this many screen px per photo px
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


def screen_size():
    """Usable screen size in the units OpenCV windows use (logical pixels)."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1280, 720


def annotate(img, auto, scale=None, label=""):
    H0, W0 = img.shape[:2]
    if scale is None:
        sw, sh = screen_size()
        scale = min(SCREEN_FILL_W * sw / W0, SCREEN_FILL_H * sh / H0)
    s = scale
    base = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bh, bw = base.shape[:2]
    fs = max(0.45, bw / 1800)                # text size relative to the window
    th = 1
    banner = int(34 * fs / 0.55)             # height of the instruction strip
    clicks = []
    cursor = [None]
    mag = min(MAG_SIZE, bw // 3, bh // 3)    # magnifier size on screen
    half = int(mag / (2 * MAG_ZOOM_FULLRES))  # half-width of the magnified photo region (photo px)
    pad_img = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    auto_full = np.asarray(auto, float)

    def magnifier(x, y):
        fx, fy = int(round(x / s)), int(round(y / s))   # photo coordinates
        crop = pad_img[fy:fy + 2 * half, fx:fx + 2 * half].copy()
        # draw the automatic outline and clicks in photo coordinates
        off = np.array([fx - half, fy - half], float)
        cv2.polylines(crop, [np.round(auto_full - off).astype(np.int32)], True, (0, 255, 255), 2)
        for p in clicks:
            q = np.round(np.array(p) / s - off).astype(int)
            cv2.circle(crop, tuple(q), 8, (0, 0, 255), 2)
        m = cv2.resize(crop, (mag, mag), interpolation=cv2.INTER_NEAREST)
        c0 = mag // 2
        cv2.line(m, (c0, 0), (c0, mag), (0, 255, 0), 1)
        cv2.line(m, (0, c0), (mag, c0), (0, 255, 0), 1)
        cv2.rectangle(m, (0, 0), (mag - 1, mag - 1), (255, 255, 255), 2)
        return m

    def redraw():
        v = base.copy()
        cv2.polylines(v, [np.round(auto_full * s).astype(np.int32)], True, (0, 255, 255), 1)
        for i, p in enumerate(clicks):
            cv2.circle(v, p, 6, (0, 0, 255), 2)
            cv2.putText(v, NAMES[i], (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), th)
        if len(clicks) > 1:
            cv2.polylines(v, [np.array(clicks, np.int32)], len(clicks) == 4, (0, 0, 255), 2)
        if cursor[0] is not None:
            x, y = cursor[0]
            m = magnifier(x, y)
            # put the magnifier in the corner furthest from the cursor
            mx = 10 if x > bw / 2 else bw - mag - 10
            my = banner + 10 if y > bh / 2 else bh - mag - 10
            v[my:my + mag, mx:mx + mag] = m
        step = (f"Click {NAMES[len(clicks)].upper()} corner ({len(clicks) + 1}/4)" if len(clicks) < 4
                else "Press Enter to accept")
        keys = "Enter=accept  Backspace=undo  s=skip  Esc=quit"
        cv2.rectangle(v, (0, 0), (bw, banner), (40, 40, 40), -1)
        y_txt = int(banner * 0.68)
        if label:   # file name and progress, e.g. "DSCN9917.JPG  (3 of 12)"
            cv2.putText(v, label, (10, y_txt), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th + 1, cv2.LINE_AA)
            (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th + 1)
            step_x = 10 + lw + int(25 * fs / 0.55)
        else:
            step_x = 10
        cv2.putText(v, step, (step_x, y_txt), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), th + 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(keys, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, th)
        cv2.putText(v, keys, (bw - tw - 10, y_txt), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, (255, 255, 255), th,
                    cv2.LINE_AA)
        cv2.imshow("bin corners", v)

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_MOUSEMOVE:
            cursor[0] = (x, y)
            redraw()
        elif ev == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            redraw()

    cv2.namedWindow("bin corners", cv2.WINDOW_AUTOSIZE)   # 1 window px = 1 image px: clicks map exactly
    if label:
        try:
            cv2.setWindowTitle("bin corners", f"bin corners - {label}")
        except Exception:
            pass
    cv2.moveWindow("bin corners", 0, 0)
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
    ap.add_argument("files", nargs="*", help="image file names in --raw (default: all images)")
    ap.add_argument("--raw", type=Path, default=ac.RAW_DIR, help="folder of original photos")
    ap.add_argument("--out", type=Path, default=ac.OUT_DIR,
                    help="output folder of a previous counter run (holds image_summary.csv); used with --flagged")
    ap.add_argument("--corners", type=Path, default=None,
                    help="corners CSV to write; default DATA/bin_corners_<raw folder>.csv")
    ap.add_argument("--scale", type=float, default=None,
                    help="display scale (e.g. 0.2); default: fit the screen")
    ap.add_argument("--flagged", action="store_true",
                    help="only images whose bin_qc in <out>/image_summary.csv is not auto_ok")
    args = ap.parse_args()
    args.corners = args.corners or ac.default_corners_csv(args.raw)
    if not args.raw.is_dir():
        sys.exit(f"Image folder not found: {args.raw.resolve()}")
    for f in args.files:
        if not (args.raw / f).is_file():
            sys.exit(f"Image not found: {(args.raw / f).resolve()}\n"
                     "(Positional arguments are image file names; use --out for the results folder.)")

    paths = [args.raw / f for f in args.files] if args.files else ac.list_images(args.raw)
    if args.flagged:
        summ_path = args.out / "image_summary.csv"
        if not summ_path.is_file():
            sys.exit(f"--flagged needs {summ_path.resolve()}\n"
                     "Run apple_counter_v2.py first, and give its output folder with --out.")
        summ = pd.read_csv(summ_path)
        bad = set(summ.loc[summ.bin_qc != "auto_ok", "filename"])
        paths = [p for p in paths if p.name in bad]
    rows = load(args.corners)
    print(f"{len(paths)} images to review; corners file: {args.corners.resolve()}")
    for n_img, p in enumerate(paths, start=1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  cannot read {p}")
            continue
        if p.name in rows:
            r = rows[p.name]
            auto = np.array([[float(r[f"{c}_x"]), float(r[f"{c}_y"])] for c in ("tl", "tr", "br", "bl")])
        else:
            auto, _, _ = ac.detect_bin_corners(img)
        res = annotate(img, auto, args.scale, label=f"{p.name}  ({n_img} of {len(paths)})")
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
