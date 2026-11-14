"""
Pair each "reference sheet" photo with its apple photo and read the code.

For every tree the bin is photographed twice: once with a white sheet bearing
the tree's reference code (e.g. "11B"), and once without. This script:

1. finds the bin (same rim detection as apple_counter_v2) and rectifies it,
2. finds white sheet(s) inside the bin, including partly shaded sheets,
3. finds the printed characters, groups them into codes, and reads each code
   with Tesseract OCR (characters read one at a time, both orientations),
4. pairs each sheet photo with the next photo (by capture time) that has no
   sheet, and checks the pairing by comparing the fruit layout,
5. writes DATA/image_pairs_<raw folder>.csv and a small image of every code it read, so
   the codes can be checked by eye.

Output columns (image_pairs_<raw folder>.csv)
--------------------------------
filename_1          photo with the reference sheet
filename_2          paired apple photo (no sheet)
reference_code      code(s) read; several codes on one photo are joined by ";"
tree_id             tree number(s) from the code(s)
crates              crate letter(s) from the code(s), joined by ";"
n_codes             number of codes found on the sheet photo
ocr_confidence      lowest Tesseract confidence (0-100) of the code(s)
time_1, time_2      capture times (EXIF DateTimeOriginal)
time_gap_s          seconds between the two photos
layout_similarity   overlap (0-1) of the fruit layout outside the sheet area;
                    true pairs scored 0.55-0.91, unrelated photos 0.22-0.33
                    in the examples
flags               see FLAGS below; empty = no problems found
code_image          image of the code region(s) as read, for checking

FLAGS
-----
check_code          a code is not of the expected form or confidence < MIN_CONF
mixed_trees         codes from more than one tree on one sheet photo
                    (several crates of the SAME tree are normal, not flagged)
duplicate_code      the same code was read on more than one sheet photo
no_code_read        a sheet was found but no code could be read
check_pair          layout_similarity < MIN_SIMILARITY or time gap > MAX_GAP_S
no_apple_photo      no following photo without a sheet
apple_photo_shared  the same apple photo follows two sheet photos (e.g. a retake)
unpaired_apple_photo  (rows with empty filename_1) apple photo with no sheet photo

Requirements: as apple_counter_v2, plus pytesseract and the Tesseract OCR
program (Windows installer: https://github.com/UB-Mannheim/tesseract/wiki).
If Tesseract is not on the PATH, set TESSERACT_CMD below.

Usage:  python SRC/pair_reference_sheets.py [--raw DIR] [--out CSV]
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apple_counter_v2 as A  # noqa: E402

# =============================== USER SETTINGS =============================== #

RAW_DIR = A.RAW_DIR
# Default outputs are named after the raw folder (camera file numbers repeat
# between seasons): DATA/image_pairs_<raw folder>.csv, DATA/reference_codes_<raw folder>/

# Set this if Tesseract is not on the PATH, e.g.
#  r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Reference code = tree number + crate letter (e.g. "20A", "20B": tree 20,
# crates A and B). If trees can have more crates, add the letters to BOTH
# settings, e.g. "[A-D]" and "0123456789ABCD".
CODE_PATTERN = r"^(\d{1,3})([AB])$"   # groups: (tree number)(crate letter)
CODE_CHARS = "0123456789AB"           # characters Tesseract may return
MIN_CONF = 70.0                   # flag codes read with lower confidence
MIN_SIMILARITY = 0.45             # flag pairs with less fruit-layout overlap
MAX_GAP_S = 120.0                 # flag pairs taken further apart than this

# Sheet / text detection (mm, in the rectified bin image)
MIN_SHEET_AREA_MM2 = 15000        # A4 = 62,370 mm2; allows folded/partly hidden sheets
SHEET_GROW_MM = 60                # how far sunlit paper is grown into shaded paper
SHEET_MIN_MEDIAN_L = 80           # sheets: median L* 85-97; sunlit/shaded floor patches 46-75
SHEET_SIDE_MM = (150, 450)        # short side >= 150 mm, long side <= 450 mm (two touching A4 ~ 390)
SHEET_MIN_RECTNESS = 0.70         # area / bounding-rectangle area (sheets 0.74-0.87)
CHAR_MIN_MM, CHAR_MAX_MM = 15, 110  # character size limits
CHAR_GROUP_MM = 50                # characters closer than this belong to one code

# ============================================================================ #

PPM = A.RECT_PX_PER_MM
PAT = re.compile(CODE_PATTERN)


def capture_time(path: Path):
    try:
        from PIL import Image
        ex = Image.open(path)._getexif() or {}
        t = ex.get(36867) or ex.get(306)          # DateTimeOriginal, DateTime
        return datetime.strptime(t, "%Y:%m:%d %H:%M:%S") if t else None
    except Exception:
        return None


# ------------------------------ sheet & text -------------------------------- #

def sheet_mask(rect):
    """White sheet(s): sunlit paper (bright, near-neutral) as seeds, grown into
    adjacent shaded paper, which is lit by blue sky (neutral-to-bluish)."""
    L, a, b = A.to_lab(rect)
    C = np.hypot(a, b)
    roi = A.roi_mask(rect.shape) > 0
    seed = ((L > 75) & (C < 18) & roi).astype(np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    shaded = (L > 50) & ((C < 15) | ((b < -5) & (np.abs(a) < 8) & (C < 32)))
    cand = ((shaded | (seed > 0)) & roi).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # bridge ink
    grown = seed.copy()
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(int(SHEET_GROW_MM * PPM)):     # geodesic dilation
        nxt = cv2.dilate(grown, k3) & cand
        if np.array_equal(nxt, grown):
            break
        grown = nxt
    m = ndi.binary_fill_holes(grown).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for k in range(1, n):
        if st[k, 4] / PPM ** 2 < MIN_SHEET_AREA_MM2:
            continue
        reg = lab == k
        cnt = max(cv2.findContours(reg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                  key=cv2.contourArea)
        (_, _), (w, h), _ = cv2.minAreaRect(cnt)
        short, long_ = min(w, h) / PPM, max(w, h) / PPM
        rectness = st[k, 4] / max(w * h, 1)
        # paper is much lighter than sunlit or shaded floor, and sheet-shaped
        if (np.median(L[reg]) >= SHEET_MIN_MEDIAN_L and SHEET_SIDE_MM[0] <= short
                and long_ <= SHEET_SIDE_MM[1] and rectness >= SHEET_MIN_RECTNESS):
            keep[reg] = 1
    return keep, L


def text_groups(rect):
    """Return (list of (minAreaRect, char_mask) per code, sheet_mask)."""
    m, L = sheet_mask(rect)
    if not m.any():
        return [], m
    inner = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(10 * PPM) | 1,) * 2))
    k_p = int(15 * PPM) | 1                            # wider than a print stroke
    paper = cv2.morphologyEx(L.astype(np.float32), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_p, k_p)))
    ink = ((paper - L) > 15) & (inner > 0)
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink)
    chars = np.zeros_like(ink)
    for k in range(1, n):
        w, h, area = st[k, 2] / PPM, st[k, 3] / PPM, st[k, 4] / PPM ** 2
        if CHAR_MIN_MM <= max(w, h) <= CHAR_MAX_MM and area >= 40 and min(w, h) >= 3:
            chars[lab == k] = 1
    k = int(CHAR_GROUP_MM * PPM) | 1
    grp = cv2.dilate(chars, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, glab, _, _ = cv2.connectedComponentsWithStats(grp)
    groups = []
    for g in range(1, n):
        sel = (glab == g) & (chars > 0)
        if ndi.label(sel)[1] < 2:                      # a code has >= 2 characters
            continue
        ys, xs = np.nonzero(sel)
        groups.append((cv2.minAreaRect(np.c_[xs, ys].astype(np.float32)), sel))
    return groups, m


def _tess(img, psm):
    d = pytesseract.image_to_data(img, config=f"--psm {psm} -c tessedit_char_whitelist={CODE_CHARS}",
                                  output_type=pytesseract.Output.DICT)
    words = [(t.strip(), float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip()]
    if not words:
        return "", 0.0
    return "".join(t for t, _ in words), min(c for _, c in words)


def _norm(img, height=80, border=30):
    ys, xs = np.nonzero(img < 128)
    if len(ys) == 0:
        return None
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = height / img.shape[0]
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=255)


def read_code(box, sel):
    """Read one code -> (text, confidence, valid, method, upright_ink_image).
    Characters are read one at a time (Tesseract tends to merge a narrow '11'),
    in both orientations; whole-line readings are used only if they contain
    the same number of characters as were found."""
    (cx, cy), (w, h), ang = box
    if h > w:
        w, h, ang = h, w, ang - 90
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv2.warpAffine((sel * 255).astype(np.uint8), M, sel.shape[1::-1], flags=cv2.INTER_NEAREST)
    pad = int(8 * PPM)
    x0, y0 = int(cx - w / 2) - pad, int(cy - h / 2) - pad
    crop = rot[max(0, y0):y0 + int(h) + 2 * pad, max(0, x0):x0 + int(w) + 2 * pad]
    results = []
    for flip in (False, True):
        c = cv2.rotate(crop, cv2.ROTATE_180) if flip else crop
        n, lab, st, _ = cv2.connectedComponentsWithStats(c)
        spans = sorted([[st[k, 0], st[k, 0] + st[k, 2], [k]] for k in range(1, n) if st[k, 4] > 30])
        chars = []
        for sp in spans:                                # pieces of one character overlap in x
            if chars and sp[0] < chars[-1][1] - 2:
                chars[-1][1] = max(chars[-1][1], sp[1])
                chars[-1][2] += sp[2]
            else:
                chars.append(sp)
        txt, confs = "", []
        for _, _, ks in chars:
            im = _norm(255 - (np.isin(lab, ks) * 255).astype(np.uint8))
            t, cf = _tess(im, 10) if im is not None else ("", 0.0)
            txt += t[:1]
            confs.append(cf if t else 0.0)
        results.append((txt, min(confs) if confs else 0.0, bool(PAT.match(txt)), "per-char", c))
        line = _norm(255 - c)
        if line is not None:
            for psm in (7, 8):
                t, cf = _tess(line, psm)
                results.append((t, cf, bool(PAT.match(t)) and len(t) == len(chars), f"line-psm{psm}", c))
    pool = [r for r in results if r[2] and r[3] == "per-char"] or [r for r in results if r[2]]
    return max(pool or results, key=lambda r: r[1])


# ------------------------------ per image ----------------------------------- #

def analyse(path: Path, bin_result, code_dir: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    corners, qc, _ = bin_result
    if corners is None:
        raise RuntimeError(qc)
    rect, _ = A.rectify(img, corners)
    groups, smask = text_groups(rect)
    roi = A.roi_mask(rect.shape)
    fmask, _, _ = A.fruit_mask(rect, roi)
    codes, tiles = [], []
    # read codes top-to-bottom, left-to-right in the bin
    for box, sel in sorted(groups, key=lambda g: (round(g[0][0][1] / (100 * PPM)), g[0][0][0])):
        txt, conf, valid, how, upright = read_code(box, sel)
        codes.append((txt, conf, valid))
        t = cv2.cvtColor(255 - upright, cv2.COLOR_GRAY2BGR)
        t = cv2.resize(t, (int(t.shape[1] * 120 / t.shape[0]), 120))
        cv2.putText(t, f"{txt} ({conf:.0f})", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if (not valid or conf < MIN_CONF) else (0, 150, 0), 2)
        tiles.append(t)
    code_img = ""
    if tiles:
        h = max(t.shape[0] for t in tiles)
        strip = np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 10, cv2.BORDER_CONSTANT,
                                              value=(200, 200, 200)) for t in tiles])
        code_dir.mkdir(parents=True, exist_ok=True)
        code_img = str(code_dir / f"{path.stem}_code.png")
        cv2.imwrite(code_img, strip)
    small = lambda m: cv2.resize(m.astype(np.float32), None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
    return dict(sheet=bool(smask.any()), codes=codes, code_img=code_img,
                fruit=small(fmask > 0), sheetm=small(smask > 0), roi=small(roi > 0))


def layout_similarity(a, b):
    excl = cv2.dilate(((a["sheetm"] > 0) | (b["sheetm"] > 0)).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    valid = (a["roi"] > 0) & ~excl
    fa, fb = a["fruit"] * valid, b["fruit"] * valid
    union = np.maximum(fa, fb).sum()
    return float(np.minimum(fa, fb).sum() / union) if union > 0 else np.nan


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pair reference-sheet photos with apple photos and read codes")
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=None, help="default: DATA/image_pairs_<raw folder>.csv")
    ap.add_argument("--codes", type=Path, default=None,
                    help="folder for code check images; default: DATA/reference_codes_<raw folder>")
    args = ap.parse_args(argv)
    args.out = args.out or A.default_pairs_csv(args.raw)
    args.codes = args.codes or A.PROJECT_ROOT / "DATA" / f"reference_codes_{args.raw.name}"
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    paths = A.list_images(args.raw)
    times = {p.name: capture_time(p) for p in paths}
    # capture order: EXIF time, then file name (camera counters wrap at 9999)
    paths.sort(key=lambda p: (times[p.name] or datetime.max, p.name))
    print(f"{len(paths)} images. Detecting bins...")
    bins, _ = A.detect_all_bins(paths, {}, verbose=False)

    info = {}
    for p in paths:
        try:
            info[p.name] = analyse(p, bins[p.name], args.codes)
            r = info[p.name]
            tag = "; ".join(f"{t}({c:.0f})" for t, c, _ in r["codes"]) if r["codes"] else (
                "sheet, no code" if r["sheet"] else "-")
            print(f"  {p.name}: {tag}")
        except Exception as e:
            print(f"  {p.name}: FAILED - {e}")

    # a photo is a sheet photo if a white sheet or a code was found in the bin
    is_sheet = {n: r["sheet"] or bool(r["codes"]) for n, r in info.items()}
    rows, used = [], set()
    names = [p.name for p in paths if p.name in info]
    for i, n in enumerate(names):
        if not is_sheet[n]:
            continue
        r = info[n]
        nxt = next((m for m in names[i + 1:] if not is_sheet[m]), None)
        codes = r["codes"]
        flags = []
        if not codes:
            flags.append("no_code_read")
        parsed = [PAT.match(t) for t, _, v in codes if v]
        trees = sorted({m.group(1) for m in parsed}, key=int)
        if len(trees) > 1:
            flags.append("mixed_trees")
        if any((not v) or c < MIN_CONF for _, c, v in codes):
            flags.append("check_code")
        row = dict(filename_1=n, filename_2=nxt or "",
                   reference_code=";".join(t for t, _, _ in codes),
                   tree_id=";".join(trees),
                   crates=";".join(sorted(m.group(2) for m in parsed)),
                   n_codes=len(codes),
                   ocr_confidence=min((c for _, c, _ in codes), default=np.nan),
                   time_1=times[n], time_2=times.get(nxt) if nxt else None,
                   time_gap_s=np.nan, layout_similarity=np.nan, flags="", code_image=r["code_img"])
        if nxt is None:
            flags.append("no_apple_photo")
        else:
            if nxt in used:
                flags.append("apple_photo_shared")
            used.add(nxt)
            if times[n] and times[nxt]:
                row["time_gap_s"] = (times[nxt] - times[n]).total_seconds()
            row["layout_similarity"] = round(layout_similarity(r, info[nxt]), 3)
            if row["layout_similarity"] < MIN_SIMILARITY or row["time_gap_s"] > MAX_GAP_S:
                flags.append("check_pair")
        row["flags"] = ";".join(flags)
        rows.append(row)
    for n in names:
        if not is_sheet[n] and n not in used:
            rows.append(dict(filename_1="", filename_2=n, reference_code="", n_codes=0,
                             time_2=times[n], flags="unpaired_apple_photo"))
    # the same code on two sheet photos: an OCR error or a retaken photo
    seen = {}
    for row in rows:
        for t in filter(None, str(row.get("reference_code") or "").split(";")):
            seen.setdefault(t, []).append(row)
    for t, rs in seen.items():
        if len(rs) > 1:
            for row in rs:
                if "duplicate_code" not in row["flags"]:
                    row["flags"] = ";".join(filter(None, [row["flags"], "duplicate_code"]))
    df = pd.DataFrame(rows, columns=["filename_1", "filename_2", "reference_code", "tree_id", "crates", "n_codes",
                                     "ocr_confidence", "time_1", "time_2", "time_gap_s",
                                     "layout_similarity", "flags", "code_image"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_flag = (df["flags"].fillna("") != "").sum()
    print(f"Wrote {len(df)} rows to {args.out}  ({n_flag} flagged for checking)")


if __name__ == "__main__":
    main()
