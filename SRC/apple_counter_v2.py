"""
Apple bin image analysis  (v2)
==============================

For each top-down photograph of the black-painted trial bin:

1. Find the bin's outer rim automatically (or read corners you clicked by hand
   from bin_corners.csv), allowing for changes in position, rotation and slight
   perspective (keystone) between photos.
2. Rectify ("warp") the bin to a true top-down view with a fixed scale
   (RECT_PX_PER_MM), using the known outer rim size (BIN_W_MM x BIN_H_MM).
   Scale is therefore calibrated per image rather than hard-coded.
3. Keep only the bin interior (outer rim minus ROI_MARGIN_MM), so poles,
   grass, cable ties and the neighbouring bin are excluded by construction.
4. Segment fruit by colour (CIELAB chroma, lightness and hue; the painted
   floor is dark and nearly neutral). Touching apples are separated using the
   dark creases between them (black-top-hat) + seeded watershed, then
   fragments are merged and merged pairs re-split using size/shape rules.
   Non-apple objects (leaves, cable ties, pole reflections, floor glare) are
   rejected by size, shape, fill and colour.
5. Per apple: diameters in mm (area-equivalent, ellipse major/minor),
   CIELAB colour, hue angle, and the fraction of skin that is red ("blush").
6. Write a per-apple CSV, a per-image summary CSV (with QC flags), and QC
   overlay images.

IMPORTANT - calibration plane
-----------------------------
The scale comes from the rim, but apples sit on the floor, below the rim and
therefore further from the camera, so they look smaller than they would at rim
height. PLANE_CORRECTION corrects for this. It must be measured once for the
rig (see README / notes): photograph objects of known diameter (e.g. apples
measured with callipers) in the bin and set
    PLANE_CORRECTION = mean(true diameter) / mean(diameter_*_mm_uncorrected)
This factor also absorbs any small systematic bias in where the segmented
outline sits relative to the true fruit edge. Until it is measured, absolute
diameters are biased (probably low) by an unknown few per cent; relative
comparisons between trees remain valid while the rig is unchanged.

Requirements: numpy, pandas, opencv-python, scipy, scikit-image
Usage:  python SRC/apple_counter_v2.py [--raw DIR] [--out DIR] [--corners CSV]
"""

from __future__ import annotations

import argparse
import re
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# =============================== USER SETTINGS =============================== #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "DATA" / "raw_2025"
OUT_DIR = PROJECT_ROOT / "DATA" / "processed_2025"
CORNERS_CSV = PROJECT_ROOT / "DATA" / "bin_corners.csv"   # optional manual corners
# Optional reference-code pairs from pair_reference_sheets.py: by default
# DATA/image_pairs_<raw folder>.csv (named per folder because camera file
# numbers repeat between seasons).


# Optional harvest record (crate weights per tree), used with the pairs file to
# assign every crate to a photo and compute mean fruit mass. Expected columns:
# a tree column, optionally a plot column, and one weight column per crate
# whose name contains the crate letter, e.g. "Fruit_Weight_A_(kg)".
HARVEST_FILE = None          # e.g. PROJECT_ROOT / "DATA" / "2026.09.05_plumford_harvest_assessment.xlsx"
HARVEST_SHEET = 0            # sheet name or index
HARVEST_TREE_COL = "Tree"
HARVEST_PLOT_COL = "Plot"    # set to None if there is no plot column
HARVEST_WEIGHT_COL_REGEX = r"Weight_([A-Z])"   # captures the crate letter
# Flag photos/trees whose mean fruit mass differs from the batch median by more
# than this factor (e.g. a crate assumed to be in a photo that is really elsewhere)
MASS_RATIO_LIMIT = 1.5


def default_pairs_csv(raw_dir: Path) -> Path:
    return PROJECT_ROOT / "DATA" / f"image_pairs_{Path(raw_dir).name}.csv"

# Bin outer rim dimensions (mm). Width = the long side (left-right in photos).
BIN_W_MM = 1220.0
BIN_H_MM = 1020.0

# Resolution of the rectified (top-down) image used for all measurements.
RECT_PX_PER_MM = 2.0

# Ignore this many mm inside the outer rim (rim top + poles lying on it).
ROI_MARGIN_MM = 40.0

# Scale correction from the rim plane to the apple-equator plane (see above).
PLANE_CORRECTION = 1.00

# Fruit colour segmentation (CIELAB, L 0-100, a/b approx -128..127).
FRUIT_MIN_CHROMA = 16.0     # painted floor is near-neutral (chroma < ~15)
FRUIT_BRIGHT_L = 68.0       # pale cheeks/highlights: low chroma but brighter than the floor
FRUIT_BRIGHT_MIN_B = 8.0    # ...and yellowish (sunlit floor glare is bluish/neutral)
FRUIT_MIN_L = 20.0
FRUIT_DARK_L = 50.0           # below this L*, ...
FRUIT_DARK_CHROMA_SLOPE = 0.8  # ...required chroma rises by this per L* unit
FRUIT_HUE_MAX_DEG = 135.0     # fruit hue angle window (green skin reaches ~125 deg)
FRUIT_HUE_MIN_WRAP_DEG = 300.0  # ...wrapping round through magenta-red

# Accepted apple size and shape (after PLANE_CORRECTION).
MIN_DIAM_MM = 25.0            # smaller objects rejected
SMALL_FLAG_MM = 40.0          # apples below this are counted but flagged (often partly hidden)
MAX_DIAM_MM = 110.0
MIN_SOLIDITY = 0.86
MIN_FILL_FRACTION = 0.75      # fruit pixels / filled outline (rejects hollow loops)
STALK_OPEN_MM = 8.0           # opening used to strip stalks/leaf tips before measuring
MIN_AXIS_RATIO = 0.60          # minor/major of fitted ellipse
OVERSIZE_FACTOR = 1.40         # regions > this x median diameter are tested as 2+ apples
KMEANS_SEED = 0                # fixed seed so repeated runs give identical results

# Watershed seeds: minimum spacing between apple centres (mm).
MIN_CENTRE_SPACING_MM = 28.0
MIN_SEED_RADIUS_MM = 10.0      # core distance-transform peak must exceed this
CREASE_WIDTH_MM = 10.0         # structuring element for the dark creases between fruit
CREASE_THRESH_L = 4.0          # L* drop marking a crease (lower = more splitting)

# Red blush: skin pixels with CIELAB hue angle (h_ab) below this are counted as
# blush. 0 deg = red-purple, 90 deg = yellow, 180 deg = bluish-green (McGuire 1992,
# HortScience 27:1254-1255). 49 deg is the blush/non-blush boundary found for
# 'Honeycrisp' by Winzeler & Schupp (2011, HortScience 46:705-709); check it
# against visual scores for your cultivar and camera.
BLUSH_HUE_MAX_DEG = 49.0

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ======================= BIN (RIM) DETECTION ================================ #

_WORK = 0.25      # detection runs on a 1/4-size copy
_OFF = 4          # px either side of a candidate edge
_BAND = 14        # px band used for paint/not-paint context
_NS = 140         # samples along each candidate line
_ASPECT_TOL = 0.03
SCALE_OUTLIER_TOL = 0.03   # flag/re-detect bins whose scale differs from the batch median by >3%


def _paint_fraction(sm: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    m = ((b <= 3) & (a < 8) & (b > -30) & (L < 150)).astype(np.float32)
    return cv2.blur(m, (_BAND, _BAND))


def _sample(img, x, y):
    h, w = img.shape
    xi = np.clip(np.round(x).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(y).astype(np.int32), 0, h - 1)
    return img[yi, xi]


def _best_edge(gray, pin, side, lo, hi, tilt, use_context, t0=0.08, t1=0.92, return_all=False):
    """Search all straight lines for one side of the rim.

    A line is given by its position a at t0 and b at t1 along the side.
    Score = mean along the line of (outside brighter than inside), optionally
    weighted by 'inside looks like painted bin, outside does not'.
    """
    H, W = gray.shape
    a = np.arange(lo, hi, dtype=float)
    d = np.arange(-tilt, tilt + 1, dtype=float)
    A = np.repeat(a, len(d))
    B = A + np.tile(d, len(a))
    t = np.linspace(t0, t1, _NS)[None, :]
    pos = A[:, None] + (B - A)[:, None] * (t - t0) / (t1 - t0)
    horiz = side in ("top", "bottom")
    along = np.broadcast_to(t * ((W if horiz else H) - 1), pos.shape)
    sgn = -1.0 if side in ("top", "left") else 1.0

    def S(img, off):
        q = pos + sgn * off
        return _sample(img, along, q) if horiz else _sample(img, q, along)

    s = np.clip(S(gray, _OFF) - S(gray, -_OFF), 0, 80) / 80.0
    if use_context:
        s = s * np.clip(S(pin, -(_OFF + _BAND / 2)) - S(pin, _OFF + _BAND / 2), 0, 1)
    score = s.mean(1)
    if return_all:
        return A, B, score
    k = int(np.argmax(score))
    return A[k], B[k], float(score[k])


def _pts(side, a, b, W, H, t0=0.08, t1=0.92):
    if side in ("top", "bottom"):
        return np.array([t0 * (W - 1), a]), np.array([t1 * (W - 1), b])
    return np.array([a, t0 * (H - 1)]), np.array([b, t1 * (H - 1)])


def _intersect(p1, p2, q1, q2):
    r, s = p2 - p1, q2 - q1
    t = np.linalg.solve(np.array([r, -s]).T, q1 - p1)
    return p1 + t[0] * r


def detect_bin_corners(img_bgr: np.ndarray, expected_px_per_mm: float | None = None):
    """Return (corners[4x2] TL,TR,BR,BL in full-res px, qc_flag, info dict).

    If expected_px_per_mm is given (e.g. the batch median), the rim edges are
    constrained to give a bin of about that size - used to re-detect images
    where shadows or clutter fooled the unconstrained search."""
    sm = cv2.resize(img_bgr, None, fx=_WORK, fy=_WORK, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    pin = _paint_fraction(sm)
    H, W = gray.shape

    e = {}
    # Top and bottom outer edges are long and rarely occluded.
    if expected_px_per_mm is None:
        e["top"] = _best_edge(gray, pin, "top", 0, int(0.40 * H), int(0.04 * W), False)
        e["bottom"] = _best_edge(gray, pin, "bottom", int(0.60 * H), H, int(0.04 * W), False)
    else:
        # choose the best top/bottom PAIR whose separation matches the expected bin height
        h_exp = expected_px_per_mm * BIN_H_MM * _WORK
        cand = {}
        for side, lo, hi in (("top", 0, int(0.45 * H)), ("bottom", int(0.55 * H), H)):
            A, B, sc = _best_edge(gray, pin, side, lo, hi, int(0.04 * W), False, return_all=True)
            keep = np.argsort(sc)[::-1][:1500]
            cand[side] = (A[keep], B[keep], sc[keep])
        At, Bt, St = cand["top"]
        Ab, Bb, Sb = cand["bottom"]
        sep = 0.5 * ((Ab + Bb)[None, :] - (At + Bt)[:, None])
        tot = np.where(np.abs(sep / h_exp - 1) < 0.04, St[:, None] + Sb[None, :], -1)
        i, j = np.unravel_index(np.argmax(tot), tot.shape)
        if tot[i, j] < 0:
            raise RuntimeError("no rim edge pair matches the expected scale")
        e["top"] = (At[i], Bt[i], float(St[i]))
        e["bottom"] = (Ab[j], Bb[j], float(Sb[j]))
    # Left/right edges are often hidden under the camera-rig poles.
    e["left"] = _best_edge(gray, pin, "left", 0, int(0.40 * W), int(0.05 * H), True)
    e["right"] = _best_edge(gray, pin, "right", int(0.60 * W), W, int(0.05 * H), True)

    h_mid = 0.5 * ((e["bottom"][0] + e["bottom"][1]) - (e["top"][0] + e["top"][1]))
    w_exp = h_mid * BIN_W_MM / BIN_H_MM
    xl = 0.5 * (e["left"][0] + e["left"][1])
    xr = 0.5 * (e["right"][0] + e["right"][1])
    width_err = (xr - xl - w_exp) / w_exp
    qc = "auto_ok"
    if abs(width_err) > _ASPECT_TOL:
        # Keep the better-supported side, place the other at the expected width.
        if e["left"][2] >= e["right"][2]:
            e["right"] = (e["left"][0] + w_exp, e["left"][1] + w_exp, 0.0)
            qc = "auto_right_inferred"
        else:
            e["left"] = (e["right"][0] - w_exp, e["right"][1] - w_exp, 0.0)
            qc = "auto_left_inferred"
    T = _pts("top", e["top"][0], e["top"][1], W, H)
    B = _pts("bottom", e["bottom"][0], e["bottom"][1], W, H)
    L = _pts("left", e["left"][0], e["left"][1], W, H)
    R = _pts("right", e["right"][0], e["right"][1], W, H)
    corners = np.array([_intersect(*T, *L), _intersect(*T, *R),
                        _intersect(*B, *R), _intersect(*B, *L)]) / _WORK
    h0, w0 = img_bgr.shape[:2]
    tol = 0.005 * w0
    if ((corners[:, 0] < -tol) | (corners[:, 0] > w0 + tol) |
            (corners[:, 1] < -tol) | (corners[:, 1] > h0 + tol)).any():
        qc += "+corner_outside_frame"
    info = {f"edge_score_{k}": round(v[2], 3) for k, v in e.items()}
    info["lr_width_error_initial"] = round(float(width_err), 4)
    return corners, qc, info


def load_pairs(path: Path):
    """From image_pairs.csv: ({apple photo: (reference_code, pair_flags, sheet photo)},
    set of sheet photos). Sheet photos are skipped by the counter."""
    if not path or not Path(path).exists():
        return {}, set()
    df = pd.read_csv(path, dtype=str).fillna("")
    if "tree_id" not in df:
        df["tree_id"] = ""
    codes = {r.filename_2: (r.reference_code, r.flags, r.filename_1, r.tree_id)
             for r in df.itertuples() if r.filename_2}
    sheets = {f for f in df.filename_1 if f}
    return codes, sheets


def load_manual_corners(path: Path) -> dict:
    """bin_corners.csv columns: filename,tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y
    (full-resolution pixel coordinates)."""
    out = {}
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                out[r["filename"]] = np.array([[float(r["tl_x"]), float(r["tl_y"])],
                                               [float(r["tr_x"]), float(r["tr_y"])],
                                               [float(r["br_x"]), float(r["br_y"])],
                                               [float(r["bl_x"]), float(r["bl_y"])]])
    return out


# ============================ RECTIFICATION ================================= #

def rectify(img_bgr, corners):
    w = int(round(BIN_W_MM * RECT_PX_PER_MM))
    h = int(round(BIN_H_MM * RECT_PX_PER_MM))
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    Hm = cv2.getPerspectiveTransform(np.float32(corners), dst)
    rect = cv2.warpPerspective(img_bgr, Hm, (w, h), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_CONSTANT)
    return rect, Hm


def image_px_per_mm(corners):
    """Mean original-image scale at the rim (for reporting / comparison only)."""
    c = np.asarray(corners)
    w = 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3]))
    h = 0.5 * (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1]))
    return w / BIN_W_MM, h / BIN_H_MM


# ============================ APPLE DETECTION =============================== #

def to_lab(bgr):
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    return lab[..., 0], lab[..., 1], lab[..., 2]


def roi_mask(shape):
    m = np.zeros(shape[:2], np.uint8)
    g = int(round(ROI_MARGIN_MM * RECT_PX_PER_MM))
    m[g:shape[0] - g, g:shape[1] - g] = 255
    return m


def fruit_mask(rect_bgr, roi):
    L, a, b = to_lab(rect_bgr)
    chroma = np.hypot(a, b)
    hue = (np.degrees(np.arctan2(b, a)) + 360) % 360
    fruit_hue = (hue <= FRUIT_HUE_MAX_DEG) | (hue >= FRUIT_HUE_MIN_WRAP_DEG)   # red..yellow-green, not blue
    # darker pixels must be more strongly coloured (rejects shadows that pick up
    # reflected colour between apples, keeps dark-red skin)
    min_chroma = FRUIT_MIN_CHROMA + FRUIT_DARK_CHROMA_SLOPE * np.clip(FRUIT_DARK_L - L, 0, None)
    m = ((((chroma > min_chroma) & (L > FRUIT_MIN_L)) & fruit_hue)
         | ((L > FRUIT_BRIGHT_L) & (b > FRUIT_BRIGHT_MIN_B))).astype(np.uint8) * 255
    m &= roi
    raw = m.copy()   # before any filling; used to reject hollow objects (e.g. cable-tie loops)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    # fill small holes (stalk cavities, specular highlights) but not gaps between apples
    holes = ndi.binary_fill_holes(m > 0) & ~(m > 0)
    lab_h, n = ndi.label(holes)
    if n:
        sizes = ndi.sum(holes, lab_h, index=np.arange(1, n + 1))
        max_hole = (0.15 * np.pi * (SMALL_FLAG_MM / 2) ** 2) * RECT_PX_PER_MM ** 2
        small = np.isin(lab_h, np.where(sizes < max_hole)[0] + 1)
        m[small] = 255
    return m, (L, a, b), raw


def split_apples(mask, L):
    """Separate touching apples.

    The narrow dark creases between touching fruit are picked out with a
    morphological black-top-hat of lightness. Removing them from the fruit
    mask leaves one 'core' per apple; distance-transform peaks of the cores are
    used as seeds, and a watershed on the crease map recovers full outlines.
    """
    Ls = cv2.GaussianBlur(L, (0, 0), 1.5)
    k = int(CREASE_WIDTH_MM * RECT_PX_PER_MM) | 1
    crease = cv2.morphologyEx(Ls, cv2.MORPH_BLACKHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    core = ((mask > 0) & (crease < CREASE_THRESH_L)).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # dark calyx/stalk marks punch holes in the cores; fill them
    core = ndi.binary_fill_holes(core).astype(np.uint8)
    core_lab, n_core = ndi.label(core)
    dist = cv2.distanceTransform(core, cv2.DIST_L2, 5)
    dist_s = cv2.GaussianBlur(dist, (0, 0), 2.0)
    peaks = peak_local_max(dist_s,
                           min_distance=int(MIN_CENTRE_SPACING_MM * RECT_PX_PER_MM),
                           threshold_abs=MIN_SEED_RADIUS_MM * RECT_PX_PER_MM,
                           labels=core_lab, exclude_border=False)
    peaks = [tuple(p) for p in peaks]
    # every core of apple-like size gets at least one seed
    seeded = {core_lab[r, c] for r, c in peaks}
    min_core_area = np.pi * (0.5 * MIN_DIAM_MM / 2) ** 2 * RECT_PX_PER_MM ** 2
    for i, sl in enumerate(ndi.find_objects(core_lab), start=1):
        if sl is None or i in seeded:
            continue
        sub = np.where(core_lab[sl] == i, dist_s[sl], 0)
        if (core_lab[sl] == i).sum() >= min_core_area:
            r, c = np.unravel_index(np.argmax(sub), sub.shape)
            peaks.append((r + sl[0].start, c + sl[1].start))
    markers = np.zeros(mask.shape, np.int32)
    for i, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = i
    # flood from seeds over (crease strength - distance) so boundaries follow creases
    relief = crease / max(CREASE_THRESH_L, 1e-6) - dist_s / (dist_s.max() + 1e-6)
    labels = watershed(relief, markers, mask=mask > 0)
    # fruit blobs that received no seed (e.g. a dark apple next to pale ones) get their own label
    comp, n = ndi.label((mask > 0) & (labels == 0))
    min_area = np.pi * (MIN_DIAM_MM / 2 / PLANE_CORRECTION) ** 2 * RECT_PX_PER_MM ** 2
    nxt = labels.max() + 1
    for i in range(1, n + 1):
        sel = comp == i
        if sel.sum() >= min_area:
            labels[sel] = nxt
            nxt += 1
    return labels, dist


def _boxes(labels):
    """{label: [y0, y1, x0, x1]} bounding boxes for all labels, in one pass."""
    return {i: [sl[0].start, sl[0].stop, sl[1].start, sl[1].stop]
            for i, sl in enumerate(ndi.find_objects(labels), start=1) if sl is not None}


def _pad(box, pad, shape):
    """Padded slices for a [y0, y1, x0, x1] box, clipped to the image."""
    y0, y1, x0, x1 = box
    return (slice(max(0, y0 - pad), min(shape[0], y1 + pad)),
            slice(max(0, x0 - pad), min(shape[1], x1 + pad)))


def _clean_region(region):
    """Fill internal holes (pale/specular skin) and remove stalks and thin
    attachments (opening with a STALK_OPEN_MM disc)."""
    region = ndi.binary_fill_holes(region > 0).astype(np.uint8)
    k = int(STALK_OPEN_MM * RECT_PX_PER_MM) | 1
    o = cv2.morphologyEx(region, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if not o.any():
        return region
    n, lab, st, _ = cv2.connectedComponentsWithStats(o, 8)
    return (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)


def _shape_stats(region):
    """(ok, reason, d_eq_mm_corrected, solidity, axis_ratio) for a small binary crop."""
    region = _clean_region(region)
    area = int(region.sum())
    d_eq = 2 * np.sqrt(area / np.pi) / RECT_PX_PER_MM * PLANE_CORRECTION

    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts or area == 0:
        return False, "empty", d_eq, 0.0, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(cnt))
    sol = cv2.contourArea(cnt) / hull if hull > 0 else 0.0
    ratio = 1.0
    if len(cnt) >= 5:
        (_, _), (a1, a2), _ = cv2.fitEllipse(cnt)
        ratio = min(a1, a2) / max(a1, a2, 1e-6)
    if not (MIN_DIAM_MM <= d_eq <= MAX_DIAM_MM):
        return False, "size", d_eq, sol, ratio
    if sol < MIN_SOLIDITY:
        return False, "solidity", d_eq, sol, ratio
    if ratio < MIN_AXIS_RATIO:
        return False, "elongated", d_eq, sol, ratio
    return True, None, d_eq, sol, ratio


def merge_fragments(labels):
    """Merge watershed fragments that are parts of one apple: a region failing
    the size/shape rules is merged with the touching neighbour for which the
    union passes the rules (choosing the most convex union)."""
    labels = labels.copy()
    k3 = np.ones((3, 3), np.uint8)
    boxes = _boxes(labels)            # kept up to date as regions merge
    changed = True
    while changed:
        changed = False
        for lab_id in sorted(boxes):
            if lab_id not in boxes:   # merged away earlier in this pass
                continue
            sl = _pad(boxes[lab_id], 8, labels.shape)
            sub = labels[sl]
            reg = (sub == lab_id).astype(np.uint8)
            if _shape_stats(reg)[0]:
                continue
            ring = cv2.dilate(reg, k3, iterations=3).astype(bool) & ~reg.astype(bool)
            best = None
            for n in np.unique(sub[ring]):
                if n in (0, lab_id):
                    continue
                b1, b2 = boxes[lab_id], boxes[n]
                ub = [min(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), max(b1[3], b2[3])]
                sl2 = _pad(ub, 4, labels.shape)
                union = np.isin(labels[sl2], [lab_id, n]).astype(np.uint8)
                union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, k3)
                ok, _, _, sol, _ = _shape_stats(union)
                if ok and (best is None or sol > best[0]):
                    best = (sol, n, sl2, union)
            if best is not None:
                _, n, sl2, union = best
                view = labels[sl2]
                view[view == n] = lab_id
                view[(union > 0) & (view == 0)] = lab_id
                ys, xs = np.nonzero(view == lab_id)
                boxes[lab_id] = [sl2[0].start + ys.min(), sl2[0].start + ys.max() + 1,
                                 sl2[1].start + xs.min(), sl2[1].start + xs.max() + 1]
                del boxes[n]
                changed = True
    return labels


def split_oversized(labels):
    """Re-split regions that are probably 2+ touching apples: elongated
    regions, or regions much larger than the typical apple in this image
    (d_eq > OVERSIZE_FACTOR x median). Seeds come from k-means on pixel
    coordinates, with k estimated from the area. The split is kept only if
    every part passes the size/shape rules."""
    labels = labels.copy()
    stats = {}
    for i, box in _boxes(labels).items():
        sl = _pad(box, 6, labels.shape)
        stats[i] = (sl, _shape_stats((labels[sl] == i).astype(np.uint8)))
    good_d = [st[2] for _, st in stats.values() if st[0]]
    if not good_d:
        return labels
    med = float(np.median(good_d))
    nxt = labels.max() + 1
    for i, (sl, (ok, reason, d, sol, ratio)) in stats.items():
        oversized = d > OVERSIZE_FACTOR * med
        if not (oversized or (not ok and ratio < MIN_AXIS_RATIO)):
            continue
        reg = (labels[sl] == i).astype(np.uint8)
        k_est = int(np.clip(round((d / med) ** 2), 2, 4))
        ys, xs = np.nonzero(reg)
        pts = np.c_[xs, ys].astype(np.float32)
        best = None
        for k in sorted({k_est, 2}):
            cv2.setRNGSeed(KMEANS_SEED)   # reproducible results between runs
            _, lab_k, ctr = cv2.kmeans(pts, k, None,
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5),
                                       5, cv2.KMEANS_PP_CENTERS)
            dist = cv2.distanceTransform(reg, cv2.DIST_L2, 5)
            mk = np.zeros(reg.shape, np.int32)
            for j, (cx, cy) in enumerate(ctr, start=1):
                cy_i, cx_i = int(np.clip(cy, 0, reg.shape[0] - 1)), int(np.clip(cx, 0, reg.shape[1] - 1))
                if reg[cy_i, cx_i]:
                    mk[cy_i, cx_i] = j
            if len(np.unique(mk)) - 1 < k:
                continue
            sub = watershed(-dist, mk, mask=reg > 0)
            if all(_shape_stats((sub == j).astype(np.uint8))[0] for j in range(1, k + 1)):
                best = (k, sub)
                break
        if best is not None:
            k, sub = best
            view = labels[sl]
            for j in range(2, k + 1):
                view[sub == j] = nxt
                nxt += 1
    return labels


def fill_label_holes(labels):
    """Give enclosed holes (pale skin, highlights) to the surrounding apple."""
    labels = labels.copy()
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        view = labels[sl]
        filled = ndi.binary_fill_holes(view == i)
        view[filled & (view == 0)] = i
    return labels


def _ellipse_from_free_boundary(region, sub_labels, lab_id):
    """Fit an ellipse to the part of the outline that borders background
    (not a neighbouring apple), so cut lines between touching apples do not
    distort the fit. Works on a crop."""
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, 0.0
    cnt = max(cnts, key=cv2.contourArea)[:, 0, :]
    k5 = np.ones((5, 5), np.uint8)
    near_bg = cv2.dilate((sub_labels == 0).astype(np.uint8), k5) > 0
    near_other = cv2.dilate(((sub_labels != 0) & (sub_labels != lab_id)).astype(np.uint8), k5) > 0
    free_px = near_bg & ~near_other
    free = cnt[free_px[cnt[:, 1], cnt[:, 0]]]
    frac_free = len(free) / max(len(cnt), 1)
    pts = (free if (len(free) >= 20 and frac_free >= 0.5) else cnt).astype(np.float32)
    if len(pts) < 5:
        return None, frac_free
    return cv2.fitEllipse(pts), frac_free


def measure_apples(labels, lab_img, rect_bgr, raw_mask):
    L, a, b = lab_img
    rows, rejected = [], []
    ppm = RECT_PX_PER_MM
    objs = ndi.find_objects(labels)
    H, W = labels.shape
    for lab_id, sl0 in enumerate(objs, start=1):
        if sl0 is None:
            continue
        sl = (slice(max(0, sl0[0].start - 6), min(H, sl0[0].stop + 6)),
              slice(max(0, sl0[1].start - 6), min(W, sl0[1].stop + 6)))
        y0, x0 = sl[0].start, sl[1].start
        sub_lab = labels[sl]
        raw = (sub_lab == lab_id).astype(np.uint8)
        ok, reason, d_eq_c, solidity, ratio = _shape_stats(raw)
        region = _clean_region(raw)
        ys, xs = np.nonzero(region)
        cx, cy = xs.mean() + x0, ys.mean() + y0
        if ok and (raw_mask[sl][region > 0] > 0).mean() < MIN_FILL_FRACTION:
            ok, reason = False, "hollow"
        if not ok:
            rejected.append(dict(cx_px=cx, cy_px=cy, area_px=int(region.sum()), reason=reason))
            continue
        d_eq = d_eq_c / PLANE_CORRECTION
        ell, frac_free = _ellipse_from_free_boundary(region, sub_lab, lab_id)
        if ell is not None:
            (_, _), (ax1, ax2), _ = ell
            d_maj, d_min = max(ax1, ax2) / ppm, min(ax1, ax2) / ppm
        else:
            d_maj = d_min = d_eq
        # colour on an eroded core (avoid dark edge/shadow pixels), minus specular highlights
        Ls, as_, bs = L[sl], a[sl], b[sl]
        chroma = np.hypot(as_, bs)
        core = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        sel = (core > 0) & ~((Ls > 85) & (chroma < 15))
        if sel.sum() < 50:
            sel = region > 0
        Lm, am, bm = float(Ls[sel].mean()), float(as_[sel].mean()), float(bs[sel].mean())
        hue = (np.degrees(np.arctan2(bs[sel], as_[sel])) + 360) % 360
        R = rect_bgr[sl][..., 2].astype(np.float32)
        rows.append(dict(
            x_mm=cx / ppm, y_mm=cy / ppm,
            diameter_eq_mm=d_eq_c, diameter_major_mm=d_maj * PLANE_CORRECTION,
            diameter_minor_mm=d_min * PLANE_CORRECTION,
            diameter_eq_mm_uncorrected=d_eq, diameter_major_mm_uncorrected=d_maj,
            solidity=solidity, free_outline_fraction=frac_free,
            small_or_partial=bool(d_eq_c < SMALL_FLAG_MM),
            L_star=Lm, a_star=am, b_star=bm,
            hue_angle_deg=float((np.degrees(np.arctan2(bm, am)) + 360) % 360),
            chroma=float(np.hypot(am, bm)),
            blush_fraction=float(((hue < BLUSH_HUE_MAX_DEG) | (hue > 300)).mean()),
            mean_R_channel=float(R[sel].mean()),
            _label=lab_id,
        ))
    return rows, rejected


# ================================ OUTPUT ==================================== #

def draw_overlays(img_bgr, corners, Hm, rect_bgr, labels, rows, rejected, qc, title):
    ppm = RECT_PX_PER_MM
    over = rect_bgr.copy()
    bnd = (cv2.morphologyEx(labels.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)
    over[bnd & (labels > 0)] = (0, 255, 0)
    g = int(round(ROI_MARGIN_MM * ppm))
    cv2.rectangle(over, (g, g), (over.shape[1] - g, over.shape[0] - g), (255, 255, 0), 2)
    for i, r in enumerate(rows, start=1):
        c = (int(r["x_mm"] * ppm), int(r["y_mm"] * ppm))
        cv2.circle(over, c, 3, (0, 0, 255), -1)
        cv2.putText(over, str(i), (c[0] + 4, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for r in rejected:
        c = (int(r["cx_px"]), int(r["cy_px"]))
        cv2.drawMarker(over, c, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 3)
    cv2.putText(over, f"{title}  n={len(rows)}  rejected={len(rejected)}  {qc}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    orig = img_bgr.copy()
    cv2.polylines(orig, [np.round(corners).astype(np.int32)], True, (0, 255, 255), 8)
    for p in corners:
        cv2.circle(orig, tuple(np.round(p).astype(int)), 25, (0, 0, 255), 6)
    cv2.putText(orig, qc, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)
    return over, orig


def list_images(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
                  and not p.stem.endswith("_processed"))


def detect_all_bins(paths, manual: dict, verbose=True):
    """Two-pass bin detection for a batch.

    Pass 1: unconstrained detection. The batch's typical scale is the median
    px/mm of clean detections (auto_ok). Pass 2: images whose scale deviates by
    more than SCALE_OUTLIER_TOL are re-detected with the rim size constrained to
    that typical scale. Anything still deviating is flagged 'check_manually'."""
    res = {}
    for p in paths:
        if p.name in manual:
            res[p.name] = (manual[p.name], "manual", {})
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        try:
            res[p.name] = detect_bin_corners(img)
        except Exception as e:
            res[p.name] = (None, f"detect_failed: {e}", {})
    scale = {k: float(np.mean(image_px_per_mm(v[0]))) for k, v in res.items() if v[0] is not None}
    clean = [scale[k] for k, v in res.items() if v[1] in ("auto_ok", "manual")]
    if not clean:
        return res, None
    typical = float(np.median(clean))
    for p in paths:
        c, qc, info = res[p.name]
        if qc == "manual":
            continue
        if c is not None and abs(scale[p.name] / typical - 1) <= SCALE_OUTLIER_TOL:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        try:
            c2, qc2, info2 = detect_bin_corners(img, expected_px_per_mm=typical)
            info2["redetected_with_scale_prior"] = True
            res[p.name] = (c2, qc2, info2)
            c, qc = c2, qc2
        except Exception:
            pass
        if c is None or abs(float(np.mean(image_px_per_mm(c))) / typical - 1) > SCALE_OUTLIER_TOL:
            res[p.name] = (c, qc + "+check_manually", info)
        if verbose:
            print(f"  re-detected {p.name}: {res[p.name][1]}")
    return res, typical


def process_image(path: Path, bin_result, out_dir: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read {path}")
    corners, qc, info = bin_result
    if corners is None:
        raise RuntimeError(qc)
    rect, Hm = rectify(img, corners)
    roi = roi_mask(rect.shape)
    mask, lab_img, raw_mask = fruit_mask(rect, roi)
    labels, _ = split_apples(mask, lab_img[0])
    labels = split_oversized(labels)
    labels = merge_fragments(labels)
    labels = fill_label_holes(labels)
    rows, rejected = measure_apples(labels, lab_img, rect, raw_mask)

    over, orig = draw_overlays(img, corners, Hm, rect, labels, rows, rejected, qc, path.name)
    cv2.imwrite(str(out_dir / f"{path.stem}_apples.jpg"), over, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(out_dir / f"{path.stem}_bin.jpg"),
                cv2.resize(orig, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))

    # floor colour (QC of lighting): pixels in ROI that are not fruit
    L, a, b = lab_img
    floor = (roi > 0) & (mask == 0)
    sx, sy = image_px_per_mm(corners)
    full = [r for r in rows if not r["small_or_partial"]]   # size stats exclude partly hidden fruit
    summary = dict(
        filename=path.name, bin_qc=qc, n_apples=len(rows),
        n_small_or_partial=len(rows) - len(full), n_rejected=len(rejected),
        median_diameter_eq_mm=float(np.median([r["diameter_eq_mm"] for r in full])) if full else np.nan,
        mean_diameter_eq_mm=float(np.mean([r["diameter_eq_mm"] for r in full])) if full else np.nan,
        median_diameter_major_mm=float(np.median([r["diameter_major_mm"] for r in full])) if full else np.nan,
        mean_blush_fraction=float(np.mean([r["blush_fraction"] for r in rows])) if rows else np.nan,
        img_px_per_mm_x=sx, img_px_per_mm_y=sy,
        floor_L=float(L[floor].mean()), floor_a=float(a[floor].mean()), floor_b=float(b[floor].mean()),
        plane_correction=PLANE_CORRECTION,
        **{f"corner_{n}_{ax}": float(corners[i][j]) for i, n in enumerate(("tl", "tr", "br", "bl"))
           for j, ax in enumerate("xy")},
        **info,
    )
    for i, r in enumerate(rows, start=1):
        r.pop("_label")
        r.update(filename=path.name, apple_number=i, n_apples_in_image=len(rows))
    return rows, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--corners", type=Path, default=CORNERS_CSV)
    ap.add_argument("--harvest", type=Path, default=HARVEST_FILE,
                    help="harvest record (xlsx/csv) with crate weights per tree")
    ap.add_argument("--pairs", type=Path, default=None,
                    help="pairs CSV from pair_reference_sheets.py; default "
                         "DATA/image_pairs_<raw folder>.csv, used if it exists")
    args = ap.parse_args(argv)
    args.pairs = args.pairs or default_pairs_csv(args.raw)
    args.out.mkdir(parents=True, exist_ok=True)

    manual = load_manual_corners(args.corners)
    pair_codes, sheet_photos = load_pairs(args.pairs)
    paths = [p for p in list_images(args.raw) if p.name not in sheet_photos]
    print(f"Found {len(paths)} images in {args.raw}  (manual corners for {len(manual)}"
          + (f"; {len(sheet_photos)} reference-sheet photos skipped" if sheet_photos else "") + ")")
    print("Detecting bins...")
    bins, typical = detect_all_bins(paths, manual)
    if typical:
        print(f"Typical rim scale in this batch: {typical:.3f} px/mm")
    all_rows, summaries = [], []
    for p in paths:
        try:
            rows, s = process_image(p, bins[p.name], args.out)
        except Exception as e:  # keep going; record the failure
            print(f"  {p.name}: FAILED - {e}")
            summaries.append(dict(filename=p.name, bin_qc=f"failed: {e}"))
            continue
        print(f"  {p.name}: {s['n_apples']} apples ({s['n_rejected']} rejected), "
              f"median d={s['median_diameter_eq_mm']:.1f} mm, bin={s['bin_qc']}")
        code, pflags, sheet_photo, tree = pair_codes.get(p.name, ("", "", "", ""))
        for r in rows:
            r["reference_code"] = code
            r["tree_id"] = tree
        s = {"filename": s.pop("filename"), "tree_id": tree, "reference_code": code,
             "sheet_photo": sheet_photo, "pair_flags": pflags, **s}
        all_rows.extend(rows)
        summaries.append(s)

    cols = ["filename", "tree_id", "reference_code", "n_apples_in_image", "apple_number", "x_mm", "y_mm",
            "diameter_eq_mm", "diameter_major_mm", "diameter_minor_mm",
            "diameter_eq_mm_uncorrected", "diameter_major_mm_uncorrected",
            "solidity", "free_outline_fraction", "small_or_partial", "L_star", "a_star", "b_star",
            "hue_angle_deg", "chroma", "blush_fraction", "mean_R_channel"]
    pd.DataFrame(all_rows, columns=cols).to_csv(args.out / "apples_results.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.out / "image_summary.csv", index=False)
    print(f"Wrote {len(all_rows)} apples to {args.out / 'apples_results.csv'}")
    if pair_codes:
        apples_df, images_df = pd.DataFrame(all_rows, columns=cols), pd.DataFrame(summaries)
        if args.harvest:
            record = load_harvest(args.harvest)
            crates_df, images_df, ts = reconcile_harvest(record, images_df, apples_df)
            crates_df.to_csv(args.out / "crate_assignment.csv", index=False)
            images_df.to_csv(args.out / "image_summary.csv", index=False)   # adds crate/mass columns
            print(f"Wrote {len(crates_df)} crates to {args.out / 'crate_assignment.csv'}")
        else:
            ts = tree_summary(apples_df, images_df)
        ts.to_csv(args.out / "tree_summary.csv", index=False)
        print(f"Wrote {len(ts)} trees to {args.out / 'tree_summary.csv'}")


def load_harvest(path: Path) -> pd.DataFrame:
    """Harvest record -> long table: tree_id, plot, crate, weight_kg (one row per crate)."""
    path = Path(path)
    df = (pd.read_excel(path, sheet_name=HARVEST_SHEET) if path.suffix.lower() in (".xlsx", ".xls")
          else pd.read_csv(path))
    wcols = {c: m.group(1) for c in df.columns
             if (m := re.search(HARVEST_WEIGHT_COL_REGEX, str(c)))}
    if not wcols:
        raise ValueError(f"No crate weight columns matching {HARVEST_WEIGHT_COL_REGEX!r} in {path}")
    rows = []
    for _, r in df.iterrows():
        t = r[HARVEST_TREE_COL]
        tree = str(int(t)) if isinstance(t, (int, float)) and float(t).is_integer() else str(t).strip()
        for col, crate in wcols.items():
            if pd.notna(r[col]):
                rows.append(dict(tree_id=tree, plot=(r[HARVEST_PLOT_COL] if HARVEST_PLOT_COL else ""),
                                 crate=crate, weight_kg=float(r[col])))
    return pd.DataFrame(rows)


def reconcile_harvest(record: pd.DataFrame, images: pd.DataFrame, apples: pd.DataFrame):
    """Assign every crate in the harvest record to an apple photo.

    * A crate whose label was read on a sheet photo is assigned to the paired
      apple photo ("labelled").
    * A crate whose label was not read anywhere is assumed to have been tipped
      into the tree's photo, if the tree has exactly one photo ("assumed_merged").
      If the tree has several photos the crate cannot be placed ("ambiguous");
      if it has none, it is "not_photographed".
    * Codes read from sheets but absent from the record are reported
      ("not_in_record": a misread label or an incomplete record).
    The crate weights in each photo give its mean fruit mass (kg / apples),
    a check on the count and on the assumptions above.

    Returns (crate_assignment, images with crate/mass columns, tree_summary).
    """
    images = images.copy()
    photos = images[images["reference_code"].fillna("") != ""]
    labelled = {}                                    # (tree, crate) -> [photos]
    for _, im in photos.iterrows():
        for code in str(im["reference_code"]).split(";"):
            m = re.match(r"^(\d+)([A-Za-z]+)$", code.strip())
            if m:
                labelled.setdefault((m.group(1), m.group(2).upper()), []).append(im["filename"])
    rec_keys = set(zip(record["tree_id"], record["crate"]))
    tree_photos = {}
    for (t, _c), fs in labelled.items():
        tree_photos.setdefault(t, set()).update(fs)

    rows = []
    for _, r in record.iterrows():
        key = (r["tree_id"], r["crate"])
        fs = labelled.get(key, [])
        tp = sorted(tree_photos.get(r["tree_id"], []))
        if fs:
            status, photo = ("labelled" if len(fs) == 1 else "labelled_in_several_photos"), fs[0]
        elif len(tp) == 1:
            status, photo = "assumed_merged", tp[0]
        elif len(tp) > 1:
            status, photo = "ambiguous", ""
        else:
            status, photo = "not_photographed", ""
        rows.append(dict(tree_id=r["tree_id"], plot=r["plot"], crate=r["crate"],
                         code=f'{r["tree_id"]}{r["crate"]}', weight_kg=r["weight_kg"],
                         photo=photo, status=status))
    plots = dict(zip(record["tree_id"], record["plot"]))
    for (t, c), fs in labelled.items():
        if (t, c) not in rec_keys:
            rows.append(dict(tree_id=t, plot=plots.get(t, ""), crate=c, code=f"{t}{c}", weight_kg=np.nan,
                             photo=fs[0], status="not_in_record"))
    crates = pd.DataFrame(rows)

    # per photo: crates, weight and mean fruit mass
    assigned = crates[crates["photo"] != ""]
    per_photo = assigned.groupby("photo").agg(
        crates_assigned=("code", lambda x: ";".join(sorted(x))),
        crate_weight_kg=("weight_kg", lambda x: x.sum(min_count=1)),
        crate_status=("status", lambda x: ";".join(sorted(set(x)))))
    images = images.drop(columns=[c for c in per_photo.columns if c in images], errors="ignore")
    images = images.merge(per_photo, left_on="filename", right_index=True, how="left")
    images["mean_fruit_mass_g"] = 1000 * images["crate_weight_kg"] / images["n_apples"].where(images["n_apples"] > 0)
    # reference: photos whose crates were all identified from labels (no assumptions)
    trusted = images.loc[images["crate_status"] == "labelled", "mean_fruit_mass_g"].dropna()
    med = trusted.median() if len(trusted) else images["mean_fruit_mass_g"].median()
    images["mass_ratio_to_median"] = images["mean_fruit_mass_g"] / med
    ratio = images["mean_fruit_mass_g"] / med
    images["mass_check"] = np.where((ratio > MASS_RATIO_LIMIT) | (ratio < 1 / MASS_RATIO_LIMIT), "check_mass", "")

    # per tree: every tree in the record, plus trees only seen on labels
    out = []
    for tree in list(dict.fromkeys(list(record["tree_id"]) + sorted({t for t, _ in labelled}))):
        cr = crates[crates["tree_id"] == tree]
        ims = images[images["filename"].isin(set(cr["photo"]) - {""})]
        a = apples[apples["filename"].isin(ims["filename"])]
        full = a[~a["small_or_partial"].astype(bool)] if len(a) else a
        w_rec = cr.loc[cr["status"] != "not_in_record", "weight_kg"].sum()
        w_ph = cr.loc[cr["photo"] != "", "weight_kg"].sum(min_count=1)
        n = len(a)
        flags = set()
        for st in cr["status"]:
            if st not in ("labelled", "assumed_merged"):
                flags.add(st)
        if "assumed_merged" in set(cr["status"]):
            flags.add("assumed_merged")
        for pf in ims.get("pair_flags", pd.Series(dtype=str)).fillna(""):
            flags.update(f for f in str(pf).split(";") if f)
        for q in ims.get("bin_qc", pd.Series(dtype=str)):
            if q not in ("auto_ok", "manual"):
                flags.add(f"bin:{q}")
        mass = 1000 * w_ph / n if n and pd.notna(w_ph) else np.nan
        if pd.notna(mass) and (mass / med > MASS_RATIO_LIMIT or mass / med < 1 / MASS_RATIO_LIMIT):
            flags.add("check_mass")
        out.append(dict(
            plot=plots.get(tree, ""), tree_id=tree,
            crates_in_record=";".join(cr.loc[cr["status"] != "not_in_record", "code"]),
            crates_labelled=";".join(cr.loc[cr["status"].str.startswith("labelled") |
                                            (cr["status"] == "not_in_record"), "code"]),
            crates_assumed=";".join(cr.loc[cr["status"] == "assumed_merged", "code"]),
            n_photos=len(ims), photos=";".join(ims["filename"]),
            total_weight_kg=w_rec if w_rec else np.nan,
            weight_photographed_kg=w_ph,
            n_apples=n if n else np.nan,
            n_small_or_partial=int(a["small_or_partial"].astype(bool).sum()) if n else np.nan,
            mean_fruit_mass_g=mass,
            median_diameter_eq_mm=full["diameter_eq_mm"].median() if n else np.nan,
            mean_diameter_eq_mm=full["diameter_eq_mm"].mean() if n else np.nan,
            sd_diameter_eq_mm=full["diameter_eq_mm"].std() if n else np.nan,
            median_diameter_major_mm=full["diameter_major_mm"].median() if n else np.nan,
            mean_blush_fraction=a["blush_fraction"].mean() if n else np.nan,
            mean_hue_angle_deg=a["hue_angle_deg"].mean() if n else np.nan,
            flags=";".join(sorted(flags)),
        ))
    return crates, images, pd.DataFrame(out)


def tree_summary(apples: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """One row per tree, combining all of its crates and photos.
    Size statistics exclude small_or_partial apples (as in image_summary)."""
    apples = apples[apples["tree_id"].fillna("") != ""]
    images = images[images.get("tree_id", pd.Series(dtype=str)).fillna("") != ""]
    out = []
    for tree, a in apples.groupby("tree_id", sort=False):
        im = images[images["tree_id"] == tree]
        full = a[~a["small_or_partial"].astype(bool)]
        codes = sorted({c for rc in im["reference_code"] for c in str(rc).split(";") if c})
        flags = sorted({f for pf in im["pair_flags"].fillna("") for f in str(pf).split(";") if f}
                       | {f"bin:{q}" for q in im["bin_qc"] if q not in ("auto_ok", "manual")})
        out.append(dict(
            tree_id=tree,
            crates=";".join(codes),
            n_photos=len(im),
            photos=";".join(im["filename"]),
            n_apples=len(a),
            n_small_or_partial=int(a["small_or_partial"].astype(bool).sum()),
            median_diameter_eq_mm=full["diameter_eq_mm"].median(),
            mean_diameter_eq_mm=full["diameter_eq_mm"].mean(),
            sd_diameter_eq_mm=full["diameter_eq_mm"].std(),
            median_diameter_major_mm=full["diameter_major_mm"].median(),
            mean_blush_fraction=a["blush_fraction"].mean(),
            mean_hue_angle_deg=a["hue_angle_deg"].mean(),
            flags=";".join(flags),
        ))
    ts = pd.DataFrame(out)
    if len(ts):
        # a tree id that is not purely numeric (e.g. "3;20" from mixed_trees) sorts last
        ts["_k"] = pd.to_numeric(ts["tree_id"], errors="coerce")
        ts = ts.sort_values(["_k", "tree_id"]).drop(columns="_k")
    return ts


if __name__ == "__main__":
    main()
