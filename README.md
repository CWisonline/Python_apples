# Apple bin image analysis (Python)

Counts apples, measures their diameter and quantifies red colour from
top-down photographs of the black-painted single-layer trial bin
(one tree per photograph).

## Scripts

| Script | Purpose |
|---|---|
| `SRC/apple_counter_v2.py` | Main pipeline (bin detection, rectification, apple segmentation, measurement). |
| `SRC/click_bin_corners.py` | Optional: click the 4 rim corners for images where automatic bin detection is doubtful. |
| `SRC/apple_counter.py` | Original Hough-circle version (kept for comparison). |

## Pipeline (v2)

1. **Bin detection, per image.** The four outer rim edges are found by searching
   full-length straight lines. Top and bottom edges are rarely hidden. Left and
   right edges are often under the camera-rig poles, so they are constrained by
   the known aspect ratio (1220 : 1020). If one side cannot be seen, it is placed
   from the other side and the image is flagged (`bin_qc = auto_left_inferred` /
   `auto_right_inferred`). Corners in `DATA/bin_corners.csv` override detection
   (`bin_qc = manual`).
2. **Rectification.** A homography maps the rim to a 1220 × 1020 mm rectangle at
   `RECT_PX_PER_MM` (default 2 px/mm). This corrects position, rotation and
   keystone, and gives the scale from each image's own rim.
3. **Region of interest.** Only the bin interior (rim minus `ROI_MARGIN_MM`) is
   analysed, so poles, grass, cable ties and the next bin cannot be counted.
4. **Fruit segmentation.** CIELAB thresholds (chroma, lightness, hue); the
   painted floor is dark and near-neutral. Touching apples are separated along
   the dark creases between them (black-top-hat + seeded watershed). Fragments
   are then merged and merged pairs re-split using size and shape rules.
5. **Measurements per apple.** Area-equivalent diameter, ellipse major/minor
   diameters (fitted only to the outline that is not shared with a
   neighbouring apple), CIELAB L\*, a\*, b\*, hue angle, chroma, and
   `blush_fraction` (share of skin pixels with hue angle < `BLUSH_HUE_MAX_DEG`).
   `mean_R_channel` is kept only for comparison with the old output.

## Outputs (in `DATA/processed_2025/`)

* `apples_results.csv`: one row per apple.
* `image_summary.csv`: one row per image: count, size and colour summaries,
  bin QC flag, rim corners, image px/mm, and floor colour (a lighting check).
* `<image>_apples.jpg`: rectified image with outlines and numbers; red crosses
  are rejected objects.
* `<image>_bin.jpg`: original image with the detected rim, for checking calibration.

## Data dictionary

All measurements describe the apple as seen from above. Sizes are in
millimetres in the bin's rectified (straightened) coordinates. Colour values
are means over the visible skin.

### `apples_results.csv` (one row per apple)

#### Identification and position

| Column | Definition |
|---|---|
| `filename` | Source image. |
| `n_apples_in_image` | Number of accepted apples in that image (equals the number of rows for the image). |
| `apple_number` | Apple index; matches the number drawn on `<image>_apples.jpg`. |
| `x_mm`, `y_mm` | Position of the apple's centre, in mm from the top-left outer corner of the rim (x to the right, y downwards). |

#### Size

| Column | Definition | Notes |
|---|---|---|
| `diameter_eq_mm` | Diameter of a circle with the same area as the apple's outline, × `PLANE_CORRECTION`. | Main size measure. Least affected by irregular shape or by the cut line where two apples touch. Holes from highlights and the calyx are filled and stalks removed before measuring. |
| `diameter_major_mm` | Long axis of an ellipse fitted to the outline, × `PLANE_CORRECTION`. | Closest equivalent to the maximum width used in grading. |
| `diameter_minor_mm` | Short axis of the same ellipse, × `PLANE_CORRECTION`. | If an apple lies on its side, this may reflect its height rather than its width. |
| `diameter_eq_mm_uncorrected` | As `diameter_eq_mm`, using only the rim-based scale. | Use this to work out `PLANE_CORRECTION` from calliper measurements. Identical to the corrected value while `PLANE_CORRECTION = 1.00`. |
| `diameter_major_mm_uncorrected` | As `diameter_major_mm`, without the correction. | There is no uncorrected minor-axis column. |

The ellipse is normally fitted only to the parts of the outline that border
the floor; cut lines between touching apples follow the creases between fruit
rather than the true edge, so they are left out. If fewer than half of the
outline points border the floor, the whole outline is used instead.

#### Quality checks

| Column | Definition | How to use it |
|---|---|---|
| `solidity` | Outline area ÷ area of its convex hull (0–1; 1 = perfectly convex). | Objects below `MIN_SOLIDITY` (0.86) are rejected. Values near the limit (about 0.86–0.90) suggest an irregular apple, a leftover bit of leaf, or a segmentation error. |
| `free_outline_fraction` | Share of the outline bordering the floor rather than another apple (0–1). | Low values mean the apple is surrounded by neighbours. Below 0.5 the major/minor diameters are less reliable; `diameter_eq_mm` is affected less. |
| `small_or_partial` | `True` if `diameter_eq_mm` < `SMALL_FLAG_MM` (40 mm). | Counted, but excluded from the size summaries in `image_summary.csv`. Cannot distinguish a genuinely small apple from one partly hidden by neighbours; check the overlay where it matters. |

#### Colour (CIELAB)

Colour values are averaged over the inside of each apple. The outer edge
(about 2 mm, shaded) and specular highlights (L\* > 85 and chroma < 15) are
excluded.

| Column | Definition | Interpretation |
|---|---|---|
| `L_star` | Mean lightness, 0 (black) to 100 (white). | Higher = paler; lower = darker skin or a dull/shaded image. |
| `a_star` | Mean position on the green (–) to red (+) axis. | Higher = redder. |
| `b_star` | Mean position on the blue (–) to yellow (+) axis. | Higher = yellower. |
| `hue_angle_deg` | `atan2(b*, a*)` of the mean colour, in degrees. | 0° = red-purple, 90° = yellow, 180° = bluish-green. Lower = redder. McGuire's (1992) recommended colour measure. |
| `chroma` | `√(a*² + b*²)` of the mean colour. | Vividness (0 = grey). On two-coloured apples, averaging red with yellow-green lowers chroma, so treat it with care for strongly blushed fruit. |
| `blush_fraction` | Share of skin pixels (0–1) whose own hue angle is < `BLUSH_HUE_MAX_DEG` (49°), or > 300° where the scale wraps to purple-red. | Blush coverage of the visible, upper side only, not the whole fruit. The 49° cut-off is from Winzeler & Schupp (2011) for 'Honeycrisp'; check it for your cultivar. |
| `mean_R_channel` | Mean red channel value (0–255). | Kept only for comparison with the old output. It mostly reflects brightness, not redness. |

`hue_angle_deg` and `chroma` are calculated from the mean a\* and b\*, not
averaged per pixel, so on a bicoloured apple they describe the blended
colour. `blush_fraction` is the measure that captures the split between red
and non-red skin.

Colour caveats:

* **No colour calibration.** Values come from OpenCV's conversion, which
  assumes standard sRGB camera colour; they are not calibrated to a reference
  card. They are fine for comparing images taken under similar light, but are
  not equivalent to colorimeter readings.
* **Lighting changes** (sun versus shade) shift the values. Use the `floor_*`
  columns in `image_summary.csv` to spot images with unusual lighting.

### `image_summary.csv` (one row per image)

#### Counts and summaries

| Column | Definition |
|---|---|
| `filename` | Source image. |
| `bin_qc` | Result of bin (rim) detection; see the table below. |
| `n_apples` | Number of accepted apples, including `small_or_partial` ones. |
| `n_small_or_partial` | Number of apples flagged `small_or_partial`. |
| `n_rejected` | Objects found but rejected (size, solidity, elongation or hollow), shown as red crosses on the overlay. Mostly pole, cable-tie, grass and leaf fragments. A high value is worth checking on the overlay. |
| `median_diameter_eq_mm`, `mean_diameter_eq_mm` | Median and mean `diameter_eq_mm`, excluding `small_or_partial` apples. |
| `median_diameter_major_mm` | Median `diameter_major_mm`, excluding `small_or_partial` apples. |
| `mean_blush_fraction` | Mean `blush_fraction` over all accepted apples. |
| `plane_correction` | Value of `PLANE_CORRECTION` used for this run. |

#### Calibration and bin geometry

| Column | Definition |
|---|---|
| `img_px_per_mm_x`, `img_px_per_mm_y` | Scale of the original photo at the rim (pixels per mm), across and down the bin, from the detected corners. Should be similar between images from the same rig; a large difference points to a detection error. Measurements themselves are made on the rectified image at `RECT_PX_PER_MM`. |
| `corner_tl_x` … `corner_bl_y` | Detected (or manually clicked) outer rim corners in full-resolution pixels of the original photo: top-left, top-right, bottom-right, bottom-left. Same format as `bin_corners.csv`. |
| `edge_score_top/bottom/left/right` | Strength of the edge evidence for each detected rim side (0–1; higher = clearer edge). 0 for a side that was inferred rather than detected. |
| `lr_width_error_initial` | Fractional difference between the detected left–right width and the width expected from the top–bottom height and the 1220 : 1020 aspect ratio, before any side was inferred. Values beyond ±0.03 trigger inference of the weaker side. |
| `redetected_with_scale_prior` | Present (`True`) only for images re-detected in the second pass, with the rim size constrained to the batch's typical scale. |

#### Lighting check

| Column | Definition |
|---|---|
| `floor_L`, `floor_a`, `floor_b` | Mean CIELAB colour of the visible painted floor inside the bin. The floor should look the same in every photo, so outliers indicate unusual lighting or exposure that may affect the apple colour values. |

#### `bin_qc` values

| Value | Meaning |
|---|---|
| `auto_ok` | All four rim edges detected and consistent with the bin's aspect ratio. |
| `auto_left_inferred` / `auto_right_inferred` | One side was hidden (usually under a pole) and was placed from the other side and the aspect ratio. Normally reliable; worth a glance at `<image>_bin.jpg`. |
| `manual` | Corners taken from `bin_corners.csv`. |
| `…+corner_outside_frame` | At least one corner lies outside the photo (bin cut off at the edge). Check the result; the scale may be less reliable. |
| `…+check_manually` | Scale still differs by more than 3% from the batch median after re-detection. Treat sizes as unreliable until the corners are clicked with `click_bin_corners.py --flagged`. |
| `detect_failed: …` / `failed: …` | The image could not be processed; the reason follows. |

### References

* McGuire, R.G. (1992). Reporting of objective color measurements.
  *HortScience* 27(12): 1254–1255.
* Winzeler, H.E. & Schupp, J.R. (2011). Image analysis of blush coverage
  extent and measures of categorical blush intensity in 'Honeycrisp' apples.
  *HortScience* 46(5): 705–709.

## Calibration: must be done once

The scale comes from the rim, but the apples sit lower, on the floor, so they
are further from the camera and look smaller. Set `PLANE_CORRECTION` from a
calibration photo:

1. Measure 20–30 apples (or discs/balls of known size, ideally about apple
   height) with callipers, and place them spread across the bin.
2. Photograph them with the normal rig and run the script.
3. `PLANE_CORRECTION = mean(calliper diameter) / mean(diameter_eq_mm_uncorrected)`
   (match each object by position, or use the means).

For a stable rig this is one number. It also absorbs any systematic offset
in where the segmented edge sits.

## Recommended for future trials

* Fix four high-contrast markers (e.g. printed ArUco markers) to the rim
  corners. That makes bin detection exact and fully automatic.
* Include a grey/colour reference card in every frame so that colour metrics
  can be normalised for lighting.

## Installation

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

## Running

    python SRC\apple_counter_v2.py
    python SRC\apple_counter_v2.py --raw DATA\raw_2025 --out DATA\processed_2025
    python SRC\click_bin_corners.py --flagged      # optional review of flagged bins
