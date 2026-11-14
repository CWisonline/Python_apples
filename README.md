# Apple bin image analysis (Python)

Counts apples, measures their diameter and quantifies red colour from
top-down photographs of the black-painted single-layer trial bin
(one tree per photograph).

## Scripts

| Script | Purpose |
|---|---|
| `SRC/apple_counter_v2.py` | Main pipeline (bin detection, rectification, apple segmentation, measurement). |
| `SRC/pair_reference_sheets.py` | Reads the reference code on the white sheet in each sheet photo (OCR) and pairs it with its apple photo. Run before the counter. |
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

## Reference-sheet pairing

Each bin is photographed twice: first with a white sheet showing the printed
reference code of each crate tipped into it, then without. A code is the tree
number plus a crate letter: `20A` and `20B` are two crates from tree 20. One
photo may hold several crates of the same tree, and a tree's crates may be
spread over several photos; `tree_summary.csv` combines them.
`pair_reference_sheets.py`:

1. finds the bin and any white sheet(s) in it, including sheets partly in shade;
2. finds the printed characters, groups them into codes, and reads each code
   with Tesseract OCR (characters read one at a time, in both orientations,
   and checked against `CODE_PATTERN`, default one to three digits + `A`/`B`;
   to allow more crate letters, add them to both `CODE_PATTERN` and
   `CODE_CHARS`);
3. pairs each sheet photo with the next photo by capture time (EXIF) that
   has no sheet, and checks the pair by comparing the fruit layout;
4. writes `DATA/image_pairs_<raw folder>.csv` and, in
   `DATA/reference_codes_<raw folder>/`, an image of
   every code as read (green = confident, red = check), for fast checking by eye.

If `DATA/image_pairs_<raw folder>.csv` exists, `apple_counter_v2.py` skips
the sheet photos and adds `reference_code` to its outputs. The file is named
after the raw folder because camera file numbers repeat between seasons
(DSCN9889 exists in both 2025 and 2026).

The sheet is **not used for colour/brightness calibration**. The camera ran
on automatic exposure and white balance, so the sheet photo and the apple
photo are exposed differently (the floor is 30–50% darker in the sheet photos
of the examples). The paper is also partly clipped (blue and green at or near
255) and, like most office paper, bluish from optical brighteners. See
"Recommended for future trials".

## Outputs (in `DATA/processed_2025/`)

* `apples_results.csv`: one row per apple.
* `image_summary.csv`: one row per image: count, size and colour summaries,
  bin QC flag, rim corners, image px/mm, and floor colour (a lighting check).
* `<image>_apples.jpg`: rectified image with outlines and numbers; red crosses
  are rejected objects.
* `<image>_bin.jpg`: original image with the detected rim, for checking calibration.
* `tree_summary.csv`: one row per tree, combining all its crates and photos
  (written only when a pairs file is used; with a harvest record it lists
  every tree in the record, including trees with no photo).
* `crate_assignment.csv`: with a harvest record, one row per crate showing
  which photo it was assigned to and how (see "Harvest record").

## Harvest record (crate weights)

With `--harvest <file>` (or `HARVEST_FILE` set in the script), the counter
reads the harvest record (one row per tree, a weight column per crate, e.g.
`Fruit_Weight_A_(kg)`, `Fruit_Weight_B_(kg)`) and assigns every crate to a photo:

| Status | Rule |
|---|---|
| `labelled` | The crate's code was read on a sheet photo; assigned to the paired apple photo. |
| `labelled_in_several_photos` | The code was read on more than one sheet photo; the first is used. Check (see `duplicate_code`). |
| `assumed_merged` | The code was not read anywhere, and the tree has exactly one photo: the crate is assumed to have been tipped in with the tree's other crate. |
| `ambiguous` | The code was not read, and the tree has several photos, so the crate cannot be placed. |
| `not_photographed` | No code of this tree was read at all. |
| `not_in_record` | A code was read that is not in the record (a misread label, or a gap in the record). |

**Run this on the complete dataset.** `assumed_merged` relies on the
crate's own photo not existing; on a subset the photo may simply be missing.

Each photo's crate weight divided by its apple count gives the mean fruit
mass. In the examples, photos whose crates were all identified from labels
gave 98–125 g per apple, while wrongly assuming a second crate gave about
250 g. Photos (and trees) whose mean fruit mass differs from the median of
the label-only photos by more than `MASS_RATIO_LIMIT` (1.5×) are flagged
`check_mass`. This catches wrong merge assumptions, missed or extra
photos, and gross counting errors.

Settings at the top of `apple_counter_v2.py`: `HARVEST_FILE`,
`HARVEST_SHEET`, `HARVEST_TREE_COL`, `HARVEST_PLOT_COL`,
`HARVEST_WEIGHT_COL_REGEX` (captures the crate letter from the weight
column names) and `MASS_RATIO_LIMIT`.

## Data dictionary

All measurements describe the apple as seen from above. Sizes are in
millimetres in the bin's rectified (straightened) coordinates. Colour values
are means over the visible skin.

### `apples_results.csv` (one row per apple)

#### Identification and position

| Column | Definition |
|---|---|
| `filename` | Source image. |
| `tree_id` | Tree number from the reference code(s) (empty if no pairs file or no paired sheet photo). |
| `reference_code` | Crate code(s) from the pairs file, e.g. `20B;20A`. |
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
| `tree_id` | Tree number from the reference code(s). |
| `reference_code` | Crate code(s) from the pairs file; several are joined by `;`. |
| `sheet_photo` | The paired reference-sheet photo. |
| `pair_flags` | `flags` from the pairs file (empty = no problems found). |
| `crates_assigned` | With a harvest record: crate codes assigned to this photo (labelled and assumed). |
| `crate_weight_kg` | With a harvest record: total weight of those crates. |
| `crate_status` | With a harvest record: how the crates were assigned (`labelled`, `assumed_merged`, `not_in_record`). |
| `mean_fruit_mass_g` | `crate_weight_kg` × 1000 ÷ `n_apples`. |
| `mass_ratio_to_median` | `mean_fruit_mass_g` ÷ the median of photos whose crates were all identified from labels. |
| `mass_check` | `check_mass` if the ratio is above `MASS_RATIO_LIMIT` or below its inverse. |
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

### `image_pairs_<raw folder>.csv` (one row per sheet photo, plus unpaired apple photos)

| Column | Definition |
|---|---|
| `filename_1` | Photo with the reference sheet. Empty for an apple photo with no sheet photo. |
| `filename_2` | Paired apple photo: the next photo by capture time with no sheet. |
| `reference_code` | Code(s) read from the sheet(s); several codes are joined by `;`. |
| `tree_id` | Tree number(s) from the code(s). More than one only if `mixed_trees`. |
| `crates` | Crate letter(s) from the code(s), joined by `;`. |
| `n_codes` | Number of codes found on the sheet photo. |
| `ocr_confidence` | Lowest Tesseract confidence (0–100) among the codes. Below `MIN_CONF` (70) the code is flagged. |
| `time_1`, `time_2` | Capture times (EXIF `DateTimeOriginal`). |
| `time_gap_s` | Seconds between the two photos. |
| `layout_similarity` | Overlap (0–1) of the fruit layout in the two photos, ignoring the sheet area. True pairs scored 0.55–0.91 and unrelated photos 0.22–0.33 in the examples. |
| `flags` | Problems to check (below); empty = none found. |
| `code_image` | Path to the image of the code(s) as read. |

| Flag | Meaning |
|---|---|
| `check_code` | A code does not match `CODE_PATTERN` or was read with confidence below `MIN_CONF`. Check the code image. |
| `mixed_trees` | Codes from more than one tree in one photo. Several crates of the same tree (e.g. `20B;20A`) are normal and not flagged. |
| `duplicate_code` | The same code was read on more than one sheet photo: a misread, or a retaken sheet photo. |
| `no_code_read` | A sheet was found but no code could be read. |
| `check_pair` | `layout_similarity` below `MIN_SIMILARITY` (0.45) or photos more than `MAX_GAP_S` (120 s) apart. The fruit may have been moved, or a photo may be missing. |
| `no_apple_photo` | No later photo without a sheet. |
| `apple_photo_shared` | The same apple photo follows two sheet photos (e.g. a retaken sheet photo). |
| `unpaired_apple_photo` | An apple photo with no sheet photo before it. |

To correct a code or a pairing, edit the pairs file before running the
counter; the counter only reads `filename_1`, `filename_2`,
`reference_code`, `tree_id` and `flags`. If you correct a code, correct
`tree_id` to match.

### `tree_summary.csv` (one row per tree)

Combines every photo and crate of a tree. Size statistics exclude
`small_or_partial` apples, as in `image_summary.csv`.

With a harvest record, the table lists every tree in the record and adds:

| Column | Definition |
|---|---|
| `plot` | Plot from the harvest record. |
| `crates_in_record` | Crate codes in the harvest record. |
| `crates_labelled` | Codes read on sheet photos (including any not in the record). |
| `crates_assumed` | Codes assigned by the `assumed_merged` rule. |
| `total_weight_kg` | Total crate weight in the record. |
| `weight_photographed_kg` | Weight of the crates assigned to photos. |
| `mean_fruit_mass_g` | `weight_photographed_kg` × 1000 ÷ `n_apples`. |

`flags` then also includes crate statuses other than `labelled`
(`assumed_merged`, `ambiguous`, `not_photographed`, `not_in_record`,
`labelled_in_several_photos`) and `check_mass`.

Without a harvest record, the table contains these columns:

| Column | Definition |
|---|---|
| `tree_id` | Tree number. |
| `crates` | Crate codes found for the tree, e.g. `20A;20B`. Check that all expected crates are present. |
| `n_photos`, `photos` | Number and names of the apple photos for the tree. |
| `n_apples` | Total apples across the tree's photos, including `small_or_partial`. |
| `n_small_or_partial` | Apples flagged `small_or_partial`. |
| `median_diameter_eq_mm`, `mean_diameter_eq_mm`, `sd_diameter_eq_mm` | Median, mean and standard deviation of `diameter_eq_mm` over all the tree's apples. |
| `median_diameter_major_mm` | Median `diameter_major_mm`. |
| `mean_blush_fraction` | Mean `blush_fraction` over all the tree's apples. |
| `mean_hue_angle_deg` | Mean `hue_angle_deg` over all the tree's apples. |
| `flags` | Pairing flags of the tree's photos, plus `bin:<bin_qc>` for any photo whose bin was not `auto_ok` or `manual`. |

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
* For colour calibration: set the camera (COOLPIX P510) to manual exposure
  and a fixed white-balance preset for each session, and fix a matte
  mid-grey card (or a small colour checker) permanently to the rim, in view
  in every apple photo. A white sheet in a separate photo cannot do this job
  (see "Reference-sheet pairing").

## Installation

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

`pair_reference_sheets.py` also needs the Tesseract OCR program. On Windows,
install it from https://github.com/UB-Mannheim/tesseract/wiki (the default
English data is sufficient). If it is not on the PATH, set `TESSERACT_CMD` in
the script, e.g. `r"C:\Program Files\Tesseract-OCR\tesseract.exe"`.

## Running

    python SRC\pair_reference_sheets.py --raw DATA\raw_2026      # 1. read codes, pair photos
    #    check DATA\image_pairs_raw_2026.csv (flags) and DATA\reference_codes_raw_2026\*.png
    python SRC\apple_counter_v2.py --raw DATA\raw_2026 --out DATA\processed_2026 ^
        --harvest DATA\2026.09.05_plumford_harvest_assessment.xlsx              # 2. count
    python SRC\click_bin_corners.py --flagged      # optional review of flagged bins
