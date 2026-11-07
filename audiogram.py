#!/usr/bin/env python3
"""
Extract color-coded air-conduction thresholds from an audiogram.

Expected symbols:
    Right ear air conduction: red O
    Left ear air conduction:  blue X

The script:
1. Separates red and blue markings in HSV color space.
2. Maps graph pixels to frequency and dB HL.
3. Scores red-O and blue-X shapes at each frequency and 5 dB position.
4. Writes:
       measurements.csv
       audiogram_annotated.png
       red_mask.png
       blue_mask.png

Speech OCR and speech-audiometry extraction are intentionally excluded.

Dependencies:
    pip install opencv-python numpy
"""

import csv
import math
from pathlib import Path

import cv2
import numpy as np


# Frequencies to extract.
TARGET_FREQUENCIES = [250, 500, 1000, 2000, 4000, 8000]

FREQUENCY_LABELS = {
    250: "250 Hz (dB HL)",
    500: "500 Hz (dB HL)",
    1000: "1K Hz (dB HL)",
    2000: "2K Hz (dB HL)",
    4000: "4K Hz (dB HL)",
    8000: "8K Hz (dB HL)",
}

# Calibration range of the audiogram plotting area.
GRAPH_MIN_FREQUENCY = 125
GRAPH_MAX_FREQUENCY = 16000
GRAPH_MIN_DB = -10
GRAPH_MAX_DB = 130

# Evaluate only standard 5 dB audiometric increments.
DB_CANDIDATES = list(range(GRAPH_MIN_DB, GRAPH_MAX_DB + 1, 5))


# ============================================================
# USER CONFIGURATION
# ============================================================

CONFIG = {
    # Input audiogram image.
    "image_path": r"C:\Users\natha\Downloads\Screenshot_audiogram.png",

    # Output directory.
    "output_dir": r"C:\Users\natha\Downloads\audiogram_output",

    # Inner audiogram plotting rectangle:
    # (left_x, top_y, right_x, bottom_y)
    #
    # It should correspond to:
    #   left edge:   125 Hz
    #   right edge:  16 kHz
    #   top edge:    -10 dB HL
    #   bottom edge: 130 dB HL
    #
    # Keep as None to select it interactively.
    "plot_rectangle": None,
    # Example:
    # "plot_rectangle": (210, 85, 1450, 930),

    # Approximate symbol radius in pixels.
    # Keep as None to estimate it from the graph dimensions.
    "symbol_radius": None,
    # Example:
    # "symbol_radius": 9,

    # Detection score thresholds.
    # Increase to reduce false positives.
    # Decrease if genuine symbols are missed.
    "red_min_score": 0.16,
    "blue_min_score": 0.16,

    # HSV color ranges. OpenCV hue values range from 0 to 179.
    "red_low_1": (0, 70, 50),
    "red_high_1": (12, 255, 255),
    "red_low_2": (165, 70, 50),
    "red_high_2": (179, 255, 255),
    "blue_low": (85, 60, 40),
    "blue_high": (140, 255, 255),

    # Output options.
    "save_masks": True,
    "save_annotated_image": True,
    "save_csv": True,
}


def select_plot_rectangle(image):
    """
    Interactively select the inner audiogram plotting rectangle.

    Select from the 125 Hz/-10 dB intersection to the
    16 kHz/130 dB intersection.
    """
    window_name = (
        "Select INNER graph: 125 Hz/-10 dB to 16 kHz/130 dB; "
        "then press ENTER"
    )

    x, y, width, height = cv2.selectROI(
        window_name,
        image,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyAllWindows()

    if width <= 0 or height <= 0:
        raise RuntimeError("No audiogram graph rectangle was selected.")

    return (
        int(x),
        int(y),
        int(x + width),
        int(y + height),
    )


def validate_rectangle(rectangle, image_shape):
    """Verify that the selected rectangle is inside the image."""
    x1, y1, x2, y2 = rectangle
    image_height, image_width = image_shape[:2]

    if not (
        0 <= x1 < x2 <= image_width
        and 0 <= y1 < y2 <= image_height
    ):
        raise ValueError(
            f"Plot rectangle {rectangle} is outside image dimensions "
            f"{image_width}x{image_height}."
        )


def separate_colors(image_bgr):
    """
    Create binary masks for red and blue audiogram markings.

    Red requires two HSV ranges because red wraps around the OpenCV
    hue scale.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    red_mask_1 = cv2.inRange(
        hsv,
        np.array(CONFIG["red_low_1"], dtype=np.uint8),
        np.array(CONFIG["red_high_1"], dtype=np.uint8),
    )

    red_mask_2 = cv2.inRange(
        hsv,
        np.array(CONFIG["red_low_2"], dtype=np.uint8),
        np.array(CONFIG["red_high_2"], dtype=np.uint8),
    )

    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    blue_mask = cv2.inRange(
        hsv,
        np.array(CONFIG["blue_low"], dtype=np.uint8),
        np.array(CONFIG["blue_high"], dtype=np.uint8),
    )

    # Suppress isolated color noise while retaining symbol strokes.
    red_mask = cv2.medianBlur(red_mask, 3)
    blue_mask = cv2.medianBlur(blue_mask, 3)

    return red_mask, blue_mask


def frequency_to_x(frequency, plot_rectangle):
    """
    Convert an audiometric frequency to an x-coordinate.

    The audiogram frequency axis is logarithmic in base 2.
    """
    x1, _, x2, _ = plot_rectangle

    fraction = (
        math.log2(frequency / GRAPH_MIN_FREQUENCY)
        / math.log2(GRAPH_MAX_FREQUENCY / GRAPH_MIN_FREQUENCY)
    )

    return x1 + fraction * (x2 - x1)


def db_to_y(db_hl, plot_rectangle):
    """
    Convert dB HL to a y-coordinate.

    Larger dB HL values appear lower on the audiogram.
    """
    _, y1, _, y2 = plot_rectangle

    fraction = (
        (db_hl - GRAPH_MIN_DB)
        / (GRAPH_MAX_DB - GRAPH_MIN_DB)
    )

    return y1 + fraction * (y2 - y1)


def estimate_symbol_radius(plot_rectangle):
    """Estimate the symbol radius from the dimensions of the graph."""
    x1, y1, x2, y2 = plot_rectangle

    graph_width = x2 - x1
    graph_height = y2 - y1

    estimated = round(
        min(
            graph_width / 140.0,
            graph_height / 70.0,
        )
    )

    return max(5, min(estimated, 16))


def extract_patch(mask, center_x, center_y, radius):
    """
    Extract a fixed-size patch around a candidate point.

    Areas outside the source image are padded with zeros.
    """
    size = 2 * radius + 1
    patch = np.zeros((size, size), dtype=np.uint8)

    x1 = center_x - radius
    y1 = center_y - radius
    x2 = center_x + radius + 1
    y2 = center_y + radius + 1

    source_x1 = max(0, x1)
    source_y1 = max(0, y1)
    source_x2 = min(mask.shape[1], x2)
    source_y2 = min(mask.shape[0], y2)

    if source_x2 <= source_x1 or source_y2 <= source_y1:
        return patch

    destination_x1 = source_x1 - x1
    destination_y1 = source_y1 - y1
    destination_x2 = destination_x1 + (source_x2 - source_x1)
    destination_y2 = destination_y1 + (source_y2 - source_y1)

    patch[
        destination_y1:destination_y2,
        destination_x1:destination_x2,
    ] = mask[
        source_y1:source_y2,
        source_x1:source_x2,
    ]

    return patch


def make_local_coordinates(radius):
    """Create normalized coordinates centered on a candidate symbol."""
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    x_coordinates, y_coordinates = np.meshgrid(axis, axis)

    scale = max(radius, 1)

    return (
        x_coordinates / scale,
        y_coordinates / scale,
    )


def score_red_circle(patch):
    """
    Score how closely a binary patch resembles a red O.

    The scoring favors:
    - color in an annular region;
    - low color occupancy in the center;
    - ring coverage across multiple angular sectors.
    """
    binary = patch.astype(np.float32) / 255.0
    radius = patch.shape[0] // 2

    normalized_x, normalized_y = make_local_coordinates(radius)

    radial_distance = np.sqrt(
        normalized_x * normalized_x
        + normalized_y * normalized_y
    )

    annulus = (
        (radial_distance >= 0.45)
        & (radial_distance <= 0.95)
    )

    center = radial_distance <= 0.28

    outside = (
        (radial_distance >= 1.05)
        & (radial_distance <= 1.30)
    )

    annulus_occupancy = (
        float(binary[annulus].mean())
        if np.any(annulus)
        else 0.0
    )

    center_occupancy = (
        float(binary[center].mean())
        if np.any(center)
        else 0.0
    )

    outside_occupancy = (
        float(binary[outside].mean())
        if np.any(outside)
        else 0.0
    )

    angles = np.arctan2(normalized_y, normalized_x)
    sector_hits = []

    for sector in range(8):
        lower_angle = -math.pi + sector * (2 * math.pi / 8)
        upper_angle = lower_angle + (2 * math.pi / 8)

        region = (
            annulus
            & (angles >= lower_angle)
            & (angles < upper_angle)
        )

        if np.any(region):
            sector_hits.append(
                float(binary[region].mean()) > 0.08
            )

    sector_coverage = (
        sum(sector_hits) / len(sector_hits)
        if sector_hits
        else 0.0
    )

    score = (
        0.55 * annulus_occupancy
        + 0.35 * sector_coverage
        - 0.35 * center_occupancy
        - 0.10 * outside_occupancy
    )

    return max(0.0, score)


def score_blue_cross(patch):
    """
    Score how closely a binary patch resembles a blue X.

    Both diagonal stroke directions must be present.
    """
    binary = patch.astype(np.float32) / 255.0
    radius = patch.shape[0] // 2

    normalized_x, normalized_y = make_local_coordinates(radius)

    extent = (
        (np.abs(normalized_x) <= 0.95)
        & (np.abs(normalized_y) <= 0.95)
    )

    band_width = 0.24

    diagonal_1 = (
        extent
        & (np.abs(normalized_y - normalized_x) <= band_width)
    )

    diagonal_2 = (
        extent
        & (np.abs(normalized_y + normalized_x) <= band_width)
    )

    off_diagonal = extent & ~(diagonal_1 | diagonal_2)

    diagonal_1_occupancy = (
        float(binary[diagonal_1].mean())
        if np.any(diagonal_1)
        else 0.0
    )

    diagonal_2_occupancy = (
        float(binary[diagonal_2].mean())
        if np.any(diagonal_2)
        else 0.0
    )

    off_diagonal_occupancy = (
        float(binary[off_diagonal].mean())
        if np.any(off_diagonal)
        else 0.0
    )

    weaker_diagonal = min(
        diagonal_1_occupancy,
        diagonal_2_occupancy,
    )

    average_diagonal = (
        diagonal_1_occupancy
        + diagonal_2_occupancy
    ) / 2.0

    score = (
        0.50 * average_diagonal
        + 0.45 * weaker_diagonal
        - 0.20 * off_diagonal_occupancy
    )

    return max(0.0, score)


def score_candidate(
    mask,
    expected_x,
    expected_y,
    radius,
    symbol_type,
    horizontal_search,
    vertical_search,
):
    """
    Score a candidate while allowing a small positional offset.

    The offset compensates for minor graph-selection and anti-aliasing errors.
    """
    if symbol_type == "O":
        scoring_function = score_red_circle
    elif symbol_type == "X":
        scoring_function = score_blue_cross
    else:
        raise ValueError(f"Unsupported symbol type: {symbol_type}")

    best_score = -1.0
    best_center = None

    for vertical_offset in range(
        -vertical_search,
        vertical_search + 1,
    ):
        for horizontal_offset in range(
            -horizontal_search,
            horizontal_search + 1,
        ):
            center_x = int(
                round(expected_x + horizontal_offset)
            )
            center_y = int(
                round(expected_y + vertical_offset)
            )

            patch = extract_patch(
                mask,
                center_x,
                center_y,
                radius,
            )

            score = scoring_function(patch)

            if score > best_score:
                best_score = score
                best_center = (center_x, center_y)

    return best_score, best_center


def extract_threshold(
    mask,
    frequency,
    plot_rectangle,
    symbol_type,
    radius,
    minimum_score,
):
    """
    Find the strongest O or X candidate at one frequency.

    Returns "Unclear" when:
    - no candidate exceeds the minimum score; or
    - a nonadjacent dB candidate has a nearly identical score.

    Only O and X shapes are targeted. Bone-conduction brackets and common
    masked air-conduction symbols are not deliberately extracted.
    """
    expected_x = frequency_to_x(
        frequency,
        plot_rectangle,
    )

    horizontal_search = max(2, radius // 3)
    vertical_search = max(1, radius // 4)

    candidates = []

    for db_hl in DB_CANDIDATES:
        expected_y = db_to_y(
            db_hl,
            plot_rectangle,
        )

        score, center = score_candidate(
            mask=mask,
            expected_x=expected_x,
            expected_y=expected_y,
            radius=radius,
            symbol_type=symbol_type,
            horizontal_search=horizontal_search,
            vertical_search=vertical_search,
        )

        candidates.append(
            {
                "db": db_hl,
                "score": score,
                "center": center,
            }
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best = candidates[0]

    if best["score"] < minimum_score:
        return {
            "value": "Unclear",
            "score": best["score"],
            "center": best["center"],
            "reason": (
                "No symbol-shaped candidate exceeded "
                "the minimum score."
            ),
        }

    # Ignore adjacent 5 dB competitors because the same symbol may overlap
    # both candidate windows. Check only candidates more than 5 dB away.
    nonadjacent_competitors = [
        candidate
        for candidate in candidates[1:]
        if abs(candidate["db"] - best["db"]) > 5
    ]

    if nonadjacent_competitors:
        strongest_competitor = nonadjacent_competitors[0]

        if strongest_competitor["score"] >= best["score"] * 0.92:
            return {
                "value": "Unclear",
                "score": best["score"],
                "center": best["center"],
                "reason": (
                    "Competing detections at "
                    f"{best['db']} and "
                    f"{strongest_competitor['db']} dB HL."
                ),
            }

    return {
        "value": best["db"],
        "score": best["score"],
        "center": best["center"],
        "reason": "",
    }


def annotate_detection(
    image,
    frequency,
    result,
    color,
    ear_label,
):
    """Draw a detected or uncertain threshold on the verification image."""
    center = result.get("center")

    if center is None:
        return

    value = result["value"]

    if isinstance(value, int):
        annotation_color = color
        label = f"{ear_label} {frequency}: {value} dB"
        thickness = 2
    else:
        # Orange identifies an uncertain candidate.
        annotation_color = (0, 165, 255)
        label = f"{ear_label} {frequency}: unclear"
        thickness = 1

    cv2.circle(
        image,
        center,
        8,
        annotation_color,
        thickness,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        label,
        (center[0] + 10, center[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        annotation_color,
        1,
        cv2.LINE_AA,
    )


def build_result_rows(thresholds):
    """Build output rows for the requested pure-tone frequencies."""
    rows = []

    for frequency in TARGET_FREQUENCIES:
        rows.append(
            [
                FREQUENCY_LABELS[frequency],
                thresholds["right"][frequency]["value"],
                thresholds["left"][frequency]["value"],
            ]
        )

    return rows


def write_csv(output_path, rows):
    """Write pure-tone threshold results to CSV."""
    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["Measurement", "Right Ear", "Left Ear"]
        )
        writer.writerows(rows)


def print_table(rows):
    """Print a dependency-free text table."""
    headers = ["Measurement", "Right Ear", "Left Ear"]

    all_rows = [headers] + [
        [str(value) for value in row]
        for row in rows
    ]

    column_widths = [
        max(len(row[column]) for row in all_rows)
        for column in range(3)
    ]

    def format_row(row):
        return " | ".join(
            str(row[index]).ljust(column_widths[index])
            for index in range(3)
        )

    print()
    print(format_row(headers))
    print(
        "-+-".join(
            "-" * width
            for width in column_widths
        )
    )

    for row in rows:
        print(format_row(row))


def main():
    # --------------------------------------------------------
    # 1. Read the audiogram image
    # --------------------------------------------------------
    image_path = Path(CONFIG["image_path"])

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image does not exist: {image_path.resolve()}"
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"OpenCV could not decode the image: "
            f"{image_path.resolve()}"
        )

    print(
        f"Image dimensions: "
        f"{image.shape[1]} x {image.shape[0]} pixels"
    )

    # --------------------------------------------------------
    # 2. Set or select the graph rectangle
    # --------------------------------------------------------
    plot_rectangle = CONFIG["plot_rectangle"]

    if plot_rectangle is None:
        print(
            "\nSelect the inner audiogram plotting rectangle.\n"
            "Upper-left:  125 Hz, -10 dB HL\n"
            "Lower-right: 16 kHz, 130 dB HL\n"
        )

        plot_rectangle = select_plot_rectangle(image)
    else:
        if len(plot_rectangle) != 4:
            raise ValueError(
                "CONFIG['plot_rectangle'] must contain "
                "(x1, y1, x2, y2)."
            )

        plot_rectangle = tuple(
            int(value)
            for value in plot_rectangle
        )

    validate_rectangle(
        plot_rectangle,
        image.shape,
    )

    # --------------------------------------------------------
    # 3. Separate red and blue markings
    # --------------------------------------------------------
    red_mask, blue_mask = separate_colors(image)

    # Limit symbol searches to the calibrated graph.
    graph_mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    x1, y1, x2, y2 = plot_rectangle
    graph_mask[y1:y2 + 1, x1:x2 + 1] = 255

    red_mask = cv2.bitwise_and(
        red_mask,
        graph_mask,
    )
    blue_mask = cv2.bitwise_and(
        blue_mask,
        graph_mask,
    )

    # --------------------------------------------------------
    # 4. Determine symbol radius
    # --------------------------------------------------------
    if CONFIG["symbol_radius"] is None:
        symbol_radius = estimate_symbol_radius(
            plot_rectangle
        )
    else:
        symbol_radius = int(
            CONFIG["symbol_radius"]
        )

    if symbol_radius < 3:
        raise ValueError(
            "CONFIG['symbol_radius'] must be at least 3 pixels."
        )

    # --------------------------------------------------------
    # 5. Extract pure-tone thresholds
    # --------------------------------------------------------
    thresholds = {
        "right": {},
        "left": {},
    }

    for frequency in TARGET_FREQUENCIES:
        # Right ear: red O.
        thresholds["right"][frequency] = extract_threshold(
            mask=red_mask,
            frequency=frequency,
            plot_rectangle=plot_rectangle,
            symbol_type="O",
            radius=symbol_radius,
            minimum_score=float(
                CONFIG["red_min_score"]
            ),
        )

        # Left ear: blue X.
        thresholds["left"][frequency] = extract_threshold(
            mask=blue_mask,
            frequency=frequency,
            plot_rectangle=plot_rectangle,
            symbol_type="X",
            radius=symbol_radius,
            minimum_score=float(
                CONFIG["blue_min_score"]
            ),
        )

    # --------------------------------------------------------
    # 6. Prepare output directory
    # --------------------------------------------------------
    output_directory = Path(
        CONFIG["output_dir"]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 7. Create annotated verification image
    # --------------------------------------------------------
    annotated = image.copy()

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    for frequency in TARGET_FREQUENCIES:
        annotate_detection(
            image=annotated,
            frequency=frequency,
            result=thresholds["right"][frequency],
            color=(0, 0, 255),
            ear_label="R",
        )

        annotate_detection(
            image=annotated,
            frequency=frequency,
            result=thresholds["left"][frequency],
            color=(255, 0, 0),
            ear_label="L",
        )

    # --------------------------------------------------------
    # 8. Save masks and annotated image
    # --------------------------------------------------------
    if CONFIG["save_masks"]:
        red_mask_path = output_directory / "red_mask.png"
        blue_mask_path = output_directory / "blue_mask.png"

        if not cv2.imwrite(
            str(red_mask_path),
            red_mask,
        ):
            raise RuntimeError(
                f"Could not save {red_mask_path}"
            )

        if not cv2.imwrite(
            str(blue_mask_path),
            blue_mask,
        ):
            raise RuntimeError(
                f"Could not save {blue_mask_path}"
            )

    if CONFIG["save_annotated_image"]:
        annotated_path = (
            output_directory
            / "audiogram_annotated.png"
        )

        if not cv2.imwrite(
            str(annotated_path),
            annotated,
        ):
            raise RuntimeError(
                f"Could not save {annotated_path}"
            )

    # --------------------------------------------------------
    # 9. Build, save, and print pure-tone results
    # --------------------------------------------------------
    rows = build_result_rows(thresholds)

    csv_path = output_directory / "measurements.csv"

    if CONFIG["save_csv"]:
        write_csv(
            output_path=csv_path,
            rows=rows,
        )

    print_table(rows)

    print("\nCalibration")
    print(f"  Plot rectangle: {plot_rectangle}")
    print(f"  Symbol radius:  {symbol_radius} pixels")

    # --------------------------------------------------------
    # 10. Identify measurements requiring manual verification
    # --------------------------------------------------------
    verification_items = []

    for ear in ("right", "left"):
        ear_name = (
            "Right"
            if ear == "right"
            else "Left"
        )

        for frequency in TARGET_FREQUENCIES:
            result = thresholds[ear][frequency]

            if result["value"] == "Unclear":
                verification_items.append(
                    f"{ear_name} ear, {frequency} Hz: "
                    f"{result['reason']} "
                    f"Best score={result['score']:.3f}"
                )

    print("\nMeasurements requiring manual verification")

    if verification_items:
        for item in verification_items:
            print(f"  - {item}")
    else:
        print("  - None flagged by the automated checks.")

    print(
        f"\nResults directory: "
        f"{output_directory.resolve()}"
    )

    if CONFIG["save_csv"]:
        print(
            f"Measurements CSV: "
            f"{csv_path.resolve()}"
        )


if __name__ == "__main__":
    main()
