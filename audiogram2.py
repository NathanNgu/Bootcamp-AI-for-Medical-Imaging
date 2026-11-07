#!/usr/bin/env python3
"""
Extract air-conduction and bone-conduction thresholds from a
color-coded audiogram.

Standard symbols supported
--------------------------

Right ear, red:
    O          Unmasked air conduction
    Triangle   Masked air conduction
    <          Unmasked bone conduction
    [          Masked bone conduction

Left ear, blue:
    X          Unmasked air conduction
    Square     Masked air conduction
    >          Unmasked bone conduction
    ]          Masked bone conduction

BC fallback rule
----------------
The fallback is applied separately to each ear:

1. If no accepted BC symbol is detected anywhere for an ear:
       BC = AC at all frequencies where AC was detected.

2. If one or more accepted BC symbols are detected for an ear:
       BC is taken only from detected BC symbols.
       Missing BC frequencies are reported as "Not recorded".

Values copied from AC are labeled "Assumed equal to AC" so they are
not confused with measured BC thresholds.

Outputs
-------
    measurements.csv
    audiogram_annotated.png
    red_mask.png
    blue_mask.png

Dependencies
------------
    pip install opencv-python numpy
"""

import csv
import math
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# AUDIOGRAM CALIBRATION
# ============================================================

TARGET_FREQUENCIES = [
    250,
    500,
    1000,
    2000,
    4000,
    8000,
]

FREQUENCY_LABELS = {
    250: "250 Hz",
    500: "500 Hz",
    1000: "1K Hz",
    2000: "2K Hz",
    4000: "4K Hz",
    8000: "8K Hz",
}

GRAPH_MIN_FREQUENCY = 125
GRAPH_MAX_FREQUENCY = 16000

GRAPH_MIN_DB = -10
GRAPH_MAX_DB = 130

DB_CANDIDATES = list(
    range(
        GRAPH_MIN_DB,
        GRAPH_MAX_DB + 1,
        5,
    )
)


# ============================================================
# SYMBOL DEFINITIONS
# ============================================================

# Internal symbol names are used instead of Unicode characters
# to avoid font and encoding problems in OpenCV annotations.

SYMBOL_DISPLAY_NAMES = {
    "O": "O",
    "X": "X",
    "TRIANGLE": "Triangle",
    "SQUARE": "Square",
    "LESS_THAN": "<",
    "GREATER_THAN": ">",
    "LEFT_BRACKET": "[",
    "RIGHT_BRACKET": "]",
}

# Standard color-coded conventions.
EAR_SYMBOLS = {
    "right": {
        "ac": ["O", "TRIANGLE"],
        "bc": ["LESS_THAN", "LEFT_BRACKET"],
    },
    "left": {
        "ac": ["X", "SQUARE"],
        "bc": ["GREATER_THAN", "RIGHT_BRACKET"],
    },
}


# ============================================================
# USER CONFIGURATION
# ============================================================

CONFIG = {
    # Input audiogram image.
    "image_path": (
        r"C:\Users\natha\Downloads\Screenshot_audiogram.png"
    ),

    # Output directory.
    "output_dir": (
        r"C:\Users\natha\Downloads\audiogram_output"
    ),

    # Inner plotting rectangle:
    #     (left_x, top_y, right_x, bottom_y)
    #
    # The selected rectangle should correspond to:
    #     left edge:   125 Hz
    #     right edge:  16 kHz
    #     top edge:    -10 dB HL
    #     bottom edge: 130 dB HL
    #
    # Keep as None for interactive selection.
    "plot_rectangle": None,

    # Example:
    # "plot_rectangle": (210, 85, 1450, 930),

    # Approximate radius of each symbol in pixels.
    # Keep as None to estimate it from the graph dimensions.
    "symbol_radius": None,

    # Example:
    # "symbol_radius": 9,

    # Minimum template-matching scores.
    #
    # Increase these values to reduce false positives.
    # Decrease them if genuine symbols are missed.
    "ac_min_score": 0.46,
    "bc_min_score": 0.46,

    # A second candidate with a score close to the best candidate
    # can cause the result to be marked Unclear.
    "ambiguity_ratio": 0.94,

    # Color segmentation ranges in OpenCV HSV coordinates.
    #
    # OpenCV hue:
    #     0-179
    #
    # Saturation/value:
    #     0-255
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


# ============================================================
# GRAPH SELECTION AND CALIBRATION
# ============================================================

def select_plot_rectangle(image):
    """
    Interactively select the inner audiogram graph.
    """
    message = (
        "Select INNER graph: 125 Hz/-10 dB to "
        "16 kHz/130 dB; then press ENTER"
    )

    x, y, width, height = cv2.selectROI(
        message,
        image,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyAllWindows()

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "No audiogram graph rectangle was selected."
        )

    return (
        int(x),
        int(y),
        int(x + width),
        int(y + height),
    )


def validate_rectangle(rectangle, image_shape):
    """
    Confirm that the graph rectangle lies inside the image.
    """
    x1, y1, x2, y2 = rectangle
    image_height, image_width = image_shape[:2]

    if not (
        0 <= x1 < x2 <= image_width
        and 0 <= y1 < y2 <= image_height
    ):
        raise ValueError(
            f"Plot rectangle {rectangle} is outside image "
            f"dimensions {image_width}x{image_height}."
        )


def frequency_to_x(frequency, plot_rectangle):
    """
    Convert frequency to an x-coordinate using the logarithmic
    audiogram frequency axis.
    """
    x1, _, x2, _ = plot_rectangle

    fraction = (
        math.log2(
            frequency / GRAPH_MIN_FREQUENCY
        )
        / math.log2(
            GRAPH_MAX_FREQUENCY
            / GRAPH_MIN_FREQUENCY
        )
    )

    return x1 + fraction * (x2 - x1)


def db_to_y(db_hl, plot_rectangle):
    """
    Convert dB HL to a y-coordinate.
    """
    _, y1, _, y2 = plot_rectangle

    fraction = (
        (db_hl - GRAPH_MIN_DB)
        / (GRAPH_MAX_DB - GRAPH_MIN_DB)
    )

    return y1 + fraction * (y2 - y1)


def estimate_symbol_radius(plot_rectangle):
    """
    Estimate the audiogram symbol radius from graph dimensions.
    """
    x1, y1, x2, y2 = plot_rectangle

    graph_width = x2 - x1
    graph_height = y2 - y1

    estimated = round(
        min(
            graph_width / 140.0,
            graph_height / 70.0,
        )
    )

    return max(5, min(estimated, 18))


# ============================================================
# COLOR SEGMENTATION
# ============================================================

def separate_colors(image_bgr):
    """
    Produce red and blue binary masks.
    """
    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV,
    )

    red_mask_1 = cv2.inRange(
        hsv,
        np.array(
            CONFIG["red_low_1"],
            dtype=np.uint8,
        ),
        np.array(
            CONFIG["red_high_1"],
            dtype=np.uint8,
        ),
    )

    red_mask_2 = cv2.inRange(
        hsv,
        np.array(
            CONFIG["red_low_2"],
            dtype=np.uint8,
        ),
        np.array(
            CONFIG["red_high_2"],
            dtype=np.uint8,
        ),
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2,
    )

    blue_mask = cv2.inRange(
        hsv,
        np.array(
            CONFIG["blue_low"],
            dtype=np.uint8,
        ),
        np.array(
            CONFIG["blue_high"],
            dtype=np.uint8,
        ),
    )

    # Remove isolated color noise while preserving symbol strokes.
    red_mask = cv2.medianBlur(
        red_mask,
        3,
    )

    blue_mask = cv2.medianBlur(
        blue_mask,
        3,
    )

    return red_mask, blue_mask


def limit_mask_to_graph(mask, plot_rectangle):
    """
    Remove colored pixels outside the calibrated graph.
    """
    graph_mask = np.zeros_like(mask)

    x1, y1, x2, y2 = plot_rectangle

    graph_mask[
        y1:y2 + 1,
        x1:x2 + 1,
    ] = 255

    return cv2.bitwise_and(
        mask,
        graph_mask,
    )


# ============================================================
# SYMBOL TEMPLATES
# ============================================================

def make_symbol_template(symbol_type, radius):
    """
    Create a binary template for an AC or BC symbol.
    """
    size = 2 * radius + 1

    template = np.zeros(
        (size, size),
        dtype=np.uint8,
    )

    center = radius

    # Keep the symbol away from the patch border.
    extent = max(3, int(round(radius * 0.72)))
    thickness = max(1, int(round(radius * 0.18)))

    left = center - extent
    right = center + extent
    top = center - extent
    bottom = center + extent

    if symbol_type == "O":
        cv2.circle(
            template,
            (center, center),
            extent,
            255,
            thickness,
            cv2.LINE_AA,
        )

    elif symbol_type == "X":
        cv2.line(
            template,
            (left, top),
            (right, bottom),
            255,
            thickness,
            cv2.LINE_AA,
        )

        cv2.line(
            template,
            (left, bottom),
            (right, top),
            255,
            thickness,
            cv2.LINE_AA,
        )

    elif symbol_type == "TRIANGLE":
        points = np.array(
            [
                [center, top],
                [left, bottom],
                [right, bottom],
            ],
            dtype=np.int32,
        )

        cv2.polylines(
            template,
            [points],
            isClosed=True,
            color=255,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    elif symbol_type == "SQUARE":
        cv2.rectangle(
            template,
            (left, top),
            (right, bottom),
            255,
            thickness,
            cv2.LINE_AA,
        )

    elif symbol_type == "LESS_THAN":
        # <
        cv2.line(
            template,
            (right, top),
            (left, center),
            255,
            thickness,
            cv2.LINE_AA,
        )

        cv2.line(
            template,
            (left, center),
            (right, bottom),
            255,
            thickness,
            cv2.LINE_AA,
        )

    elif symbol_type == "GREATER_THAN":
        # >
        cv2.line(
            template,
            (left, top),
            (right, center),
            255,
            thickness,
            cv2.LINE_AA,
        )

        cv2.line(
            template,
            (right, center),
            (left, bottom),
            255,
            thickness,
            cv2.LINE_AA,
        )

    elif symbol_type == "LEFT_BRACKET":
        # [
        cv2.line(
            template,
            (left, top),
            (left, bottom),
            255,
            thickness,
            cv2.LINE_AA,
        )

        cv2.line(
            template,
            (left, top),
            (right, top),
            255,
            thickness,
            cv2.LINE_AA,
        )

        cv2.line(
            template,
            (left, bottom),
            (right, bottom),
            255,
            thickness,
            cv2.LINE_AA,
        )

    elif symbol_type == "RIGHT_BRACKET":
        # ]
        cv2.line(
            template,
            (right, top),
            (right, bottom),
            255,
            thickness,
            cv2.LINE_AA,
        )

        cv2.line(
            template,
            (left, top),
            (right, top),
            255,
            thickness,
            cv2.LINE_AA,
        )

        cv2.line(
            template,
            (left, bottom),
            (right, bottom),
            255,
            thickness,
            cv2.LINE_AA,
        )

    else:
        raise ValueError(
            f"Unsupported symbol type: {symbol_type}"
        )

    # Convert anti-aliased pixels to a binary template.
    template = np.where(
        template > 40,
        255,
        0,
    ).astype(np.uint8)

    return template


def build_templates(radius):
    """
    Build all AC and BC templates.
    """
    symbol_types = [
        "O",
        "X",
        "TRIANGLE",
        "SQUARE",
        "LESS_THAN",
        "GREATER_THAN",
        "LEFT_BRACKET",
        "RIGHT_BRACKET",
    ]

    return {
        symbol_type: make_symbol_template(
            symbol_type,
            radius,
        )
        for symbol_type in symbol_types
    }


# ============================================================
# PATCH EXTRACTION AND TEMPLATE SCORING
# ============================================================

def extract_patch(mask, center_x, center_y, radius):
    """
    Extract a fixed-size patch, padding outside-image areas
    with zeros.
    """
    size = 2 * radius + 1

    patch = np.zeros(
        (size, size),
        dtype=np.uint8,
    )

    x1 = center_x - radius
    y1 = center_y - radius
    x2 = center_x + radius + 1
    y2 = center_y + radius + 1

    source_x1 = max(0, x1)
    source_y1 = max(0, y1)
    source_x2 = min(mask.shape[1], x2)
    source_y2 = min(mask.shape[0], y2)

    if (
        source_x2 <= source_x1
        or source_y2 <= source_y1
    ):
        return patch

    destination_x1 = source_x1 - x1
    destination_y1 = source_y1 - y1

    destination_x2 = (
        destination_x1
        + source_x2
        - source_x1
    )

    destination_y2 = (
        destination_y1
        + source_y2
        - source_y1
    )

    patch[
        destination_y1:destination_y2,
        destination_x1:destination_x2,
    ] = mask[
        source_y1:source_y2,
        source_x1:source_x2,
    ]

    return patch


def score_template(patch, template):
    """
    Compare a color-mask patch with a symbol template.

    A one-pixel tolerance is added to account for anti-aliasing,
    image scaling, and line thickness.
    """
    observed = np.where(
        patch > 0,
        255,
        0,
    ).astype(np.uint8)

    expected = np.where(
        template > 0,
        255,
        0,
    ).astype(np.uint8)

    if (
        cv2.countNonZero(observed) == 0
        or cv2.countNonZero(expected) == 0
    ):
        return 0.0

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    observed_dilated = cv2.dilate(
        observed,
        kernel,
        iterations=1,
    )

    expected_dilated = cv2.dilate(
        expected,
        kernel,
        iterations=1,
    )

    # Recall: how much of the expected symbol is represented.
    expected_overlap = cv2.bitwise_and(
        expected,
        observed_dilated,
    )

    recall = (
        cv2.countNonZero(expected_overlap)
        / max(cv2.countNonZero(expected), 1)
    )

    # Precision: how much of the observed marking belongs near
    # the expected symbol.
    observed_overlap = cv2.bitwise_and(
        observed,
        expected_dilated,
    )

    precision = (
        cv2.countNonZero(observed_overlap)
        / max(cv2.countNonZero(observed), 1)
    )

    # Require both recall and precision. The harmonic mean helps
    # prevent a single connecting line from scoring highly.
    if recall + precision == 0:
        return 0.0

    f1_score = (
        2.0
        * recall
        * precision
        / (recall + precision)
    )

    return float(f1_score)


def score_candidate_position(
    mask,
    expected_x,
    expected_y,
    radius,
    template,
    horizontal_search,
    vertical_search,
):
    """
    Search around an expected frequency/dB coordinate to allow
    for small calibration errors.
    """
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
                round(
                    expected_x
                    + horizontal_offset
                )
            )

            center_y = int(
                round(
                    expected_y
                    + vertical_offset
                )
            )

            patch = extract_patch(
                mask,
                center_x,
                center_y,
                radius,
            )

            score = score_template(
                patch,
                template,
            )

            if score > best_score:
                best_score = score
                best_center = (
                    center_x,
                    center_y,
                )

    return best_score, best_center


# ============================================================
# THRESHOLD EXTRACTION
# ============================================================

def extract_threshold(
    mask,
    frequency,
    plot_rectangle,
    allowed_symbols,
    templates,
    radius,
    minimum_score,
):
    """
    Find the best allowed symbol at a given frequency.

    The result includes:
        value
        symbol
        score
        center
        reason
    """
    expected_x = frequency_to_x(
        frequency,
        plot_rectangle,
    )

    horizontal_search = max(
        2,
        radius // 2,
    )

    vertical_search = max(
        1,
        radius // 3,
    )

    candidates = []

    for db_hl in DB_CANDIDATES:
        expected_y = db_to_y(
            db_hl,
            plot_rectangle,
        )

        for symbol_type in allowed_symbols:
            score, center = score_candidate_position(
                mask=mask,
                expected_x=expected_x,
                expected_y=expected_y,
                radius=radius,
                template=templates[symbol_type],
                horizontal_search=horizontal_search,
                vertical_search=vertical_search,
            )

            candidates.append(
                {
                    "db": db_hl,
                    "symbol": symbol_type,
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
            "symbol": None,
            "score": best["score"],
            "center": best["center"],
            "reason": (
                "No allowed symbol exceeded the "
                "minimum detection score."
            ),
        }

    # Ignore the same symbol at an adjacent 5 dB position because
    # one symbol can overlap both candidate windows.
    meaningful_competitors = []

    for candidate in candidates[1:]:
        same_symbol = (
            candidate["symbol"]
            == best["symbol"]
        )

        adjacent_db = (
            abs(candidate["db"] - best["db"])
            <= 5
        )

        if same_symbol and adjacent_db:
            continue

        meaningful_competitors.append(candidate)

    if meaningful_competitors:
        competitor = meaningful_competitors[0]

        if (
            competitor["score"]
            >= best["score"]
            * float(CONFIG["ambiguity_ratio"])
        ):
            return {
                "value": "Unclear",
                "symbol": None,
                "score": best["score"],
                "center": best["center"],
                "reason": (
                    "Competing detections: "
                    f"{SYMBOL_DISPLAY_NAMES[best['symbol']]} "
                    f"at {best['db']} dB HL and "
                    f"{SYMBOL_DISPLAY_NAMES[competitor['symbol']]} "
                    f"at {competitor['db']} dB HL."
                ),
            }

    return {
        "value": best["db"],
        "symbol": best["symbol"],
        "score": best["score"],
        "center": best["center"],
        "reason": "",
    }


def extract_all_thresholds(
    red_mask,
    blue_mask,
    plot_rectangle,
    templates,
    symbol_radius,
):
    """
    Extract AC and BC candidates for both ears.
    """
    results = {
        "right": {
            "ac": {},
            "bc_detected": {},
            "bc_final": {},
            "bc_mode": None,
        },
        "left": {
            "ac": {},
            "bc_detected": {},
            "bc_final": {},
            "bc_mode": None,
        },
    }

    ear_masks = {
        "right": red_mask,
        "left": blue_mask,
    }

    for ear in ("right", "left"):
        mask = ear_masks[ear]

        for frequency in TARGET_FREQUENCIES:
            results[ear]["ac"][frequency] = (
                extract_threshold(
                    mask=mask,
                    frequency=frequency,
                    plot_rectangle=plot_rectangle,
                    allowed_symbols=EAR_SYMBOLS[ear]["ac"],
                    templates=templates,
                    radius=symbol_radius,
                    minimum_score=float(
                        CONFIG["ac_min_score"]
                    ),
                )
            )

            results[ear]["bc_detected"][frequency] = (
                extract_threshold(
                    mask=mask,
                    frequency=frequency,
                    plot_rectangle=plot_rectangle,
                    allowed_symbols=EAR_SYMBOLS[ear]["bc"],
                    templates=templates,
                    radius=symbol_radius,
                    minimum_score=float(
                        CONFIG["bc_min_score"]
                    ),
                )
            )

    return results


# ============================================================
# BC FALLBACK
# ============================================================

def apply_bc_fallback(results):
    """
    Apply the BC=AC fallback separately to each ear.

    If any BC value is confidently detected for an ear, that ear
    is placed in detected-BC mode and no AC-to-BC substitution is
    performed at its other frequencies.
    """
    for ear in ("right", "left"):
        accepted_bc_count = sum(
            isinstance(
                results[ear]["bc_detected"][frequency]["value"],
                int,
            )
            for frequency in TARGET_FREQUENCIES
        )

        if accepted_bc_count == 0:
            results[ear]["bc_mode"] = (
                "BC assumed equal to AC; "
                "no BC symbols detected"
            )

            for frequency in TARGET_FREQUENCIES:
                ac_result = results[ear]["ac"][frequency]

                if isinstance(ac_result["value"], int):
                    results[ear]["bc_final"][frequency] = {
                        "value": ac_result["value"],
                        "symbol": None,
                        "score": ac_result["score"],
                        "center": ac_result["center"],
                        "source": "Assumed equal to AC",
                        "reason": (
                            "No BC symbols were detected "
                            "for this ear."
                        ),
                    }
                else:
                    results[ear]["bc_final"][frequency] = {
                        "value": "Unclear",
                        "symbol": None,
                        "score": ac_result["score"],
                        "center": ac_result["center"],
                        "source": "AC fallback unavailable",
                        "reason": (
                            "No BC symbols were detected and "
                            "the AC threshold was unclear."
                        ),
                    }

        else:
            results[ear]["bc_mode"] = (
                "BC symbols detected; no AC fallback used"
            )

            for frequency in TARGET_FREQUENCIES:
                detected = (
                    results[ear]["bc_detected"][frequency]
                )

                if isinstance(detected["value"], int):
                    results[ear]["bc_final"][frequency] = {
                        **detected,
                        "source": "Detected BC symbol",
                    }
                else:
                    results[ear]["bc_final"][frequency] = {
                        "value": "Not recorded",
                        "symbol": None,
                        "score": detected["score"],
                        "center": detected["center"],
                        "source": "No accepted BC symbol",
                        "reason": detected["reason"],
                    }


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def symbol_description(result):
    """
    Return a readable symbol description.
    """
    symbol = result.get("symbol")

    if symbol is None:
        return ""

    return SYMBOL_DISPLAY_NAMES.get(
        symbol,
        symbol,
    )


def build_rows(results):
    """
    Build AC and BC output rows.
    """
    rows = []

    for frequency in TARGET_FREQUENCIES:
        right_ac = results["right"]["ac"][frequency]
        left_ac = results["left"]["ac"][frequency]

        rows.append(
            [
                f"AC {FREQUENCY_LABELS[frequency]} (dB HL)",
                right_ac["value"],
                symbol_description(right_ac),
                left_ac["value"],
                symbol_description(left_ac),
            ]
        )

    for frequency in TARGET_FREQUENCIES:
        right_bc = results["right"]["bc_final"][frequency]
        left_bc = results["left"]["bc_final"][frequency]

        rows.append(
            [
                f"BC {FREQUENCY_LABELS[frequency]} (dB HL)",
                right_bc["value"],
                right_bc["source"],
                left_bc["value"],
                left_bc["source"],
            ]
        )

    return rows


def write_csv(output_path, rows):
    """
    Save AC and BC measurements to CSV.
    """
    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "Measurement",
                "Right Ear",
                "Right Symbol/Source",
                "Left Ear",
                "Left Symbol/Source",
            ]
        )

        writer.writerows(rows)


def print_table(rows):
    """
    Print a text table without additional dependencies.
    """
    headers = [
        "Measurement",
        "Right Ear",
        "Right Symbol/Source",
        "Left Ear",
        "Left Symbol/Source",
    ]

    text_rows = [
        [str(value) for value in row]
        for row in rows
    ]

    all_rows = [headers] + text_rows

    widths = [
        max(
            len(row[column])
            for row in all_rows
        )
        for column in range(len(headers))
    ]

    def format_row(row):
        return " | ".join(
            str(row[index]).ljust(widths[index])
            for index in range(len(headers))
        )

    print()
    print(format_row(headers))

    print(
        "-+-".join(
            "-" * width
            for width in widths
        )
    )

    for row in text_rows:
        print(format_row(row))


# ============================================================
# IMAGE ANNOTATION
# ============================================================

def annotate_result(
    image,
    frequency,
    result,
    ear,
    measurement_type,
    color,
):
    """
    Add one accepted AC or measured BC threshold to the
    verification image.

    Assumed BC values are not marked a second time because they
    occupy the same location as AC and were not independently
    measured.
    """
    value = result.get("value")
    center = result.get("center")

    if not isinstance(value, int):
        return

    if (
        measurement_type == "BC"
        and result.get("source") == "Assumed equal to AC"
    ):
        return

    if center is None:
        return

    ear_label = "R" if ear == "right" else "L"
    symbol = symbol_description(result)

    if measurement_type == "AC":
        marker_radius = 9
        thickness = 2
    else:
        marker_radius = 6
        thickness = 1

    cv2.circle(
        image,
        center,
        marker_radius,
        color,
        thickness,
        cv2.LINE_AA,
    )

    label = (
        f"{ear_label}-{measurement_type} "
        f"{frequency}: {value} "
        f"{symbol}"
    ).strip()

    cv2.putText(
        image,
        label,
        (
            center[0] + 8,
            center[1] - 6,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color,
        1,
        cv2.LINE_AA,
    )


def annotate_results(
    image,
    results,
    plot_rectangle,
):
    """
    Create the annotated verification image.
    """
    annotated = image.copy()

    x1, y1, x2, y2 = plot_rectangle

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 200, 0),
        2,
    )

    ear_colors = {
        "right": (0, 0, 255),
        "left": (255, 0, 0),
    }

    for ear in ("right", "left"):
        color = ear_colors[ear]

        for frequency in TARGET_FREQUENCIES:
            annotate_result(
                image=annotated,
                frequency=frequency,
                result=results[ear]["ac"][frequency],
                ear=ear,
                measurement_type="AC",
                color=color,
            )

            annotate_result(
                image=annotated,
                frequency=frequency,
                result=results[ear]["bc_final"][frequency],
                ear=ear,
                measurement_type="BC",
                color=color,
            )

    return annotated


# ============================================================
# MANUAL VERIFICATION
# ============================================================

def collect_verification_items(results):
    """
    Identify measurements that should be manually reviewed.
    """
    items = []

    for ear in ("right", "left"):
        ear_name = (
            "Right"
            if ear == "right"
            else "Left"
        )

        # Review unclear AC measurements.
        for frequency in TARGET_FREQUENCIES:
            result = results[ear]["ac"][frequency]

            if not isinstance(result["value"], int):
                items.append(
                    f"{ear_name} AC, {frequency} Hz: "
                    f"{result.get('reason', 'Unclear')} "
                    f"Best score={result.get('score', 0):.3f}"
                )

        # In detected-BC mode, review missing or unclear BC.
        if (
            results[ear]["bc_mode"]
            == "BC symbols detected; no AC fallback used"
        ):
            for frequency in TARGET_FREQUENCIES:
                final_result = (
                    results[ear]["bc_final"][frequency]
                )

                if final_result["value"] == "Not recorded":
                    detected_result = (
                        results[ear]["bc_detected"][frequency]
                    )

                    items.append(
                        f"{ear_name} BC, {frequency} Hz: "
                        "no accepted BC symbol. "
                        f"Best score="
                        f"{detected_result.get('score', 0):.3f}"
                    )

        else:
            items.append(
                f"{ear_name} BC: no accepted BC symbols "
                "were detected; BC values were copied from AC "
                "and should not be treated as measured BC."
            )

    return items


# ============================================================
# MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # 1. Read image
    # --------------------------------------------------------
    image_path = Path(
        CONFIG["image_path"]
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image does not exist: "
            f"{image_path.resolve()}"
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
    # 2. Select or read plotting rectangle
    # --------------------------------------------------------
    plot_rectangle = CONFIG["plot_rectangle"]

    if plot_rectangle is None:
        print(
            "\nSelect the inner audiogram graph.\n"
            "Upper-left:  125 Hz, -10 dB HL\n"
            "Lower-right: 16 kHz, 130 dB HL\n"
        )

        plot_rectangle = select_plot_rectangle(
            image
        )

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
    # 3. Separate red and blue pixels
    # --------------------------------------------------------
    red_mask, blue_mask = separate_colors(
        image
    )

    red_mask = limit_mask_to_graph(
        red_mask,
        plot_rectangle,
    )

    blue_mask = limit_mask_to_graph(
        blue_mask,
        plot_rectangle,
    )

    # --------------------------------------------------------
    # 4. Set symbol radius and templates
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
            "CONFIG['symbol_radius'] must be at least 3."
        )

    templates = build_templates(
        symbol_radius
    )

    # --------------------------------------------------------
    # 5. Extract AC and BC symbols
    # --------------------------------------------------------
    results = extract_all_thresholds(
        red_mask=red_mask,
        blue_mask=blue_mask,
        plot_rectangle=plot_rectangle,
        templates=templates,
        symbol_radius=symbol_radius,
    )

    # --------------------------------------------------------
    # 6. Apply requested BC=AC fallback
    # --------------------------------------------------------
    apply_bc_fallback(results)

    # --------------------------------------------------------
    # 7. Prepare outputs
    # --------------------------------------------------------
    output_directory = Path(
        CONFIG["output_dir"]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotated = annotate_results(
        image=image,
        results=results,
        plot_rectangle=plot_rectangle,
    )

    rows = build_rows(results)

    # --------------------------------------------------------
    # 8. Save masks
    # --------------------------------------------------------
    if CONFIG["save_masks"]:
        red_mask_path = (
            output_directory
            / "red_mask.png"
        )

        blue_mask_path = (
            output_directory
            / "blue_mask.png"
        )

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

    # --------------------------------------------------------
    # 9. Save annotated image
    # --------------------------------------------------------
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
    # 10. Save CSV
    # --------------------------------------------------------
    csv_path = (
        output_directory
        / "measurements.csv"
    )

    if CONFIG["save_csv"]:
        write_csv(
            csv_path,
            rows,
        )

    # --------------------------------------------------------
    # 11. Print results
    # --------------------------------------------------------
    print_table(rows)

    print("\nCalibration")
    print(
        f"  Plot rectangle: {plot_rectangle}"
    )
    print(
        f"  Symbol radius:  {symbol_radius} pixels"
    )

    print("\nBone-conduction mode")
    print(
        "  Right ear: "
        f"{results['right']['bc_mode']}"
    )
    print(
        "  Left ear:  "
        f"{results['left']['bc_mode']}"
    )

    # --------------------------------------------------------
    # 12. Print manual verification list
    # --------------------------------------------------------
    verification_items = (
        collect_verification_items(results)
    )

    print(
        "\nMeasurements requiring manual verification"
    )

    if verification_items:
        for item in verification_items:
            print(f"  - {item}")
    else:
        print(
            "  - None flagged by the automated checks."
        )

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
