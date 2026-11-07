#!/usr/bin/env python3
"""
Extract air-conduction (AC) and bone-conduction (BC) thresholds
from a color-coded audiogram.

Supported standard symbols
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

Important AC-line handling
--------------------------
Air-conduction symbols are commonly joined by a colored line.

This script therefore:
    1. Does not penalize extra colored pixels entering a symbol.
    2. Requires every structural part of a symbol to be present.
    3. Detects AC first.
    4. Removes the AC connecting-line segments before searching for BC.
    5. Prevents AC symbols from being reclassified as BC symbols.

BC fallback
-----------
The fallback is applied separately to each ear.

If no accepted BC symbols are detected for an ear:
    BC is assumed equal to AC.

If one or more accepted BC symbols are detected for an ear:
    Only detected BC thresholds are reported.
    Missing BC frequencies are reported as "Not recorded".

Copied BC values are explicitly labeled "Assumed equal to AC" and
should not be interpreted as independently measured BC thresholds.

Outputs
-------
    measurements.csv
    audiogram_annotated.png
    red_mask.png
    blue_mask.png
    right_bc_search_mask.png
    left_bc_search_mask.png

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

# Full calibrated graph range.
GRAPH_MIN_FREQUENCY = 125
GRAPH_MAX_FREQUENCY = 16000

GRAPH_MIN_DB = -10
GRAPH_MAX_DB = 130

# Thresholds are evaluated only at conventional 5 dB steps.
DB_CANDIDATES = list(
    range(
        GRAPH_MIN_DB,
        GRAPH_MAX_DB + 1,
        5,
    )
)


# ============================================================
# STANDARD AUDIOGRAM SYMBOLS
# ============================================================

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

EAR_SYMBOLS = {
    "right": {
        "ac": [
            "O",
            "TRIANGLE",
        ],
        "bc": [
            "LESS_THAN",
            "LEFT_BRACKET",
        ],
    },
    "left": {
        "ac": [
            "X",
            "SQUARE",
        ],
        "bc": [
            "GREATER_THAN",
            "RIGHT_BRACKET",
        ],
    },
}


# ============================================================
# USER CONFIGURATION
# ============================================================

CONFIG = {
    # Input audiogram image.
    "image_path": (
        r"C:\Users\natha\Downloads\audiogram_output\Screenshot_audiogram.png"
    ),

    # Output directory.
    "output_dir": (
        r"C:\Users\natha\Downloads\audiogram_output"
    ),

    # Inner audiogram graph rectangle:
    #     (left_x, top_y, right_x, bottom_y)
    #
    # It should correspond to:
    #     left:   125 Hz
    #     right:  16 kHz
    #     top:    -10 dB HL
    #     bottom: 130 dB HL
    #
    # Keep as None to select it interactively.
    "plot_rectangle": None,

    # Example:
    # "plot_rectangle": (210, 85, 1450, 930),

    # Approximate radius of an audiogram symbol in pixels.
    # Keep as None for automatic estimation.
    "symbol_radius": None,

    # Example:
    # "symbol_radius": 9,

    # Minimum structural symbol scores.
    #
    # Increase to reduce false positives.
    # Decrease if real symbols are missed.
    "ac_min_score": 0.42,
    "bc_min_score": 0.62,

    # If a second nonadjacent dB candidate has nearly the same score,
    # the result is marked Unclear.
    "ambiguity_ratio": 0.94,

    # Number of accepted BC symbols required before treating BC as
    # independently measured for that ear.
    #
    # Keep at 1 to follow:
    #     if any BC sign is present, BC is not assumed equal to AC.
    #
    # Change to 2 if one isolated false-positive BC detection is
    # frequently triggering measured-BC mode.
    "minimum_bc_symbols_for_measured_mode": 1,

    # Remove the AC lines between successfully detected AC symbols
    # before looking for BC symbols.
    "remove_ac_connecting_lines": True,

    # Thickness of the line-removal mask.
    #
    # Keep as None to estimate it from the symbol radius.
    "ac_line_removal_thickness": None,

    # Keep the area immediately surrounding an AC symbol intact when
    # removing connecting lines. The AC symbol itself is handled by
    # the BC exclusion rule below.
    "ac_endpoint_preservation_radius_factor": 1.15,

    # Reject a BC candidate whose center is too close to a detected
    # AC symbol at the same frequency.
    #
    # This is particularly important for preventing half of a blue X
    # from being interpreted as > or <.
    "reject_bc_near_ac": True,
    "bc_ac_exclusion_radius_factor": 0.90,

    # HSV color ranges.
    #
    # OpenCV hue range is 0-179.
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
    Interactively select the inner audiogram plotting rectangle.
    """
    window_name = (
        "Select INNER graph: 125 Hz/-10 dB to "
        "16 kHz/130 dB; then press ENTER"
    )

    x, y, width, height = cv2.selectROI(
        window_name,
        image,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyAllWindows()

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "No audiogram plotting rectangle was selected."
        )

    return (
        int(x),
        int(y),
        int(x + width),
        int(y + height),
    )


def validate_rectangle(rectangle, image_shape):
    """
    Confirm that the graph rectangle is inside the image.
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
    Convert frequency to an x-coordinate.

    The audiogram frequency axis is logarithmic in base 2.
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

    Larger dB HL values appear lower on the graph.
    """
    _, y1, _, y2 = plot_rectangle

    fraction = (
        (db_hl - GRAPH_MIN_DB)
        / (GRAPH_MAX_DB - GRAPH_MIN_DB)
    )

    return y1 + fraction * (y2 - y1)


def estimate_symbol_radius(plot_rectangle):
    """
    Estimate the symbol radius from the graph dimensions.
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

    return max(
        5,
        min(estimated, 18),
    )


# ============================================================
# COLOR SEGMENTATION
# ============================================================

def separate_colors(image_bgr):
    """
    Produce red and blue binary masks.

    Red requires two HSV ranges because red wraps around the
    OpenCV hue scale.
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

    # Suppress isolated noise while retaining symbol strokes.
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
# PATCH EXTRACTION
# ============================================================

def extract_patch(mask, center_x, center_y, radius):
    """
    Extract a fixed-size patch around a candidate coordinate.

    Areas outside the image are padded with zeros.
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


# ============================================================
# LINE-TOLERANT SYMBOL TEMPLATES
# ============================================================

def blank_stroke(radius):
    """
    Create an empty binary stroke mask.
    """
    size = 2 * radius + 1

    return np.zeros(
        (size, size),
        dtype=np.uint8,
    )


def make_line_stroke(
    radius,
    point_1,
    point_2,
    thickness,
):
    """
    Create one required line-stroke mask.
    """
    stroke = blank_stroke(radius)

    cv2.line(
        stroke,
        point_1,
        point_2,
        255,
        thickness,
        cv2.LINE_AA,
    )

    return np.where(
        stroke > 40,
        255,
        0,
    ).astype(np.uint8)


def make_arc_stroke(
    radius,
    center,
    axes,
    start_angle,
    end_angle,
    thickness,
):
    """
    Create one required arc-stroke mask.
    """
    stroke = blank_stroke(radius)

    cv2.ellipse(
        stroke,
        center,
        axes,
        0,
        start_angle,
        end_angle,
        255,
        thickness,
        cv2.LINE_AA,
    )

    return np.where(
        stroke > 40,
        255,
        0,
    ).astype(np.uint8)


def build_symbol_strokes(radius):
    """
    Build each audiogram symbol as a collection of required strokes.

    Extra pixels are not penalized because AC symbols commonly have
    a colored line entering and leaving the symbol.
    """
    center = radius

    extent = max(
        3,
        int(round(radius * 0.72)),
    )

    thickness = max(
        1,
        int(round(radius * 0.16)),
    )

    left = center - extent
    right = center + extent
    top = center - extent
    bottom = center + extent

    templates = {}

    # --------------------------------------------------------
    # O: eight ring segments are required.
    # --------------------------------------------------------
    templates["O"] = []

    for start_angle in range(0, 360, 45):
        templates["O"].append(
            make_arc_stroke(
                radius=radius,
                center=(center, center),
                axes=(extent, extent),
                start_angle=start_angle,
                end_angle=start_angle + 40,
                thickness=thickness,
            )
        )

    # --------------------------------------------------------
    # X: both diagonals are required.
    # --------------------------------------------------------
    templates["X"] = [
        make_line_stroke(
            radius,
            (left, top),
            (right, bottom),
            thickness,
        ),
        make_line_stroke(
            radius,
            (left, bottom),
            (right, top),
            thickness,
        ),
    ]

    # --------------------------------------------------------
    # Triangle: all three sides are required.
    # --------------------------------------------------------
    templates["TRIANGLE"] = [
        make_line_stroke(
            radius,
            (center, top),
            (left, bottom),
            thickness,
        ),
        make_line_stroke(
            radius,
            (left, bottom),
            (right, bottom),
            thickness,
        ),
        make_line_stroke(
            radius,
            (right, bottom),
            (center, top),
            thickness,
        ),
    ]

    # --------------------------------------------------------
    # Square: all four sides are required.
    # --------------------------------------------------------
    templates["SQUARE"] = [
        make_line_stroke(
            radius,
            (left, top),
            (right, top),
            thickness,
        ),
        make_line_stroke(
            radius,
            (right, top),
            (right, bottom),
            thickness,
        ),
        make_line_stroke(
            radius,
            (right, bottom),
            (left, bottom),
            thickness,
        ),
        make_line_stroke(
            radius,
            (left, bottom),
            (left, top),
            thickness,
        ),
    ]

    # --------------------------------------------------------
    # Right unmasked BC: <
    # --------------------------------------------------------
    templates["LESS_THAN"] = [
        make_line_stroke(
            radius,
            (right, top),
            (left, center),
            thickness,
        ),
        make_line_stroke(
            radius,
            (left, center),
            (right, bottom),
            thickness,
        ),
    ]

    # --------------------------------------------------------
    # Left unmasked BC: >
    # --------------------------------------------------------
    templates["GREATER_THAN"] = [
        make_line_stroke(
            radius,
            (left, top),
            (right, center),
            thickness,
        ),
        make_line_stroke(
            radius,
            (right, center),
            (left, bottom),
            thickness,
        ),
    ]

    # --------------------------------------------------------
    # Right masked BC: [
    # --------------------------------------------------------
    templates["LEFT_BRACKET"] = [
        make_line_stroke(
            radius,
            (left, top),
            (left, bottom),
            thickness,
        ),
        make_line_stroke(
            radius,
            (left, top),
            (right, top),
            thickness,
        ),
        make_line_stroke(
            radius,
            (left, bottom),
            (right, bottom),
            thickness,
        ),
    ]

    # --------------------------------------------------------
    # Left masked BC: ]
    # --------------------------------------------------------
    templates["RIGHT_BRACKET"] = [
        make_line_stroke(
            radius,
            (right, top),
            (right, bottom),
            thickness,
        ),
        make_line_stroke(
            radius,
            (left, top),
            (right, top),
            thickness,
        ),
        make_line_stroke(
            radius,
            (left, bottom),
            (right, bottom),
            thickness,
        ),
    ]

    return templates


# ============================================================
# LINE-TOLERANT SYMBOL SCORING
# ============================================================

def score_required_strokes(
    patch,
    required_strokes,
):
    """
    Score whether all required symbol strokes are present.

    Extra colored pixels are not penalized. This allows a colored
    AC line to enter and leave an otherwise valid symbol.

    The weakest required stroke is weighted strongly. Therefore:
        - one diagonal should not become an X;
        - one arm should not become < or >;
        - one line should not become a bracket;
        - an AC connecting line should not become a complete symbol.
    """
    observed = np.where(
        patch > 0,
        255,
        0,
    ).astype(np.uint8)

    if cv2.countNonZero(observed) == 0:
        return 0.0

    tolerance_kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    observed_with_tolerance = cv2.dilate(
        observed,
        tolerance_kernel,
        iterations=1,
    )

    stroke_recalls = []

    for stroke in required_strokes:
        expected_pixel_count = cv2.countNonZero(
            stroke
        )

        if expected_pixel_count == 0:
            continue

        overlap = cv2.bitwise_and(
            stroke,
            observed_with_tolerance,
        )

        recall = (
            cv2.countNonZero(overlap)
            / expected_pixel_count
        )

        stroke_recalls.append(
            float(recall)
        )

    if not stroke_recalls:
        return 0.0

    average_recall = float(
        np.mean(stroke_recalls)
    )

    weakest_recall = float(
        np.min(stroke_recalls)
    )

    score = (
        0.60 * average_recall
        + 0.40 * weakest_recall
    )

    return max(
        0.0,
        min(1.0, score),
    )


def score_candidate_position(
    mask,
    expected_x,
    expected_y,
    radius,
    required_strokes,
    horizontal_search,
    vertical_search,
    excluded_center=None,
    exclusion_radius=0,
):
    """
    Search around the expected graph coordinate.

    If excluded_center is provided, candidate centers too close to
    that point are skipped. This prevents a detected AC symbol from
    being classified again as a BC symbol.
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

            if excluded_center is not None:
                distance = math.hypot(
                    center_x - excluded_center[0],
                    center_y - excluded_center[1],
                )

                if distance < exclusion_radius:
                    continue

            patch = extract_patch(
                mask,
                center_x,
                center_y,
                radius,
            )

            score = score_required_strokes(
                patch,
                required_strokes,
            )

            if score > best_score:
                best_score = score
                best_center = (
                    center_x,
                    center_y,
                )

    if best_score < 0:
        return 0.0, None

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
    excluded_center=None,
    exclusion_radius=0,
):
    """
    Find the strongest allowed symbol at one frequency.

    Different symbol types at the same dB level are treated as
    alternatives rather than competing threshold values.

    Ambiguity is evaluated between nonadjacent dB levels.
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

    candidates_by_level = []

    for db_hl in DB_CANDIDATES:
        expected_y = db_to_y(
            db_hl,
            plot_rectangle,
        )

        symbol_candidates = []

        for symbol_type in allowed_symbols:
            score, center = score_candidate_position(
                mask=mask,
                expected_x=expected_x,
                expected_y=expected_y,
                radius=radius,
                required_strokes=templates[symbol_type],
                horizontal_search=horizontal_search,
                vertical_search=vertical_search,
                excluded_center=excluded_center,
                exclusion_radius=exclusion_radius,
            )

            symbol_candidates.append(
                {
                    "db": db_hl,
                    "symbol": symbol_type,
                    "score": score,
                    "center": center,
                }
            )

        best_at_level = max(
            symbol_candidates,
            key=lambda item: item["score"],
        )

        candidates_by_level.append(
            best_at_level
        )

    candidates_by_level.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best = candidates_by_level[0]

    if best["score"] < minimum_score:
        return {
            "value": "Unclear",
            "symbol": None,
            "score": best["score"],
            "center": best["center"],
            "reason": (
                "No complete allowed symbol exceeded "
                "the minimum detection score."
            ),
        }

    # An actual symbol can overlap the adjacent 5 dB search window.
    # Only compare candidates more than 5 dB away.
    nonadjacent_candidates = [
        candidate
        for candidate in candidates_by_level[1:]
        if abs(candidate["db"] - best["db"]) > 5
    ]

    if nonadjacent_candidates:
        competitor = nonadjacent_candidates[0]

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
                    "Competing complete-symbol detections at "
                    f"{best['db']} and "
                    f"{competitor['db']} dB HL."
                ),
            }

    return {
        "value": best["db"],
        "symbol": best["symbol"],
        "score": best["score"],
        "center": best["center"],
        "reason": "",
    }


# ============================================================
# AC CONNECTING-LINE REMOVAL
# ============================================================

def remove_ac_connecting_lines(
    color_mask,
    ac_results,
    symbol_radius,
):
    """
    Remove line segments joining consecutive accepted AC symbols.

    The endpoint neighborhoods are preserved so the AC symbols
    themselves are not removed.

    This cleaned mask is used only for BC searching. The original
    color mask remains available for AC analysis and output.
    """
    if not CONFIG["remove_ac_connecting_lines"]:
        return color_mask.copy()

    line_removal_mask = np.zeros_like(
        color_mask
    )

    accepted_points = []

    for frequency in TARGET_FREQUENCIES:
        result = ac_results[frequency]

        if (
            isinstance(result.get("value"), int)
            and result.get("center") is not None
        ):
            accepted_points.append(
                (
                    frequency,
                    result["center"],
                )
            )

    accepted_points.sort(
        key=lambda item: item[0]
    )

    configured_thickness = CONFIG[
        "ac_line_removal_thickness"
    ]

    if configured_thickness is None:
        line_thickness = max(
            2,
            int(round(symbol_radius * 0.35)),
        )
    else:
        line_thickness = max(
            1,
            int(configured_thickness),
        )

    # Draw expected connecting segments between consecutive
    # accepted AC thresholds.
    for index in range(
        len(accepted_points) - 1
    ):
        _, first_center = accepted_points[index]
        _, second_center = accepted_points[index + 1]

        cv2.line(
            line_removal_mask,
            first_center,
            second_center,
            255,
            line_thickness,
            cv2.LINE_AA,
        )

    # Preserve neighborhoods surrounding the AC symbols.
    preservation_radius = max(
        1,
        int(
            round(
                symbol_radius
                * float(
                    CONFIG[
                        "ac_endpoint_preservation_radius_factor"
                    ]
                )
            )
        ),
    )

    for _, center in accepted_points:
        cv2.circle(
            line_removal_mask,
            center,
            preservation_radius,
            0,
            -1,
            cv2.LINE_AA,
        )

    inverse_removal_mask = cv2.bitwise_not(
        line_removal_mask
    )

    return cv2.bitwise_and(
        color_mask,
        inverse_removal_mask,
    )


# ============================================================
# AC AND BC EXTRACTION
# ============================================================

def extract_air_conduction(
    red_mask,
    blue_mask,
    plot_rectangle,
    templates,
    symbol_radius,
):
    """
    Extract AC thresholds before searching for BC.
    """
    results = {
        "right": {},
        "left": {},
    }

    ear_masks = {
        "right": red_mask,
        "left": blue_mask,
    }

    for ear in ("right", "left"):
        for frequency in TARGET_FREQUENCIES:
            results[ear][frequency] = extract_threshold(
                mask=ear_masks[ear],
                frequency=frequency,
                plot_rectangle=plot_rectangle,
                allowed_symbols=EAR_SYMBOLS[ear]["ac"],
                templates=templates,
                radius=symbol_radius,
                minimum_score=float(
                    CONFIG["ac_min_score"]
                ),
            )

    return results


def extract_bone_conduction(
    right_bc_mask,
    left_bc_mask,
    ac_results,
    plot_rectangle,
    templates,
    symbol_radius,
):
    """
    Extract BC thresholds from masks in which detected AC connecting
    lines have been removed.
    """
    results = {
        "right": {},
        "left": {},
    }

    ear_masks = {
        "right": right_bc_mask,
        "left": left_bc_mask,
    }

    exclusion_radius = max(
        1,
        int(
            round(
                symbol_radius
                * float(
                    CONFIG[
                        "bc_ac_exclusion_radius_factor"
                    ]
                )
            )
        ),
    )

    for ear in ("right", "left"):
        for frequency in TARGET_FREQUENCIES:
            ac_result = ac_results[ear][frequency]

            excluded_center = None

            if (
                CONFIG["reject_bc_near_ac"]
                and isinstance(
                    ac_result.get("value"),
                    int,
                )
            ):
                excluded_center = ac_result.get(
                    "center"
                )

            results[ear][frequency] = extract_threshold(
                mask=ear_masks[ear],
                frequency=frequency,
                plot_rectangle=plot_rectangle,
                allowed_symbols=EAR_SYMBOLS[ear]["bc"],
                templates=templates,
                radius=symbol_radius,
                minimum_score=float(
                    CONFIG["bc_min_score"]
                ),
                excluded_center=excluded_center,
                exclusion_radius=exclusion_radius,
            )

    return results


# ============================================================
# BC FALLBACK
# ============================================================

def apply_bc_fallback(
    ac_results,
    detected_bc_results,
):
    """
    Apply BC=AC separately to each ear when no sufficient BC signs
    were detected for that ear.
    """
    final_bc_results = {
        "right": {},
        "left": {},
    }

    bc_modes = {}

    required_count = max(
        1,
        int(
            CONFIG[
                "minimum_bc_symbols_for_measured_mode"
            ]
        ),
    )

    for ear in ("right", "left"):
        accepted_count = sum(
            isinstance(
                detected_bc_results[ear][frequency].get(
                    "value"
                ),
                int,
            )
            for frequency in TARGET_FREQUENCIES
        )

        if accepted_count < required_count:
            bc_modes[ear] = (
                "No sufficient BC symbols detected; "
                "BC assumed equal to AC"
            )

            for frequency in TARGET_FREQUENCIES:
                ac_result = ac_results[ear][frequency]

                if isinstance(
                    ac_result.get("value"),
                    int,
                ):
                    final_bc_results[ear][frequency] = {
                        "value": ac_result["value"],
                        "symbol": None,
                        "score": ac_result["score"],
                        "center": ac_result["center"],
                        "source": "Assumed equal to AC",
                        "reason": (
                            "No sufficient BC symbols were "
                            "detected for this ear."
                        ),
                    }
                else:
                    final_bc_results[ear][frequency] = {
                        "value": "Unclear",
                        "symbol": None,
                        "score": ac_result.get(
                            "score",
                            0.0,
                        ),
                        "center": ac_result.get(
                            "center"
                        ),
                        "source": (
                            "AC fallback unavailable"
                        ),
                        "reason": (
                            "No sufficient BC symbols were "
                            "detected and AC was unclear."
                        ),
                    }

        else:
            bc_modes[ear] = (
                f"{accepted_count} BC symbol(s) detected; "
                "AC fallback not used"
            )

            for frequency in TARGET_FREQUENCIES:
                detected = detected_bc_results[
                    ear
                ][frequency]

                if isinstance(
                    detected.get("value"),
                    int,
                ):
                    final_bc_results[ear][frequency] = {
                        **detected,
                        "source": "Detected BC symbol",
                    }
                else:
                    final_bc_results[ear][frequency] = {
                        "value": "Not recorded",
                        "symbol": None,
                        "score": detected.get(
                            "score",
                            0.0,
                        ),
                        "center": detected.get(
                            "center"
                        ),
                        "source": (
                            "No accepted BC symbol"
                        ),
                        "reason": detected.get(
                            "reason",
                            "",
                        ),
                    }

    return final_bc_results, bc_modes


# ============================================================
# RESULT FORMATTING
# ============================================================

def symbol_description(result):
    """
    Return a printable symbol name.
    """
    symbol = result.get("symbol")

    if symbol is None:
        return ""

    return SYMBOL_DISPLAY_NAMES.get(
        symbol,
        symbol,
    )


def build_output_rows(
    ac_results,
    bc_results,
):
    """
    Build AC and BC output rows.
    """
    rows = []

    for frequency in TARGET_FREQUENCIES:
        right_result = ac_results[
            "right"
        ][frequency]

        left_result = ac_results[
            "left"
        ][frequency]

        rows.append(
            [
                (
                    f"AC {FREQUENCY_LABELS[frequency]} "
                    "(dB HL)"
                ),
                right_result["value"],
                symbol_description(right_result),
                left_result["value"],
                symbol_description(left_result),
            ]
        )

    for frequency in TARGET_FREQUENCIES:
        right_result = bc_results[
            "right"
        ][frequency]

        left_result = bc_results[
            "left"
        ][frequency]

        right_description = (
            symbol_description(right_result)
            if right_result.get("symbol")
            else right_result.get("source", "")
        )

        left_description = (
            symbol_description(left_result)
            if left_result.get("symbol")
            else left_result.get("source", "")
        )

        rows.append(
            [
                (
                    f"BC {FREQUENCY_LABELS[frequency]} "
                    "(dB HL)"
                ),
                right_result["value"],
                right_description,
                left_result["value"],
                left_description,
            ]
        )

    return rows


def write_csv(output_path, rows):
    """
    Save extracted AC and BC measurements.
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
    Print a dependency-free results table.
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

    all_rows = [
        headers,
        *text_rows,
    ]

    widths = [
        max(
            len(row[column])
            for row in all_rows
        )
        for column in range(len(headers))
    ]

    def format_row(row):
        return " | ".join(
            str(row[index]).ljust(
                widths[index]
            )
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
# ANNOTATION
# ============================================================

def annotate_result(
    image,
    frequency,
    result,
    ear,
    measurement_type,
):
    """
    Mark one accepted measured threshold.

    Assumed BC values are not marked separately because they are
    copied from AC and occupy the same graph location.
    """
    value = result.get("value")
    center = result.get("center")

    if not isinstance(value, int):
        return

    if (
        measurement_type == "BC"
        and result.get("source")
        == "Assumed equal to AC"
    ):
        return

    if center is None:
        return

    if ear == "right":
        color = (0, 0, 255)
        ear_label = "R"
    else:
        color = (255, 0, 0)
        ear_label = "L"

    if measurement_type == "AC":
        marker_radius = 9
        marker_thickness = 2
    else:
        marker_radius = 6
        marker_thickness = 1

    cv2.circle(
        image,
        center,
        marker_radius,
        color,
        marker_thickness,
        cv2.LINE_AA,
    )

    symbol = symbol_description(result)

    label = (
        f"{ear_label}-{measurement_type} "
        f"{frequency}: {value}"
    )

    if symbol:
        label += f" {symbol}"

    cv2.putText(
        image,
        label,
        (
            center[0] + 9,
            center[1] - 6,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color,
        1,
        cv2.LINE_AA,
    )


def create_annotated_image(
    image,
    plot_rectangle,
    ac_results,
    bc_results,
):
    """
    Create the verification image.
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

    for ear in ("right", "left"):
        for frequency in TARGET_FREQUENCIES:
            annotate_result(
                image=annotated,
                frequency=frequency,
                result=ac_results[ear][frequency],
                ear=ear,
                measurement_type="AC",
            )

            annotate_result(
                image=annotated,
                frequency=frequency,
                result=bc_results[ear][frequency],
                ear=ear,
                measurement_type="BC",
            )

    return annotated


# ============================================================
# MANUAL VERIFICATION
# ============================================================

def collect_verification_items(
    ac_results,
    detected_bc_results,
    final_bc_results,
    bc_modes,
):
    """
    Identify results that should be manually checked.
    """
    items = []

    for ear in ("right", "left"):
        ear_name = (
            "Right"
            if ear == "right"
            else "Left"
        )

        for frequency in TARGET_FREQUENCIES:
            ac_result = ac_results[
                ear
            ][frequency]

            if not isinstance(
                ac_result.get("value"),
                int,
            ):
                items.append(
                    f"{ear_name} AC, {frequency} Hz: "
                    f"{ac_result.get('reason', 'Unclear')}; "
                    f"best score="
                    f"{ac_result.get('score', 0.0):.3f}"
                )

        if "fallback not used" in bc_modes[ear]:
            for frequency in TARGET_FREQUENCIES:
                bc_result = final_bc_results[
                    ear
                ][frequency]

                if bc_result["value"] == "Not recorded":
                    detected_result = (
                        detected_bc_results[
                            ear
                        ][frequency]
                    )

                    items.append(
                        f"{ear_name} BC, {frequency} Hz: "
                        "no accepted BC symbol; "
                        f"best score="
                        f"{detected_result.get('score', 0.0):.3f}"
                    )
        else:
            items.append(
                f"{ear_name} BC: no sufficient BC symbols "
                "were detected. BC was assumed equal to AC "
                "and was not independently measured."
            )

    return items


# ============================================================
# MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # 1. Read the input image
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
    # 2. Select or load the graph rectangle
    # --------------------------------------------------------
    plot_rectangle = CONFIG[
        "plot_rectangle"
    ]

    if plot_rectangle is None:
        print(
            "\nSelect the inner audiogram plotting rectangle.\n"
            "Upper-left:  125 Hz, -10 dB HL\n"
            "Lower-right: 16 kHz, 130 dB HL\n"
        )

        plot_rectangle = (
            select_plot_rectangle(image)
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
    # 3. Separate red and blue markings
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
    # 4. Determine the symbol radius
    # --------------------------------------------------------
    if CONFIG["symbol_radius"] is None:
        symbol_radius = (
            estimate_symbol_radius(
                plot_rectangle
            )
        )
    else:
        symbol_radius = int(
            CONFIG["symbol_radius"]
        )

    if symbol_radius < 3:
        raise ValueError(
            "CONFIG['symbol_radius'] must be "
            "at least 3 pixels."
        )

    templates = build_symbol_strokes(
        symbol_radius
    )

    # --------------------------------------------------------
    # 5. Extract AC first
    # --------------------------------------------------------
    ac_results = extract_air_conduction(
        red_mask=red_mask,
        blue_mask=blue_mask,
        plot_rectangle=plot_rectangle,
        templates=templates,
        symbol_radius=symbol_radius,
    )

    # --------------------------------------------------------
    # 6. Remove AC connecting lines for BC searching
    # --------------------------------------------------------
    right_bc_search_mask = (
        remove_ac_connecting_lines(
            color_mask=red_mask,
            ac_results=ac_results["right"],
            symbol_radius=symbol_radius,
        )
    )

    left_bc_search_mask = (
        remove_ac_connecting_lines(
            color_mask=blue_mask,
            ac_results=ac_results["left"],
            symbol_radius=symbol_radius,
        )
    )

    # --------------------------------------------------------
    # 7. Extract actual BC symbols
    # --------------------------------------------------------
    detected_bc_results = (
        extract_bone_conduction(
            right_bc_mask=right_bc_search_mask,
            left_bc_mask=left_bc_search_mask,
            ac_results=ac_results,
            plot_rectangle=plot_rectangle,
            templates=templates,
            symbol_radius=symbol_radius,
        )
    )

    # --------------------------------------------------------
    # 8. Apply BC=AC fallback when appropriate
    # --------------------------------------------------------
    final_bc_results, bc_modes = (
        apply_bc_fallback(
            ac_results=ac_results,
            detected_bc_results=(
                detected_bc_results
            ),
        )
    )

    # --------------------------------------------------------
    # 9. Prepare output directory
    # --------------------------------------------------------
    output_directory = Path(
        CONFIG["output_dir"]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 10. Build results
    # --------------------------------------------------------
    rows = build_output_rows(
        ac_results=ac_results,
        bc_results=final_bc_results,
    )

    annotated = create_annotated_image(
        image=image,
        plot_rectangle=plot_rectangle,
        ac_results=ac_results,
        bc_results=final_bc_results,
    )

    # --------------------------------------------------------
    # 11. Save masks
    # --------------------------------------------------------
    if CONFIG["save_masks"]:
        mask_outputs = {
            "red_mask.png": red_mask,
            "blue_mask.png": blue_mask,
            "right_bc_search_mask.png": (
                right_bc_search_mask
            ),
            "left_bc_search_mask.png": (
                left_bc_search_mask
            ),
        }

        for filename, mask in mask_outputs.items():
            output_path = (
                output_directory / filename
            )

            if not cv2.imwrite(
                str(output_path),
                mask,
            ):
                raise RuntimeError(
                    f"Could not save {output_path}"
                )

    # --------------------------------------------------------
    # 12. Save annotated image
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
    # 13. Save CSV
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
    # 14. Print results
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
        f"  Right ear: {bc_modes['right']}"
    )
    print(
        f"  Left ear:  {bc_modes['left']}"
    )

    # --------------------------------------------------------
    # 15. Manual verification list
    # --------------------------------------------------------
    verification_items = (
        collect_verification_items(
            ac_results=ac_results,
            detected_bc_results=(
                detected_bc_results
            ),
            final_bc_results=(
                final_bc_results
            ),
            bc_modes=bc_modes,
        )
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
