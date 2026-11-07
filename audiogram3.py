#!/usr/bin/env python3
"""
Extract bone-conduction (BC) thresholds from a color-coded audiogram.

Supported standard symbols
--------------------------
Right ear, red:    <  unmasked BC      [  masked BC
Left ear, blue:    >  unmasked BC      ]  masked BC

Method
------
1. Separate red (right) and blue (left) markings by HSV color.
2. At each target frequency, test every 5 dB step for each allowed symbol.
3. A symbol is accepted only if every structural stroke is present.
   Extra colored pixels are not penalized.
4. Frequencies with no accepted symbol are reported as "Not recorded".
   Frequencies with competing detections are reported as "Unclear".

Outputs
-------
    measurements.csv, audiogram_annotated.png, red_mask.png, blue_mask.png

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
# CALIBRATION AND SYMBOLS
# ============================================================

TARGET_FREQUENCIES = [250, 500, 1000, 2000, 4000, 8000]
FREQUENCY_LABELS = {250: "250 Hz", 500: "500 Hz", 1000: "1K Hz",
                    2000: "2K Hz", 4000: "4K Hz", 8000: "8K Hz"}

# Full calibrated graph range.
GRAPH_MIN_FREQUENCY, GRAPH_MAX_FREQUENCY = 125, 16000
GRAPH_MIN_DB, GRAPH_MAX_DB = -10, 130

# Thresholds are evaluated only at conventional 5 dB steps.
DB_CANDIDATES = list(range(GRAPH_MIN_DB, GRAPH_MAX_DB + 1, 5))

SYMBOL_DISPLAY_NAMES = {"LESS_THAN": "<", "GREATER_THAN": ">",
                        "LEFT_BRACKET": "[", "RIGHT_BRACKET": "]"}

# Allowed BC symbols per ear (unmasked, masked).
EAR_SYMBOLS = {
    "right": ["LESS_THAN", "LEFT_BRACKET"],
    "left": ["GREATER_THAN", "RIGHT_BRACKET"],
}

# ============================================================
# USER CONFIGURATION
# ============================================================

CONFIG = {
    "image_path": r"C:\Users\natha\Downloads\Screenshot_audiogram.png",
    "output_dir": r"C:\Users\natha\Downloads\audiogram_output",

    # Inner graph rectangle (left_x, top_y, right_x, bottom_y), corresponding to
    # 125 Hz / -10 dB HL (top-left) and 16 kHz / 130 dB HL (bottom-right).
    # Keep as None to select it interactively. Example: (210, 85, 1450, 930)
    "plot_rectangle": None,

    # Approximate symbol radius in pixels. None = automatic estimate.
    "symbol_radius": None,

    # Minimum structural score. Increase to reduce false positives;
    # decrease if real symbols are missed.
    "bc_min_score": 0.62,

    # If a second nonadjacent dB candidate scores nearly as well as the best,
    # the result is marked Unclear.
    "ambiguity_ratio": 0.94,

    # HSV color ranges (OpenCV hue range is 0-179; red wraps around).
    "red_ranges": [((0, 70, 50), (12, 255, 255)),
                   ((165, 70, 50), (179, 255, 255))],
    "blue_ranges": [((85, 60, 40), (140, 255, 255))],

    # Output options.
    "save_masks": True,
    "save_annotated_image": True,
    "save_csv": True,
}

# ============================================================
# GRAPH SELECTION AND CALIBRATION
# ============================================================

def select_plot_rectangle(image):
    """Interactively select the inner audiogram plotting rectangle."""
    x, y, w, h = cv2.selectROI(
        "Select INNER graph: 125 Hz/-10 dB to 16 kHz/130 dB; then press ENTER",
        image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if w <= 0 or h <= 0:
        raise RuntimeError("No audiogram plotting rectangle was selected.")
    return int(x), int(y), int(x + w), int(y + h)


def validate_rectangle(rect, image_shape):
    """Confirm that the graph rectangle is inside the image."""
    x1, y1, x2, y2 = rect
    height, width = image_shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"Plot rectangle {rect} is outside image dimensions {width}x{height}.")


def frequency_to_x(frequency, rect):
    """Convert frequency to x (the frequency axis is logarithmic, base 2)."""
    x1, _, x2, _ = rect
    fraction = (math.log2(frequency / GRAPH_MIN_FREQUENCY)
                / math.log2(GRAPH_MAX_FREQUENCY / GRAPH_MIN_FREQUENCY))
    return x1 + fraction * (x2 - x1)


def db_to_y(db_hl, rect):
    """Convert dB HL to y (larger dB HL values appear lower on the graph)."""
    _, y1, _, y2 = rect
    return y1 + (db_hl - GRAPH_MIN_DB) / (GRAPH_MAX_DB - GRAPH_MIN_DB) * (y2 - y1)


def estimate_symbol_radius(rect):
    """Estimate the symbol radius from the graph dimensions."""
    x1, y1, x2, y2 = rect
    estimated = round(min((x2 - x1) / 140.0, (y2 - y1) / 70.0))
    return max(5, min(estimated, 18))

# ============================================================
# COLOR SEGMENTATION
# ============================================================

def color_mask(hsv, ranges):
    """Combine one or more HSV ranges into a denoised binary mask."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    # Suppress isolated noise while retaining symbol strokes.
    return cv2.medianBlur(mask, 3)


def separate_colors(image_bgr, rect):
    """Produce red and blue masks limited to the calibrated graph area."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    x1, y1, x2, y2 = rect

    graph = np.zeros(hsv.shape[:2], dtype=np.uint8)
    graph[y1:y2 + 1, x1:x2 + 1] = 255

    return (cv2.bitwise_and(color_mask(hsv, CONFIG["red_ranges"]), graph),
            cv2.bitwise_and(color_mask(hsv, CONFIG["blue_ranges"]), graph))

# ============================================================
# SYMBOL TEMPLATES AND SCORING
# ============================================================

def build_symbol_strokes(radius):
    """
    Build each BC symbol as a list of required line strokes.

    Extra pixels are not penalized, so a symbol that touches other
    colored markings can still be recognized.
    """
    c = radius
    e = max(3, int(round(radius * 0.72)))
    thickness = max(1, int(round(radius * 0.16)))
    l, r, t, b = c - e, c + e, c - e, c + e

    # Each symbol: a list of (point_1, point_2) strokes.
    segments = {
        "LESS_THAN": [((r, t), (l, c)), ((l, c), (r, b))],
        "GREATER_THAN": [((l, t), (r, c)), ((r, c), (l, b))],
        "LEFT_BRACKET": [((l, t), (l, b)), ((l, t), (r, t)), ((l, b), (r, b))],
        "RIGHT_BRACKET": [((r, t), (r, b)), ((l, t), (r, t)), ((l, b), (r, b))],
    }

    def stroke(p1, p2):
        canvas = np.zeros((2 * radius + 1, 2 * radius + 1), np.uint8)
        cv2.line(canvas, p1, p2, 255, thickness, cv2.LINE_AA)
        return np.where(canvas > 40, 255, 0).astype(np.uint8)

    return {name: [stroke(*s) for s in strokes]
            for name, strokes in segments.items()}


def score_required_strokes(patch, strokes):
    """
    Score whether all required strokes are present.

    The weakest stroke is weighted strongly, so:
        - one arm alone should not become < or >;
        - one line alone should not become a bracket.
    """
    observed = cv2.dilate((patch > 0).astype(np.uint8) * 255,
                          np.ones((3, 3), np.uint8))  # 1-pixel tolerance

    recalls = [cv2.countNonZero(cv2.bitwise_and(s, observed))
               / max(1, cv2.countNonZero(s)) for s in strokes]

    return 0.60 * float(np.mean(recalls)) + 0.40 * float(np.min(recalls))


def best_position(padded, x, y, radius, strokes, h_search, v_search):
    """
    Search around the expected coordinate and return (best_score, best_center).

    The mask is pre-padded by `radius`, so patches near the image edge
    are zero-filled automatically.
    """
    size = 2 * radius + 1
    best = (0.0, None)

    for dy in range(-v_search, v_search + 1):
        for dx in range(-h_search, h_search + 1):
            cx, cy = int(round(x + dx)), int(round(y + dy))
            if cx < -radius or cy < -radius:
                continue
            patch = padded[cy + radius:cy + radius + size,
                           cx + radius:cx + radius + size]
            if patch.shape != (size, size):
                continue
            score = score_required_strokes(patch, strokes)
            if best[1] is None or score > best[0]:
                best = (score, (cx, cy))
    return best

# ============================================================
# THRESHOLD EXTRACTION
# ============================================================

def extract_threshold(padded, frequency, rect, symbols, templates, radius):
    """
    Find the strongest allowed BC symbol at one frequency.

    Different symbol types at the same dB level are alternatives, not
    competing values. Ambiguity is evaluated only between nonadjacent
    dB levels, since a real symbol can overlap the neighboring 5 dB window.
    """
    x = frequency_to_x(frequency, rect)
    h_search, v_search = max(2, radius // 2), max(1, radius // 3)

    # Best (score, symbol, center) for every dB level.
    per_level = []
    for db in DB_CANDIDATES:
        y = db_to_y(db, rect)
        score, symbol, center = max(
            ((*best_position(padded, x, y, radius, templates[s], h_search, v_search)[:1],
              s,
              best_position(padded, x, y, radius, templates[s], h_search, v_search)[1])
             for s in symbols),
            key=lambda item: item[0])
        per_level.append({"db": db, "symbol": symbol, "score": score, "center": center})

    per_level.sort(key=lambda item: item["score"], reverse=True)
    best = per_level[0]

    def result(value, symbol=None, reason=""):
        return {"value": value, "symbol": symbol, "score": best["score"],
                "center": best["center"], "reason": reason}

    if best["score"] < CONFIG["bc_min_score"]:
        return result("Not recorded",
                      reason="No complete BC symbol exceeded the minimum score.")

    competitor = next((c for c in per_level[1:] if abs(c["db"] - best["db"]) > 5), None)
    if competitor and competitor["score"] >= best["score"] * CONFIG["ambiguity_ratio"]:
        return result("Unclear",
                      reason=f"Competing complete-symbol detections at "
                             f"{best['db']} and {competitor['db']} dB HL.")

    return result(best["db"], best["symbol"])


def extract_bone_conduction(red_mask, blue_mask, rect, templates, radius):
    """Extract BC thresholds for both ears at every target frequency."""
    results = {}
    for ear, mask in (("right", red_mask), ("left", blue_mask)):
        padded = cv2.copyMakeBorder(mask, radius, radius, radius, radius,
                                    cv2.BORDER_CONSTANT, value=0)
        results[ear] = {
            f: extract_threshold(padded, f, rect, EAR_SYMBOLS[ear], templates, radius)
            for f in TARGET_FREQUENCIES}
    return results

# ============================================================
# RESULT FORMATTING
# ============================================================

def symbol_description(result):
    """Return a printable symbol name (blank if none)."""
    return SYMBOL_DISPLAY_NAMES.get(result.get("symbol"), "")


def build_output_rows(bc_results):
    """Build one BC output row per frequency."""
    return [[f"BC {FREQUENCY_LABELS[f]} (dB HL)",
             bc_results["right"][f]["value"], symbol_description(bc_results["right"][f]),
             bc_results["left"][f]["value"], symbol_description(bc_results["left"][f])]
            for f in TARGET_FREQUENCIES]


HEADERS = ["Measurement", "Right Ear", "Right Symbol", "Left Ear", "Left Symbol"]


def write_csv(path, rows):
    """Save extracted BC measurements."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(rows)


def print_table(rows):
    """Print a dependency-free results table."""
    text = [HEADERS] + [[str(v) for v in row] for row in rows]
    widths = [max(len(row[i]) for row in text) for i in range(len(HEADERS))]
    fmt = lambda row: " | ".join(v.ljust(w) for v, w in zip(row, widths))

    print()
    print(fmt(text[0]))
    print("-+-".join("-" * w for w in widths))
    for row in text[1:]:
        print(fmt(row))

# ============================================================
# ANNOTATION AND VERIFICATION
# ============================================================

def create_annotated_image(image, rect, bc_results):
    """Create the verification image: graph outline plus each accepted BC threshold."""
    annotated = image.copy()
    x1, y1, x2, y2 = rect
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)

    for ear, color, tag in (("right", (0, 0, 255), "R"), ("left", (255, 0, 0), "L")):
        for f, res in bc_results[ear].items():
            if not isinstance(res["value"], int) or res["center"] is None:
                continue
            cx, cy = res["center"]
            cv2.circle(annotated, (cx, cy), 6, color, 1, cv2.LINE_AA)
            cv2.putText(annotated, f"{tag}-BC {f}: {res['value']} {symbol_description(res)}",
                        (cx + 9, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1,
                        cv2.LINE_AA)
    return annotated


def collect_verification_items(bc_results):
    """List every frequency without an accepted BC value for manual review."""
    return [f"{ear.title()} BC, {f} Hz: {res['value']}; {res['reason']} "
            f"(best score={res['score']:.3f})"
            for ear in ("right", "left")
            for f, res in bc_results[ear].items()
            if not isinstance(res["value"], int)]

# ============================================================
# MAIN
# ============================================================

def main():
    # 1. Read the input image.
    image_path = Path(CONFIG["image_path"])
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path.resolve()}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode the image: {image_path.resolve()}")
    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]} pixels")

    # 2. Select or load the graph rectangle.
    rect = CONFIG["plot_rectangle"]
    if rect is None:
        print("\nSelect the inner audiogram plotting rectangle.\n"
              "Upper-left:  125 Hz, -10 dB HL\nLower-right: 16 kHz, 130 dB HL\n")
        rect = select_plot_rectangle(image)
    else:
        if len(rect) != 4:
            raise ValueError("CONFIG['plot_rectangle'] must contain (x1, y1, x2, y2).")
        rect = tuple(int(v) for v in rect)
    validate_rectangle(rect, image.shape)

    # 3. Separate red and blue markings.
    red_mask, blue_mask = separate_colors(image, rect)

    # 4. Determine the symbol radius and build templates.
    radius = CONFIG["symbol_radius"] or estimate_symbol_radius(rect)
    if int(radius) < 3:
        raise ValueError("CONFIG['symbol_radius'] must be at least 3 pixels.")
    radius = int(radius)
    templates = build_symbol_strokes(radius)

    # 5. Extract BC thresholds.
    bc_results = extract_bone_conduction(red_mask, blue_mask, rect, templates, radius)
    rows = build_output_rows(bc_results)

    # 6. Save outputs.
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_image(name, img):
        if not cv2.imwrite(str(out_dir / name), img):
            raise RuntimeError(f"Could not save {out_dir / name}")

    if CONFIG["save_masks"]:
        save_image("red_mask.png", red_mask)
        save_image("blue_mask.png", blue_mask)
    if CONFIG["save_annotated_image"]:
        save_image("audiogram_annotated.png",
                   create_annotated_image(image, rect, bc_results))
    if CONFIG["save_csv"]:
        write_csv(out_dir / "measurements.csv", rows)

    # 7. Print results and manual-verification list.
    print_table(rows)
    print(f"\nCalibration\n  Plot rectangle: {rect}\n  Symbol radius:  {radius} pixels")

    print("\nMeasurements requiring manual verification")
    items = collect_verification_items(bc_results)
    for item in items or ["None flagged by the automated checks."]:
        print(f"  - {item}")

    print(f"\nResults directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
