#!/usr/bin/env python3
"""
Trace the red (right) and blue (left) threshold lines on an audiogram
and read dB HL at each target frequency from the traced path.

Method
------
1. Separate red/blue by HSV color, limited to the calibrated graph.
2. For each color, find the best continuous left-to-right path through the
   colored pixels (dynamic programming: reward colored pixels, penalize
   steep vertical jumps). Gaps in dashed lines are bridged automatically.
3. Sample the path at each frequency's calibrated x-position and convert y to dB HL.
4. Support score = fraction of columns near that x where colored pixels
   actually lie on the path. Low support -> "Not recorded".

Outputs: measurements.csv, audiogram_traced.png, red_mask.png, blue_mask.png
Dependencies: pip install opencv-python numpy
"""

import csv
import math
from pathlib import Path

import cv2
import numpy as np

TARGET_FREQUENCIES = [250, 500, 1000, 2000, 4000, 8000]
FREQUENCY_LABELS = {250: "250 Hz", 500: "500 Hz", 1000: "1K Hz",
                    2000: "2K Hz", 4000: "4K Hz", 8000: "8K Hz"}

GRAPH_MIN_FREQUENCY, GRAPH_MAX_FREQUENCY = 125, 16000
GRAPH_MIN_DB, GRAPH_MAX_DB = -10, 130

CONFIG = {
    "image_path": r"C:\Users\natha\Downloads\Screenshot_audiogram.png",
    "output_dir": r"C:\Users\natha\Downloads\audiogram_output",
    "plot_rectangle": None,        # (left_x, top_y, right_x, bottom_y) or None to select

    "symbol_radius": None,         # None = automatic estimate

    # Path tracing
    "max_step_px": 6,              # max vertical jump per pixel column (steepest segment)
    "step_penalty": 0.1,          # higher = straighter path, lower = follows sharp turns
    "min_support": 0.60,           # below this, the value is "Not recorded"
    "support_tolerance_px": 2,     # how close mask pixels must be to the path

    "red_ranges": [((0, 70, 50), (12, 255, 255)),
                   ((165, 70, 50), (179, 255, 255))],
    "blue_ranges": [((85, 60, 40), (140, 255, 255))],
}

# ------------------------------------------------------------
# Calibration
# ------------------------------------------------------------

def select_plot_rectangle(image):
    x, y, w, h = cv2.selectROI(
        "Select INNER graph: 125 Hz/-10 dB to 16 kHz/130 dB; then press ENTER",
        image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if w <= 0 or h <= 0:
        raise RuntimeError("No audiogram plotting rectangle was selected.")
    return int(x), int(y), int(x + w), int(y + h)


def frequency_to_x(frequency, rect):
    """Frequency axis is logarithmic (base 2)."""
    x1, _, x2, _ = rect
    fraction = (math.log2(frequency / GRAPH_MIN_FREQUENCY)
                / math.log2(GRAPH_MAX_FREQUENCY / GRAPH_MIN_FREQUENCY))
    return int(round(x1 + fraction * (x2 - x1)))


def y_to_db(y, rect, step=5):
    """Convert y to dB HL, snapped to the nearest 5 dB."""
    _, y1, _, y2 = rect
    db = GRAPH_MIN_DB + (y - y1) / (y2 - y1) * (GRAPH_MAX_DB - GRAPH_MIN_DB)
    return int(round(db / step) * step)


def estimate_symbol_radius(rect):
    x1, y1, x2, y2 = rect
    return max(5, min(round(min((x2 - x1) / 140.0, (y2 - y1) / 70.0)), 18))

# ------------------------------------------------------------
# Color segmentation
# ------------------------------------------------------------

def color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    return cv2.medianBlur(mask, 3)


def separate_colors(image_bgr, rect):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    x1, y1, x2, y2 = rect
    graph = np.zeros(hsv.shape[:2], dtype=np.uint8)
    graph[y1:y2 + 1, x1:x2 + 1] = 255
    return (cv2.bitwise_and(color_mask(hsv, CONFIG["red_ranges"]), graph),
            cv2.bitwise_and(color_mask(hsv, CONFIG["blue_ranges"]), graph))

# ------------------------------------------------------------
# Line tracing
# ------------------------------------------------------------

def trace_line(mask, x_start, x_end, rect):
    """
    Find the best left-to-right path through the mask (Viterbi-style DP).
    Returns an array of y-values, one per column from x_start to x_end.
    """
    _, y1, _, y2 = rect
    height = mask.shape[0]
    max_step = int(CONFIG["max_step_px"])
    penalty = float(CONFIG["step_penalty"])
    steps = np.arange(-max_step, max_step + 1)

    # Vertical blur pulls the path toward the center of thick strokes.
    reward = cv2.GaussianBlur((mask > 0).astype(np.float32), (1, 7), 0)

    def confine(cost):
        cost[:y1] = np.inf
        cost[y2 + 1:] = np.inf
        return cost

    n = x_end - x_start + 1
    back = np.zeros((n, height), dtype=np.int8)
    cost = confine(-reward[:, x_start].copy())

    for i in range(1, n):
        cand = np.full((len(steps), height), np.inf, dtype=np.float32)
        for k, s in enumerate(steps):
            move = penalty * abs(s)
            if s >= 0:
                cand[k, s:] = cost[:height - s] + move
            else:
                cand[k, :height + s] = cost[-s:] + move
        best = cand.argmin(axis=0)
        back[i] = steps[best]
        cost = confine(cand[best, np.arange(height)] - reward[:, x_start + i])

    path = np.zeros(n, dtype=int)
    y = int(cost.argmin())
    for i in range(n - 1, 0, -1):
        path[i] = y
        y -= int(back[i, y])
    path[0] = y
    return path


def path_support(mask, path, x_start, x, radius):
    """Fraction of columns around x where colored pixels lie on the path."""
    tol = int(CONFIG["support_tolerance_px"])
    hits = total = 0
    for xx in range(x - radius, x + radius + 1):
        y = int(path[xx - x_start])
        hits += bool(mask[max(0, y - tol):y + tol + 1, xx].any())
        total += 1
    return hits / total


def read_thresholds(mask, rect, radius):
    """Trace one ear's line and sample it at each target frequency."""
    xs = {f: frequency_to_x(f, rect) for f in TARGET_FREQUENCIES}
    x_start, x_end = xs[TARGET_FREQUENCIES[0]] - radius, xs[TARGET_FREQUENCIES[-1]] + radius
    path = trace_line(mask, x_start, x_end, rect)

    results = {}
    for f, x in xs.items():
        y = int(path[x - x_start])
        support = path_support(mask, path, x_start, x, radius)
        value = y_to_db(y, rect) if support >= CONFIG["min_support"] else "Not recorded"
        results[f] = {"value": value, "support": support, "center": (x, y)}
    return results, (x_start, path)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

HEADERS = ["Frequency", "Right Ear (dB HL)", "Right Support", "Left Ear (dB HL)", "Left Support"]


def build_rows(results):
    return [[FREQUENCY_LABELS[f],
             results["right"][f]["value"], f"{results['right'][f]['support']:.2f}",
             results["left"][f]["value"], f"{results['left'][f]['support']:.2f}"]
            for f in TARGET_FREQUENCIES]


def print_table(rows):
    text = [HEADERS] + [[str(v) for v in row] for row in rows]
    widths = [max(len(r[i]) for r in text) for i in range(len(HEADERS))]
    fmt = lambda r: " | ".join(v.ljust(w) for v, w in zip(r, widths))
    print()
    print(fmt(text[0]))
    print("-+-".join("-" * w for w in widths))
    for r in text[1:]:
        print(fmt(r))


def create_traced_image(image, rect, results, paths):
    out = image.copy()
    x1, y1, x2, y2 = rect
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)

    for ear, color, tag in (("right", (0, 0, 255), "R"), ("left", (255, 0, 0), "L")):
        x_start, path = paths[ear]
        pts = np.array([[x_start + i, y] for i, y in enumerate(path)], np.int32)
        cv2.polylines(out, [pts], False, (0, 220, 255), 1, cv2.LINE_AA)  # traced path

        for f, res in results[ear].items():
            if not isinstance(res["value"], int):
                continue
            cx, cy = res["center"]
            cv2.circle(out, (cx, cy), 6, color, 1, cv2.LINE_AA)
            cv2.putText(out, f"{tag} {f}: {res['value']}", (cx + 9, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return out

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    image_path = Path(CONFIG["image_path"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path.resolve()}")

    rect = CONFIG["plot_rectangle"] or select_plot_rectangle(image)
    rect = tuple(int(v) for v in rect)

    red_mask, blue_mask = separate_colors(image, rect)
    radius = int(CONFIG["symbol_radius"] or estimate_symbol_radius(rect))

    results, paths = {}, {}
    for ear, mask in (("right", red_mask), ("left", blue_mask)):
        results[ear], paths[ear] = read_thresholds(mask, rect, radius)

    rows = build_rows(results)

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "red_mask.png"), red_mask)
    cv2.imwrite(str(out_dir / "blue_mask.png"), blue_mask)
    cv2.imwrite(str(out_dir / "audiogram_traced.png"),
                create_traced_image(image, rect, results, paths))
    with open(out_dir / "measurements.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADERS)
        writer.writerows(rows)

    print_table(rows)
    print(f"\nPlot rectangle: {rect}  |  Symbol radius: {radius} px")
    print("\nMeasurements requiring manual verification")
    flagged = [f"{ear.title()} {f} Hz: support={r['support']:.2f}"
               for ear in ("right", "left") for f, r in results[ear].items()
               if not isinstance(r["value"], int)]
    for item in flagged or ["None flagged by the automated checks."]:
        print(f"  - {item}")
    print(f"\nResults directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
