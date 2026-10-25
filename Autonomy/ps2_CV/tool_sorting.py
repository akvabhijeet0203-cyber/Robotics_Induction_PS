"""Explainable tool sorting demo using color segmentation and contour features.

The detector is a teaching baseline, not a robot safety system. Its generated demo
images use distinct object colors to make segmentation reproducible. Real images
may need calibrated color ranges, improved segmentation, or instance segmentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Detection:
    label: str
    center_px: tuple[int, int]
    angle_deg: float
    area_px: float
    template_distance: float
    solidity: float
    aspect_ratio: float
    grasp_px: tuple[int, int]
    bbox_xywh: tuple[int, int, int, int]


def _draw_wrench(image: np.ndarray, center: tuple[int, int], color: tuple[int, int, int], angle: float = 0) -> None:
    """Draw a recognizable simplified open-ended wrench on a transparent layer."""
    h, w = image.shape[:2]
    layer = np.zeros((h, w, 4), np.uint8)
    cv2.rectangle(layer, (center[0] - 12, center[1] - 75), (center[0] + 12, center[1] + 75), (*color, 255), -1)
    cv2.circle(layer, (center[0], center[1] - 68), 34, (*color, 255), -1)
    cv2.rectangle(layer, (center[0] - 11, center[1] - 104), (center[0] + 11, center[1] - 63), (0, 0, 0, 0), -1)
    cv2.circle(layer, (center[0], center[1] + 70), 20, (*color, 255), -1)
    cv2.circle(layer, (center[0], center[1] + 70), 8, (0, 0, 0, 0), -1)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(layer, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    alpha = rotated[:, :, 3:4] / 255.0
    image[:] = (image * (1 - alpha) + rotated[:, :, :3] * alpha).astype(np.uint8)


def _draw_caliper(image: np.ndarray, center: tuple[int, int], color: tuple[int, int, int], angle: float = 0) -> None:
    h, w = image.shape[:2]
    layer = np.zeros((h, w, 4), np.uint8)
    # Main beam, fixed jaw, movable jaw, and depth rod.
    cv2.rectangle(layer, (center[0] - 82, center[1] - 8), (center[0] + 82, center[1] + 8), (*color, 255), -1)
    cv2.rectangle(layer, (center[0] - 82, center[1] - 48), (center[0] - 70, center[1] + 8), (*color, 255), -1)
    cv2.rectangle(layer, (center[0] + 62, center[1] - 30), (center[0] + 74, center[1] + 8), (*color, 255), -1)
    cv2.rectangle(layer, (center[0] - 24, center[1] + 8), (center[0] + 16, center[1] + 30), (*color, 255), -1)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(layer, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    alpha = rotated[:, :, 3:4] / 255.0
    image[:] = (image * (1 - alpha) + rotated[:, :, :3] * alpha).astype(np.uint8)


def _draw_cable(image: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    cv2.polylines(image, [np.array(points, np.int32)], False, color, 13, cv2.LINE_AA)
    cv2.circle(image, points[0], 7, color, -1, cv2.LINE_AA)
    cv2.circle(image, points[-1], 7, color, -1, cv2.LINE_AA)


def create_demo_images(output_dir: Path) -> list[Path]:
    """Generate two transparent, reproducible, illustrative tool-layout images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bg = (242, 242, 238)
    wrench = (30, 55, 220)   # BGR red
    caliper = (210, 100, 20)  # BGR blue
    cable = (20, 160, 35)    # BGR green
    paths = []

    first = np.full((480, 720, 3), bg, np.uint8)
    _draw_wrench(first, (170, 225), wrench, 22)
    _draw_caliper(first, (425, 220), caliper, -12)
    _draw_cable(first, [(560, 345), (620, 300), (660, 350), (615, 405), (560, 365)], cable)
    p1 = output_dir / "demo_01_separated_tools.png"
    cv2.imwrite(str(p1), first)
    paths.append(p1)

    second = np.full((480, 720, 3), bg, np.uint8)
    _draw_caliper(second, (355, 220), caliper, 28)
    _draw_wrench(second, (360, 260), wrench, -25)
    _draw_cable(second, [(150, 350), (220, 300), (275, 360), (210, 410)], cable)
    p2 = output_dir / "demo_02_overlapping_tools.png"
    cv2.imwrite(str(p2), second)
    paths.append(p2)
    return paths


def _color_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    """Make color-region proposals for the demo; color does not assign labels."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Hue ranges are OpenCV's 0..179 scale. Saturation/value bounds suppress the
    # near-gray table background. Red wraps around hue zero.
    ranges = {
        "red_proposal": [((0, 70, 50), (12, 255, 255)), ((168, 70, 50), (179, 255, 255))],
        "blue_proposal": [((95, 70, 40), (132, 255, 255))],
        "green_proposal": [((35, 65, 40), (88, 255, 255))],
    }
    masks = {}
    for label, limits in ranges.items():
        mask = np.zeros(image.shape[:2], np.uint8)
        for low, high in limits:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8)))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        masks[label] = mask
    return masks


def _grabcut_mask(image: np.ndarray) -> np.ndarray:
    """Estimate foreground against a mostly visible border background."""
    height, width = image.shape[:2]
    if min(height, width) < 8:
        raise ValueError("GrabCut requires an image at least 8 pixels in each dimension")
    pad_x = max(1, int(width * 0.015))
    pad_y = max(1, int(height * 0.015))
    rect = (pad_x, pad_y, width - 2 * pad_x, height - 2 * pad_y)
    labels = np.zeros((height, width), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, labels, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    foreground = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)


def _normalized_shape(contour: np.ndarray, size: int = 128) -> np.ndarray:
    """Render a contour centered and scaled into a fixed square binary mask."""
    x, y, width, height = cv2.boundingRect(contour)
    pad = 10
    scale = min((size - 2 * pad) / max(width, 1), (size - 2 * pad) / max(height, 1))
    points = contour.reshape(-1, 2).astype(np.float32)
    points[:, 0] = (points[:, 0] - x) * scale
    points[:, 1] = (points[:, 1] - y) * scale
    points[:, 0] += (size - width * scale) / 2
    points[:, 1] += (size - height * scale) / 2
    normalized = np.zeros((size, size), np.uint8)
    cv2.fillPoly(normalized, [np.round(points).astype(np.int32)], 255)
    return normalized


def _shape_templates() -> dict[str, np.ndarray]:
    """Build deterministic normalized silhouettes used by shape matching."""
    canvas_size = 240
    result: dict[str, np.ndarray] = {}
    for label, draw in (("wrench", _draw_wrench), ("caliper", _draw_caliper)):
        canvas = np.zeros((canvas_size, canvas_size, 3), np.uint8)
        draw(canvas, (120, 120), (255, 255, 255), 0)
        mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result[label] = _normalized_shape(max(contours, key=cv2.contourArea))
    cable = np.zeros((canvas_size, canvas_size, 3), np.uint8)
    _draw_cable(cable, [(67, 112), (127, 67), (167, 117), (122, 172), (67, 132)], (255, 255, 255))
    cable = cv2.cvtColor(cable, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(cable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result["cable"] = _normalized_shape(max(contours, key=cv2.contourArea))
    return result


_SHAPE_TEMPLATES = _shape_templates()
_ROTATED_SHAPE_TEMPLATES = {
    label: [cv2.warpAffine(mask, cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2), angle, 1.0), mask.shape[::-1], flags=cv2.INTER_NEAREST)
            for angle in range(0, 180, 5)]
    for label, mask in _SHAPE_TEMPLATES.items()
}


def _classify_shape(contour: np.ndarray) -> tuple[str, float]:
    signature = _normalized_shape(contour)
    distances = {}
    for label, variants in _ROTATED_SHAPE_TEMPLATES.items():
        best_iou = 0.0
        for rotated in variants:
            intersection = cv2.countNonZero(cv2.bitwise_and(signature, rotated))
            union = cv2.countNonZero(cv2.bitwise_or(signature, rotated))
            best_iou = max(best_iou, intersection / union if union else 0.0)
        distances[label] = 1.0 - best_iou
    label = min(distances, key=distances.get)
    distance = distances[label]
    return (label if distance <= 0.62 else "unknown"), distance


def _watershed_masks(image: np.ndarray, foreground: np.ndarray) -> dict[str, np.ndarray]:
    """Use distance-transform peaks to propose splits for touching foreground."""
    distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    if not np.any(distance):
        return {"foreground": foreground}
    local_max = cv2.dilate(distance, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    peak_mask = ((distance >= 0.48 * float(distance.max())) & (distance >= local_max - 1e-5)).astype(np.uint8)
    count, components = cv2.connectedComponents(peak_mask)
    # If only one foreground seed exists, splitting would be arbitrary.
    if count <= 2:
        return {"foreground": foreground}
    markers = np.ones(foreground.shape, dtype=np.int32)
    markers[foreground > 0] = 0
    markers[peak_mask > 0] = components[peak_mask > 0].astype(np.int32) + 1
    cv2.watershed(image.copy(), markers)
    instances = {}
    instance_id = 1
    for marker_id in range(2, count + 1):
        instance = np.where(markers == marker_id, 255, 0).astype(np.uint8)
        if cv2.countNonZero(instance) >= 100:
            instances[f"instance_{instance_id}"] = instance
            instance_id += 1
    return instances or {"foreground": foreground}


def _orientation_and_grasp(contour: np.ndarray, center: tuple[int, int], label: str) -> tuple[float, tuple[int, int]]:
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) >= 5:
        mean, eigenvectors, _ = cv2.PCACompute2(points, mean=None)
        axis = eigenvectors[0]
        angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    else:
        angle = 0.0
    # A conservative visual candidate: centroid for compact shapes; for long
    # tools, center of the bounding box. This is a candidate, not a safe grasp.
    x, y, w, h = cv2.boundingRect(contour)
    grasp = (int(x + w / 2), int(y + h / 2))
    if label == "cable":
        grasp = center
    return angle, grasp


def detect_tools(image: np.ndarray, min_area: float = 140.0, segmentation: str = "hsv") -> tuple[list[Detection], dict[str, np.ndarray]]:
    """Segment candidate regions, classify shape, and estimate image geometry.

    ``hsv`` is the controlled demo mode (three color proposals); ``grabcut`` is
    a general-purpose classical alternative for a visible, mostly uniform
    background; ``watershed`` adds marker-based splitting proposals. The class
    is selected by normalized binary-template overlap over rotations.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a decoded three-channel BGR image")
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    if segmentation == "hsv":
        masks = _color_masks(blurred)
    elif segmentation in ("grabcut", "watershed"):
        masks = {"foreground": _grabcut_mask(blurred)}
        if segmentation == "watershed":
            masks = _watershed_masks(blurred, masks["foreground"])
    else:
        raise ValueError("segmentation must be 'hsv' or 'grabcut'")
    detections: list[Detection] = []
    for mask in masks.values():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
            label, template_distance = _classify_shape(contour)
            angle, grasp = _orientation_and_grasp(contour, center, label)
            x, y, w, h = cv2.boundingRect(contour)
            perimeter = cv2.arcLength(contour, True)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area else 0.0
            aspect_ratio = max(w, h) / max(1, min(w, h))
            detections.append(Detection(label, center, angle, area, template_distance, solidity, aspect_ratio, grasp, (x, y, w, h)))
    return detections, masks


def visualize(image: np.ndarray, detections: Iterable[Detection], masks: dict[str, np.ndarray], output_path: Path) -> None:
    detections = list(detections)
    # Use restrained neutral annotation so labels do not compete with the source image.
    colors = {"wrench": (65, 65, 65), "caliper": (65, 65, 65), "cable": (65, 65, 65), "unknown": (65, 65, 65)}
    overlay = image.copy()
    for item in detections:
        x, y, w, h = item.bbox_xywh
        color = colors[item.label]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.circle(overlay, item.center_px, 5, (0, 0, 0), -1)
        cv2.drawMarker(overlay, item.grasp_px, (35, 35, 35), cv2.MARKER_CROSS, 18, 2)
        text = f"{item.label} ({item.center_px[0]}, {item.center_px[1]})"
        cv2.putText(overlay, text, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("Input image")
    axes[0, 1].imshow(gray, cmap="gray"); axes[0, 1].set_title("Grayscale")
    axes[0, 2].imshow(edges, cmap="gray"); axes[0, 2].set_title("Canny edges")
    mask_items = list(masks.items())
    for ax, (name, mask) in zip((axes[0, 3], axes[1, 0], axes[1, 1]), mask_items):
        ax.imshow(mask, cmap="gray"); ax.set_title(f"Foreground: {name}")
    axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); axes[1, 2].set_title("Contours, centers, grasp candidates")
    axes[1, 3].axis("off")
    if detections:
        rows = [[d.label, str(d.center_px), f"{d.angle_deg:.1f}°", str(d.grasp_px), f"{d.template_distance:.2f}"] for d in detections]
        table = axes[1, 3].table(cellText=rows, colLabels=["Shape class", "Center px", "Axis", "Grasp px", "Template error"], loc="center")
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.0, 1.5)
    axes[1, 3].set_title("Estimated image coordinates")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("PS2 tool sorting: classical CV baseline", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def pixel_to_table(point_px: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    """Map an image point to planar table coordinates using a calibrated H."""
    homography = np.asarray(homography, dtype=float)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("Homography must be a finite 3x3 matrix")
    p = np.array([point_px[0], point_px[1], 1.0], dtype=float)
    q = homography @ p
    if abs(q[2]) < 1e-12:
        raise ValueError("Homography maps point to infinity")
    return float(q[0] / q[2]), float(q[1] / q[2])


def estimate_homography(image_points: np.ndarray, table_points: np.ndarray) -> np.ndarray:
    """Estimate pixel-to-table mapping from >=4 corresponding planar points."""
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    table_points = np.asarray(table_points, dtype=np.float32).reshape(-1, 2)
    if image_points.shape != table_points.shape or len(image_points) < 4:
        raise ValueError("Provide at least four paired image/table points")
    if not np.isfinite(image_points).all() or not np.isfinite(table_points).all():
        raise ValueError("Calibration points must be finite")
    homography, inliers = cv2.findHomography(image_points, table_points, cv2.RANSAC, 3.0)
    if homography is None or inliers is None or int(inliers.sum()) < 4:
        raise ValueError("Could not estimate a valid homography from the correspondences")
    return homography


def table_to_robot(point_table: tuple[float, float], transform: np.ndarray) -> tuple[float, float, float]:
    """Apply a calibrated 4x4 planar table-to-robot rigid transform."""
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Table-to-robot transform must be a finite 4x4 matrix")
    point = transform @ np.array([point_table[0], point_table[1], 0.0, 1.0])
    if abs(point[3]) < 1e-12:
        raise ValueError("Transform maps point to infinity")
    point = point / point[3]
    return float(point[0]), float(point[1]), float(point[2])


def cartesian_to_polar(point_xy: tuple[float, float]) -> tuple[float, float]:
    """Convert a planar robot-relative point to distance and bearing radians."""
    x, y = float(point_xy[0]), float(point_xy[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Coordinates must be finite")
    return math.hypot(x, y), math.atan2(y, x)


def match_template_variants(
    image: np.ndarray,
    template: np.ndarray,
    scales: Iterable[float] = (0.8, 1.0, 1.2),
    angles: Iterable[float] = range(0, 360, 30),
    threshold: float = 0.72,
    max_results: int = 10,
) -> list[dict[str, object]]:
    """Find rotated/scaled template candidates with normalized correlation.

    Returns bounding boxes and similarity scores; this is an auxiliary candidate
    matcher, not the color-based labeler used in the generated demo. Searching
    multiple variants is slower, and partial occlusion can still lower the score.
    """
    if image is None or template is None or image.size == 0 or template.size == 0:
        raise ValueError("Image and template must be non-empty")
    src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    base = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template
    src = cv2.GaussianBlur(src, (3, 3), 0)
    base = cv2.GaussianBlur(base, (3, 3), 0)
    found: list[dict[str, object]] = []
    for scale in scales:
        if scale <= 0:
            continue
        scaled = cv2.resize(base, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        for angle in angles:
            h, w = scaled.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
            cos_a, sin_a = abs(matrix[0, 0]), abs(matrix[0, 1])
            out_w, out_h = int(h * sin_a + w * cos_a), int(h * cos_a + w * sin_a)
            matrix[0, 2] += out_w / 2 - w / 2
            matrix[1, 2] += out_h / 2 - h / 2
            rotated = cv2.warpAffine(scaled, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            th, tw = rotated.shape[:2]
            if th > src.shape[0] or tw > src.shape[1] or min(th, tw) < 3:
                continue
            scores = cv2.matchTemplate(src, rotated, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(scores)
            if score >= threshold:
                found.append({"bbox_xywh": (location[0], location[1], tw, th), "score": float(score), "scale": float(scale), "angle_deg": float(angle)})
    # Non-maximum suppression across highly overlapping variant detections.
    found.sort(key=lambda item: float(item["score"]), reverse=True)
    kept: list[dict[str, object]] = []
    for candidate in found:
        x, y, w, h = candidate["bbox_xywh"]
        overlap = False
        for prior in kept:
            px, py, pw, ph = prior["bbox_xywh"]
            ix, iy = max(x, px), max(y, py)
            iw, ih = max(0, min(x + w, px + pw) - ix), max(0, min(y + h, py + ph) - iy)
            intersection = iw * ih
            union = w * h + pw * ph - intersection
            if union and intersection / union > 0.35:
                overlap = True
                break
        if not overlap:
            kept.append(candidate)
        if len(kept) >= max_results:
            break
    return kept


def process_image(image_path: Path, output_dir: Path, homography_path: Path | None = None, segmentation: str = "hsv") -> list[Detection]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    detections, masks = detect_tools(image, segmentation=segmentation)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image = output_dir / f"{image_path.stem}_analysis.png"
    visualize(image, detections, masks, output_image)
    homography = np.load(homography_path) if homography_path else None
    records = []
    for d in detections:
        record = asdict(d)
        record["image"] = image_path.name
        if homography is not None:
            record["table_xy"] = pixel_to_table(d.grasp_px, homography)
        records.append(record)
    (output_dir / f"{image_path.stem}_detections.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (output_dir / f"{image_path.stem}_detections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()) if records else ["image", "label", "center_px", "angle_deg", "grasp_px"])
        writer.writeheader()
        writer.writerows(records)
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Input BGR-compatible image path")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--make-demo", action="store_true", help="Generate two illustrative images and analyze them")
    parser.add_argument("--homography", type=Path, help="Optional .npy 3x3 image-to-table calibration matrix")
    parser.add_argument("--segmentation", choices=("hsv", "grabcut", "watershed"), default="hsv", help="HSV demo proposals, GrabCut foreground, or GrabCut with watershed splitting proposals")
    args = parser.parse_args()
    if args.make_demo:
        input_paths = create_demo_images(Path(__file__).resolve().parent / "demo_images")
    elif args.input:
        input_paths = [args.input]
    else:
        parser.error("Provide --input IMAGE or --make-demo")
    for input_path in input_paths:
        detections = process_image(input_path, args.output_dir, args.homography, args.segmentation)
        print(f"{input_path}: {len(detections)} candidate(s)")
        for d in detections:
            print(f"  {d.label}: center_px={d.center_px}, grasp_px={d.grasp_px}, angle={d.angle_deg:.1f} deg, template_error={d.template_distance:.2f}")


if __name__ == "__main__":
    main()
