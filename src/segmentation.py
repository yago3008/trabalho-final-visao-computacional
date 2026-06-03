from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def resize_image(image: np.ndarray, size: int = 256) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def clean_mask(mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return keep_largest_component(mask)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    largest = max(contours, key=cv2.contourArea)
    output = np.zeros_like(mask)
    cv2.drawContours(output, [largest], -1, 255, thickness=cv2.FILLED)
    return output


def segment_hsv(image: np.ndarray) -> np.ndarray:
    """Segment fruit from a mostly light background using saturation/value cues."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation > 25) | (value < 245)).astype(np.uint8) * 255
    return clean_mask(mask)


def _valid_area_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / float(mask.size)


def segment_otsu(image: np.ndarray) -> np.ndarray:
    """Try Otsu and inverse Otsu, selecting the most plausible foreground."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, inverse = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    candidates = [clean_mask(binary), clean_mask(inverse)]
    scored: list[tuple[float, np.ndarray]] = []
    for mask in candidates:
        ratio = _valid_area_ratio(mask)
        # Prefer masks that are neither tiny nor the whole frame.
        penalty = 0.0 if 0.03 <= ratio <= 0.90 else 1.0
        score = abs(ratio - 0.35) + penalty
        scored.append((score, mask))
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def segment_image(image: np.ndarray, method: str = "hsv") -> np.ndarray:
    if method == "hsv":
        return segment_hsv(image)
    if method == "otsu":
        return segment_otsu(image)
    raise ValueError(f"Unknown segmentation method: {method}")


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    color = np.zeros_like(image)
    color[:, :, 1] = 255
    mask_bool = mask > 0
    overlay = image.copy()
    overlay[mask_bool] = cv2.addWeighted(
        image[mask_bool],
        1.0 - alpha,
        color[mask_bool],
        alpha,
        0,
    )
    return overlay


def compare_segmentations(image: np.ndarray) -> np.ndarray:
    hsv_mask = segment_hsv(image)
    otsu_mask = segment_otsu(image)
    hsv_overlay = overlay_mask(image, hsv_mask)
    otsu_overlay = overlay_mask(image, otsu_mask)
    labels = [
        ("original", image),
        ("hsv", hsv_overlay),
        ("otsu", otsu_overlay),
    ]
    panels = []
    for text, panel in labels:
        panel = panel.copy()
        cv2.putText(
            panel,
            text,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            text,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    return np.hstack(panels)

