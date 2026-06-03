from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
except ImportError:  # pragma: no cover - compatibility with older scikit-image
    from skimage.feature import greycoprops as graycoprops
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import local_binary_pattern

from src.segmentation import read_image, resize_image, segment_image


def _safe_regionprops(mask: np.ndarray):
    labeled = (mask > 0).astype(np.uint8)
    props = regionprops(labeled)
    return props[0] if props else None


def shape_features(mask: np.ndarray) -> dict[str, float]:
    mask_u8 = (mask > 0).astype(np.uint8)
    area = float(np.count_nonzero(mask_u8))
    contours, _ = cv2.findContours(mask_u8 * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), True)) if contours else 0.0
    circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0

    prop = _safe_regionprops(mask)
    return {
        "shape_area": area,
        "shape_perimeter": perimeter,
        "shape_circularity": circularity,
        "shape_eccentricity": float(prop.eccentricity) if prop else 0.0,
        "shape_solidity": float(prop.solidity) if prop else 0.0,
        "shape_extent": float(prop.extent) if prop else 0.0,
    }


def hu_features(mask: np.ndarray) -> dict[str, float]:
    moments = cv2.moments((mask > 0).astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    logged = [-math.copysign(math.log10(abs(value) + 1e-12), value) for value in hu]
    return {f"hu_{idx + 1}": float(value) for idx, value in enumerate(logged)}


def color_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mask_bool = mask > 0
    if not np.any(mask_bool):
        mask_bool = np.ones(mask.shape, dtype=bool)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features: dict[str, float] = {}
    for prefix, data, channels in [
        ("rgb", rgb, ("r", "g", "b")),
        ("hsv", hsv, ("h", "s", "v")),
    ]:
        pixels = data[mask_bool]
        means = pixels.mean(axis=0)
        stds = pixels.std(axis=0)
        for idx, channel in enumerate(channels):
            features[f"{prefix}_{channel}_mean"] = float(means[idx])
            features[f"{prefix}_{channel}_std"] = float(stds[idx])
    return features


def _crop_to_mask(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image, mask
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def texture_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    crop, crop_mask = _crop_to_mask(image, mask)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if np.any(crop_mask > 0):
        gray = gray.copy()
        gray[crop_mask == 0] = 0

    quantized = np.floor(gray / 32).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )
    features = {
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_energy": float(graycoprops(glcm, "energy").mean()),
        "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
    }

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    values = lbp[crop_mask > 0] if np.any(crop_mask > 0) else lbp.ravel()
    hist, _ = np.histogram(values, bins=np.arange(0, 11), range=(0, 10), density=True)
    for idx, value in enumerate(hist):
        features[f"lbp_bin_{idx}"] = float(value)
    return features


def extract_image_features(
    image_path: str | Path,
    segmentation_method: str = "hsv",
    image_size: int = 256,
) -> dict[str, float | str]:
    image = resize_image(read_image(image_path), size=image_size)
    mask = segment_image(image, method=segmentation_method)
    row: dict[str, float | str] = {"image_path": str(image_path)}
    row.update(shape_features(mask))
    row.update(hu_features(mask))
    row.update(color_features(image, mask))
    row.update(texture_features(image, mask))
    return row


def build_feature_table(
    manifest_path: Path,
    output_x: Path,
    output_y: Path,
    segmentation_method: str = "hsv",
    image_size: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    rows = []
    labels = []
    for record in manifest.to_dict("records"):
        row = extract_image_features(
            record["image_path"],
            segmentation_method=segmentation_method,
            image_size=image_size,
        )
        row["split"] = record["split"]
        row["original_class"] = record["original_class"]
        rows.append(row)
        labels.append(
            {
                "image_path": record["image_path"],
                "split": record["split"],
                "label": record["label"],
            }
        )

    x_df = pd.DataFrame(rows)
    y_df = pd.DataFrame(labels)
    output_x.parent.mkdir(parents=True, exist_ok=True)
    output_y.parent.mkdir(parents=True, exist_ok=True)
    x_df.to_csv(output_x, index=False)
    y_df.to_csv(output_y, index=False)
    return x_df, y_df

