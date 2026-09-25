#!/usr/bin/env python3
"""Python version of the attached CellProfiler 3D measurement pipeline.

The workflow mirrors the provided .cppipe:

1. Load z-slice TIFF stacks for brightfield, green/FUCCI-1, and red/FUCCI-2.
2. Rescale the Cellpose mask by its maximum, then convert it to labeled objects.
3. Save colored object-outline overlays on green and red channels.
4. Measure object intensity, 3D size/shape, and object GLCM texture.
5. Export object-level features and per-image summaries as spreadsheets.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import QhullError
from skimage import exposure, measure, segmentation, util
from skimage.morphology import convex_hull_image


DEFAULT_IMAGE_DIR = Path(
    "/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/"
    "NCI-N87-combined/A01_rawData"
)
DEFAULT_MASK_DIR = Path(
    "/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/"
    "NCI-N87-combined/A04_CellposeOutput"
)

CHANNELS = {
    "bright": "01",
    "green": "00",
    "red": "02",
}

OVERLAY_COLORS = {
    "green": (255, 0, 236),  # CellProfiler #FF00EC on fluor_1/green
    "red": (57, 255, 0),  # CellProfiler #39FF00 on fluor_2/red
}

TEXTURE_FEATURES = (
    "AngularSecondMoment",
    "Contrast",
    "Correlation",
    "Variance",
    "InverseDifferenceMoment",
    "SumAverage",
    "SumVariance",
    "SumEntropy",
    "Entropy",
    "DifferenceVariance",
    "DifferenceEntropy",
    "InfoMeas1",
    "InfoMeas2",
)


@dataclass(frozen=True)
class ImageSet:
    stem: str
    image_dir: Path
    mask_file: Path
    channel_files: dict[str, list[Path]]
    fof: str
    date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Python 3D image analysis pipeline equivalent to the attached CellProfiler pipeline."
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument(
        "--input-layout",
        choices=("auto", "slices", "ome"),
        default="auto",
        help="Use z-slice folders, CellProfiler-style OME files, or auto-detect both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "outputs" / "cellprofiler_python",
    )
    parser.add_argument(
        "--fields",
        nargs="*",
        help="Optional image-set folder names to process, e.g. FoF1_241016_fucci.nucleus.",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N discovered image sets.")
    parser.add_argument(
        "--object-limit",
        type=int,
        help="Debug option: process only the first N objects in each image set.",
    )
    parser.add_argument(
        "--preserve-mask-labels",
        action="store_true",
        help="Keep Cellpose label IDs as ObjectNumber instead of sequential relabeling.",
    )
    parser.add_argument(
        "--object-conversion",
        choices=("cellprofiler", "labels", "binary-components"),
        default="cellprofiler",
        help=(
            "How to convert masks to pipeline objects. 'cellprofiler' splits connected components of each "
            "raw mask value with full 3D connectivity, matching ConvertImageToObjects with Preserve original "
            "labels = No; 'labels' treats each nonzero raw mask value as one object; 'binary-components' "
            "labels connected foreground regions regardless of raw value."
        ),
    )
    parser.add_argument(
        "--signal-core-percentile",
        type=float,
        default=99.0,
        help="Within each object, measure extra FUCCI signal-core features from voxels above this green/red percentile.",
    )
    parser.add_argument(
        "--signal-core-min-voxels",
        type=int,
        default=1,
        help="Minimum number of voxels to include in the signal-core feature region.",
    )
    parser.add_argument("--texture-distance", type=int, default=3)
    parser.add_argument("--texture-levels", type=int, default=256)
    parser.add_argument("--skip-texture", action="store_true")
    parser.add_argument("--skip-overlays", action="store_true")
    parser.add_argument(
        "--write-xlsx",
        dest="write_xlsx",
        action="store_true",
        default=True,
        help="Write .xlsx workbooks in addition to CSV exports.",
    )
    parser.add_argument(
        "--no-write-xlsx",
        dest="write_xlsx",
        action="store_false",
        help="Write only CSV exports.",
    )
    return parser.parse_args()


def discover_image_sets(image_dir: Path, mask_dir: Path, fields: Iterable[str] | None) -> list[ImageSet]:
    return discover_slice_image_sets(image_dir, mask_dir, fields)


def discover_image_sets_by_layout(
    image_dir: Path, mask_dir: Path, fields: Iterable[str] | None, input_layout: str
) -> list[ImageSet]:
    image_sets: list[ImageSet] = []
    if input_layout in {"auto", "slices"}:
        image_sets.extend(discover_slice_image_sets(image_dir, mask_dir, fields))
    if input_layout in {"auto", "ome"}:
        existing_stems = {image_set.stem for image_set in image_sets}
        for image_set in discover_ome_image_sets(image_dir, mask_dir, fields):
            if image_set.stem not in existing_stems:
                image_sets.append(image_set)
    return image_sets


def discover_slice_image_sets(image_dir: Path, mask_dir: Path, fields: Iterable[str] | None) -> list[ImageSet]:
    requested = set(fields or [])
    image_sets: list[ImageSet] = []

    for subdir in sorted(image_dir.iterdir(), key=lambda path: natural_key(path.name)):
        if not subdir.is_dir() or subdir.name.startswith("._"):
            continue
        if requested and subdir.name not in requested:
            continue

        channel_files = collect_channel_files(subdir)
        if set(channel_files) != set(CHANNELS):
            missing = sorted(set(CHANNELS) - set(channel_files))
            print(f"Skipping {subdir.name}: missing channel(s) {missing}")
            continue

        mask_file = mask_dir / subdir.name / f"{subdir.name}_masks.tif"
        if not mask_file.exists():
            candidates = sorted((mask_dir / subdir.name).glob("*_masks.tif"))
            if not candidates:
                print(f"Skipping {subdir.name}: no matching Cellpose *_masks.tif")
                continue
            mask_file = candidates[0]

        metadata = parse_field_metadata(subdir.name)
        image_sets.append(
            ImageSet(
                stem=subdir.name,
                image_dir=subdir,
                mask_file=mask_file,
                channel_files=channel_files,
                fof=metadata["fof"],
                date=metadata["date"],
            )
        )

    return image_sets


def discover_ome_image_sets(image_dir: Path, mask_dir: Path, fields: Iterable[str] | None) -> list[ImageSet]:
    requested = set(fields or [])
    image_sets: list[ImageSet] = []
    mask_roots = [mask_dir]
    if mask_dir != image_dir:
        mask_roots.append(image_dir)

    mask_files: list[Path] = []
    for root in mask_roots:
        if root.exists():
            mask_files.extend(path for path in root.glob("*_mask.ome.tif") if not path.name.startswith("._"))
            mask_files.extend(path for path in root.glob("*_masks.ome.tif") if not path.name.startswith("._"))
            mask_files.extend(path for path in root.glob("*_masks.tif") if not path.name.startswith("._"))

    for mask_file in sorted(set(mask_files), key=lambda path: natural_key(path.name)):
        stem = ome_stem_from_mask(mask_file)
        if requested and stem not in requested:
            continue
        channel_files = collect_ome_channel_files(image_dir, stem)
        if set(channel_files) != set(CHANNELS):
            missing = sorted(set(CHANNELS) - set(channel_files))
            print(f"Skipping {stem}: missing OME channel(s) {missing}")
            continue
        metadata = parse_field_metadata(stem)
        image_sets.append(
            ImageSet(
                stem=stem,
                image_dir=image_dir,
                mask_file=mask_file,
                channel_files=channel_files,
                fof=metadata["fof"],
                date=metadata["date"],
            )
        )

    return image_sets


def ome_stem_from_mask(mask_file: Path) -> str:
    name = mask_file.name
    for suffix in ("_mask.ome.tif", "_masks.ome.tif", "_masks.tif"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return mask_file.stem


def collect_ome_channel_files(image_dir: Path, stem: str) -> dict[str, list[Path]]:
    channel_files: dict[str, list[Path]] = {}
    for channel, channel_id in CHANNELS.items():
        candidates = [
            image_dir / f"{stem}_ch{channel_id}.ome.tif",
            image_dir / f"{stem}_ch{channel_id}.tif",
            image_dir / f"{stem}_channel{channel_id}.ome.tif",
        ]
        if channel == "bright":
            candidates.extend([image_dir / f"{stem}.ome.tif", image_dir / f"{stem}.tif"])
        existing = [path for path in candidates if path.exists() and not path.name.startswith("._")]
        if existing:
            channel_files[channel] = [existing[0]]
    return channel_files


def collect_channel_files(image_dir: Path) -> dict[str, list[Path]]:
    pattern = re.compile(r"_z(?P<z>\d+)_ch(?P<channel>\d+)\.tif$", re.IGNORECASE)
    by_channel: dict[str, list[tuple[int, Path]]] = {name: [] for name in CHANNELS}
    channel_lookup = {value: key for key, value in CHANNELS.items()}

    for path in image_dir.glob("*.tif"):
        if path.name.startswith("._"):
            continue
        match = pattern.search(path.name)
        if not match:
            continue
        channel = channel_lookup.get(match.group("channel"))
        if channel is None:
            continue
        by_channel[channel].append((int(match.group("z")), path))

    return {
        channel: [path for _, path in sorted(items)]
        for channel, items in by_channel.items()
        if items
    }


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def parse_field_metadata(stem: str) -> dict[str, str]:
    match = re.match(r"^FoF(?P<fof>\d+)_(?P<date>\d{6})", stem)
    if match:
        return {"fof": match.group("fof"), "date": match.group("date")}
    return {"fof": "", "date": ""}


def load_stack(paths: list[Path]) -> np.ndarray:
    if len(paths) == 1:
        image = np.squeeze(tifffile.imread(paths[0]))
        if image.ndim == 2:
            return image[np.newaxis, ...]
        if image.ndim == 3:
            return image
        raise ValueError(f"Expected 2D or 3D image stack in {paths[0]}, got shape {image.shape}")

    planes = [tifffile.imread(path) for path in paths]
    return np.stack(planes, axis=0)


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    if np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        if max_value == 0:
            return image.astype(np.float32)
        return image.astype(np.float32) / np.float32(max_value)

    image = image.astype(np.float32, copy=False)
    finite = np.isfinite(image)
    if finite.any() and image[finite].max() > 1.0:
        return exposure.rescale_intensity(image, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)
    return image


def rescale_mask_intensity(mask: np.ndarray) -> np.ndarray:
    max_value = mask.max(initial=0)
    if max_value == 0:
        return mask.astype(np.float32)
    return mask.astype(np.float32) / np.float32(max_value)


def convert_image_to_objects(mask: np.ndarray, preserve_labels: bool, object_conversion: str) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask stack, got shape {mask.shape}")

    labels = mask.astype(np.int32, copy=False)
    labels[labels < 0] = 0
    if preserve_labels:
        return labels

    if object_conversion == "labels":
        relabeled, _, _ = segmentation.relabel_sequential(labels)
        return relabeled.astype(np.int32, copy=False)

    if object_conversion == "cellprofiler":
        return measure.label(labels, background=0, connectivity=labels.ndim).astype(np.int32, copy=False)

    if object_conversion == "binary-components":
        return measure.label(labels > 0, connectivity=1).astype(np.int32, copy=False)

    raise ValueError(f"Unsupported object conversion mode: {object_conversion}")


def save_overlay(base: np.ndarray, labels: np.ndarray, color: tuple[int, int, int], output_path: Path) -> None:
    boundaries = segmentation.find_boundaries(labels, mode="inner")
    base_u8 = util.img_as_ubyte(np.clip(base, 0.0, 1.0))
    rgb = np.repeat(base_u8[..., np.newaxis], 3, axis=-1)
    rgb[boundaries] = color
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tifffile.imwrite(output_path, rgb, photometric="rgb", compression="zlib")
    except Exception:
        tifffile.imwrite(output_path, rgb, photometric="rgb")


def make_metadata_row(image_number: int, image_set: ImageSet, stacks: dict[str, np.ndarray]) -> dict[str, object]:
    row: dict[str, object] = {
        "ImageNumber": image_number,
        "Metadata_Date": image_set.date,
        "Metadata_Field": f"{image_set.fof}_{image_set.date}" if image_set.fof and image_set.date else image_set.stem,
        "FoF": image_set.fof,
        "Date": image_set.date,
        "ImageSet": image_set.stem,
        "PathName_mask": str(image_set.mask_file.parent),
        "FileName_mask": image_set.mask_file.name,
    }
    for channel, paths in image_set.channel_files.items():
        row[f"PathName_{channel}"] = str(paths[0].parent)
        row[f"FileName_{channel}"] = image_set.stem
        row[f"PlaneCount_{channel}"] = len(paths)
        row[f"Height_{channel}"] = stacks[channel].shape[1]
        row[f"Width_{channel}"] = stacks[channel].shape[2]
    return row


def measure_size_shape(
    region: measure._regionprops.RegionProperties, raw_mask: np.ndarray
) -> dict[str, object]:
    min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
    bbox_volume = (max_z - min_z) * (max_y - min_y) * (max_x - min_x)
    object_number = int(region.label)
    row: dict[str, object] = {
        "ObjectNumber": object_number,
        "CellposeLabel": cellpose_label_from_raw_mask(region, raw_mask),
        "Number_Object_Number": object_number,
        "AreaShape_BoundingBoxMinimum_X": min_x,
        "AreaShape_BoundingBoxMinimum_Y": min_y,
        "AreaShape_BoundingBoxMinimum_Z": min_z,
        "AreaShape_BoundingBoxMaximum_X": max_x,
        "AreaShape_BoundingBoxMaximum_Y": max_y,
        "AreaShape_BoundingBoxMaximum_Z": max_z,
        "AreaShape_BoundingBoxVolume": bbox_volume,
        "AreaShape_Center_X": region.centroid[2],
        "AreaShape_Center_Y": region.centroid[1],
        "AreaShape_Center_Z": region.centroid[0],
        "X": region.centroid[2],
        "Y": region.centroid[1],
        "Z": region.centroid[0],
        "AreaShape_Volume": region.area,
        "AreaShape_EquivalentDiameter": safe_property(region, "equivalent_diameter_area"),
        "AreaShape_EulerNumber": safe_property(region, "euler_number"),
        "AreaShape_Extent": safe_property(region, "extent"),
        "AreaShape_MajorAxisLength": safe_property(region, "axis_major_length"),
        "AreaShape_MinorAxisLength": safe_property(region, "axis_minor_length"),
        "AreaShape_Solidity": safe_solidity(region),
        "AreaShape_SurfaceArea": surface_area(region.image),
    }
    return row


def cellpose_label_from_raw_mask(region: measure._regionprops.RegionProperties, raw_mask: np.ndarray) -> object:
    coords = region.coords
    raw_values = raw_mask[coords[:, 0], coords[:, 1], coords[:, 2]]
    raw_values = raw_values[raw_values > 0]
    if raw_values.size == 0:
        return np.nan

    unique_raw_values = np.unique(raw_values)
    if unique_raw_values.size != 1:
        preview = ", ".join(str(int(value)) for value in unique_raw_values[:10])
        raise ValueError(
            f"Processed object {int(region.label)} contains multiple raw Cellpose labels: {preview}. "
            "Each object must come from exactly one raw Cellpose mask value."
        )

    return int(unique_raw_values[0])


def safe_property(region: measure._regionprops.RegionProperties, name: str) -> float:
    try:
        value = getattr(region, name)
    except (AttributeError, NotImplementedError, ValueError, RuntimeError, QhullError):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def safe_solidity(region: measure._regionprops.RegionProperties) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            value = float(region.solidity)
    except (AttributeError, NotImplementedError, ValueError, RuntimeError, QhullError):
        return fallback_solidity(region.image, float(region.area))
    if np.isfinite(value):
        return value
    return fallback_solidity(region.image, float(region.area))


def fallback_solidity(binary_object: np.ndarray, object_volume: float) -> float:
    coords = np.argwhere(binary_object)
    if coords.size == 0:
        return np.nan

    occupied_axes = np.flatnonzero(np.ptp(coords, axis=0) > 0)
    if occupied_axes.size <= 1:
        return 1.0

    if occupied_axes.size == 2:
        projection = binary_object.any(axis=int(np.setdiff1d(np.arange(binary_object.ndim), occupied_axes)[0]))
        try:
            convex_area = float(convex_hull_image(projection).sum())
        except (ValueError, RuntimeError, QhullError):
            return np.nan
        return object_volume / convex_area if convex_area > 0 else np.nan

    return np.nan


def surface_area(binary_object: np.ndarray) -> float:
    if binary_object.sum() == 0:
        return 0.0
    padded = np.pad(binary_object.astype(np.float32), 1, mode="constant")
    try:
        vertices, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    except ValueError:
        return 0.0
    return float(measure.mesh_surface_area(vertices, faces))


def measure_intensity(
    region: measure._regionprops.RegionProperties,
    labels: np.ndarray,
    edge_mask: np.ndarray,
    stacks: dict[str, np.ndarray],
) -> dict[str, object]:
    coords = region.coords
    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    object_edge = edge_mask[z, y, x]
    row: dict[str, object] = {}

    for channel, image in stacks.items():
        values = image[z, y, x].astype(np.float64, copy=False)
        edge_values = values[object_edge]
        prefix = f"Intensity_{{metric}}_{channel}"

        row[prefix.format(metric="IntegratedIntensity")] = float(values.sum())
        row[prefix.format(metric="MeanIntensity")] = safe_stat(values, np.mean)
        row[prefix.format(metric="StdIntensity")] = safe_stat(values, np.std)
        row[prefix.format(metric="MinIntensity")] = safe_stat(values, np.min)
        row[prefix.format(metric="MaxIntensity")] = safe_stat(values, np.max)
        row[prefix.format(metric="MedianIntensity")] = safe_stat(values, np.median)
        row[prefix.format(metric="MADIntensity")] = median_absolute_deviation(values)
        row[prefix.format(metric="LowerQuartileIntensity")] = percentile(values, 25)
        row[prefix.format(metric="UpperQuartileIntensity")] = percentile(values, 75)

        row[prefix.format(metric="IntegratedIntensityEdge")] = float(edge_values.sum()) if edge_values.size else 0.0
        row[prefix.format(metric="MeanIntensityEdge")] = safe_stat(edge_values, np.mean)
        row[prefix.format(metric="StdIntensityEdge")] = safe_stat(edge_values, np.std)
        row[prefix.format(metric="MinIntensityEdge")] = safe_stat(edge_values, np.min)
        row[prefix.format(metric="MaxIntensityEdge")] = safe_stat(edge_values, np.max)

        max_index = int(np.argmax(values)) if values.size else 0
        row[f"Location_MaxIntensity_X_{channel}"] = int(x[max_index]) if values.size else np.nan
        row[f"Location_MaxIntensity_Y_{channel}"] = int(y[max_index]) if values.size else np.nan
        row[f"Location_MaxIntensity_Z_{channel}"] = int(z[max_index]) if values.size else np.nan

        center_mass = weighted_center(coords, values, fallback=np.asarray(region.centroid))
        row[f"Location_CenterMassIntensity_X_{channel}"] = center_mass[2]
        row[f"Location_CenterMassIntensity_Y_{channel}"] = center_mass[1]
        row[f"Location_CenterMassIntensity_Z_{channel}"] = center_mass[0]
        row[prefix.format(metric="MassDisplacement")] = float(
            np.linalg.norm(center_mass - np.asarray(region.centroid))
        )

    return row


def measure_signal_core_intensity(
    region: measure._regionprops.RegionProperties,
    stacks: dict[str, np.ndarray],
    percentile: float,
    min_voxels: int,
) -> dict[str, object]:
    coords = region.coords
    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    green_values = stacks["green"][z, y, x].astype(np.float64, copy=False)
    red_values = stacks["red"][z, y, x].astype(np.float64, copy=False)
    signal_score = np.maximum(green_values, red_values)

    if signal_score.size == 0:
        core_indices = np.array([], dtype=int)
    else:
        min_voxels = max(1, min(int(min_voxels), signal_score.size))
        threshold = np.percentile(signal_score, float(percentile))
        core_indices = np.flatnonzero(signal_score >= threshold)
        if core_indices.size < min_voxels:
            core_indices = np.argpartition(signal_score, -min_voxels)[-min_voxels:]

    row: dict[str, object] = {
        "SignalCore_Volume": int(core_indices.size),
        "SignalCore_PercentObjectVolume": (
            float(core_indices.size / coords.shape[0]) if coords.shape[0] else np.nan
        ),
    }
    for channel, image in stacks.items():
        values = image[z, y, x].astype(np.float64, copy=False)
        core_values = values[core_indices] if core_indices.size else np.array([], dtype=np.float64)
        row[f"SignalCore_MeanIntensity_{channel}"] = safe_stat(core_values, np.mean)
        row[f"SignalCore_MedianIntensity_{channel}"] = safe_stat(core_values, np.median)
        row[f"SignalCore_MaxIntensity_{channel}"] = safe_stat(core_values, np.max)
        row[f"SignalCore_IntegratedIntensity_{channel}"] = (
            float(core_values.sum()) if core_values.size else 0.0
        )

    green_mean = row["SignalCore_MeanIntensity_green"]
    red_mean = row["SignalCore_MeanIntensity_red"]
    row["SignalCore_GreenRedRatio"] = safe_ratio(green_mean, red_mean)
    row["SignalCore_RedGreenRatio"] = safe_ratio(red_mean, green_mean)
    row["SignalCore_GreenMinusRed"] = safe_difference(green_mean, red_mean)
    return row


def safe_ratio(numerator: object, denominator: object) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or float(denominator) == 0.0:
        return np.nan
    return float(numerator) / float(denominator)


def safe_difference(left: object, right: object) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return np.nan
    return float(left) - float(right)


def safe_stat(values: np.ndarray, func) -> float:
    if values.size == 0:
        return np.nan
    return float(func(values))


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return np.nan
    return float(np.percentile(values, q))


def median_absolute_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def weighted_center(coords: np.ndarray, weights: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    total = weights.sum()
    if total <= 0:
        return fallback.astype(float)
    return np.average(coords.astype(float), axis=0, weights=weights)


def quantize_image(image: np.ndarray, levels: int) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    return np.floor(clipped * (levels - 1)).astype(np.uint16 if levels > 256 else np.uint8)


def texture_offsets(distance: int) -> list[tuple[int, int, int]]:
    d = int(distance)
    return [
        (0, 0, d),
        (0, d, 0),
        (d, 0, 0),
        (0, d, d),
        (0, d, -d),
        (d, 0, d),
        (d, 0, -d),
        (d, d, 0),
        (d, -d, 0),
        (d, d, d),
        (d, d, -d),
        (d, -d, d),
        (d, -d, -d),
    ]


def measure_texture(
    region: measure._regionprops.RegionProperties,
    labels: np.ndarray,
    quantized_stacks: dict[str, np.ndarray],
    levels: int,
    distance: int,
) -> dict[str, object]:
    row: dict[str, object] = {}
    min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
    object_mask = labels[min_z:max_z, min_y:max_y, min_x:max_x] == region.label
    offsets = texture_offsets(distance)

    for channel, image in quantized_stacks.items():
        crop = image[min_z:max_z, min_y:max_y, min_x:max_x]
        for direction_index, offset in enumerate(offsets):
            matrix = cooccurrence_matrix(crop, object_mask, offset, levels)
            features = haralick_features(matrix)
            for feature_name in TEXTURE_FEATURES:
                row[
                    f"Texture_{feature_name}_{channel}_{distance}_{direction_index:02d}_{levels}"
                ] = features[feature_name]

    return row


def cooccurrence_matrix(
    image: np.ndarray, mask: np.ndarray, offset: tuple[int, int, int], levels: int
) -> np.ndarray:
    dz, dy, dx = offset
    src_slices, dst_slices = paired_slices(image.shape, dz, dy, dx)
    if src_slices is None or dst_slices is None:
        return np.zeros((levels, levels), dtype=np.float64)

    src_mask = mask[src_slices]
    dst_mask = mask[dst_slices]
    valid = src_mask & dst_mask
    if not valid.any():
        return np.zeros((levels, levels), dtype=np.float64)

    src_values = image[src_slices][valid].ravel()
    dst_values = image[dst_slices][valid].ravel()
    matrix = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(matrix, (src_values, dst_values), 1.0)
    matrix += matrix.T
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def paired_slices(
    shape: tuple[int, int, int], dz: int, dy: int, dx: int
) -> tuple[tuple[slice, slice, slice] | None, tuple[slice, slice, slice] | None]:
    src: list[slice] = []
    dst: list[slice] = []
    for size, delta in zip(shape, (dz, dy, dx)):
        if abs(delta) >= size:
            return None, None
        if delta >= 0:
            src.append(slice(0, size - delta))
            dst.append(slice(delta, size))
        else:
            src.append(slice(-delta, size))
            dst.append(slice(0, size + delta))
    return (src[0], src[1], src[2]), (dst[0], dst[1], dst[2])


def haralick_features(matrix: np.ndarray) -> dict[str, float]:
    if matrix.sum() <= 0:
        return {name: 0.0 for name in TEXTURE_FEATURES}

    levels = matrix.shape[0]
    i, j = np.indices(matrix.shape)
    diff = i - j
    absdiff = np.abs(diff)
    eps = np.finfo(np.float64).eps

    px = matrix.sum(axis=1)
    py = matrix.sum(axis=0)
    mux = float((np.arange(levels) * px).sum())
    muy = float((np.arange(levels) * py).sum())
    sigx = math.sqrt(float(((np.arange(levels) - mux) ** 2 * px).sum()))
    sigy = math.sqrt(float(((np.arange(levels) - muy) ** 2 * py).sum()))

    entropy = -float((matrix[matrix > 0] * np.log2(matrix[matrix > 0])).sum())
    contrast = float((diff**2 * matrix).sum())
    variance = float(((i - mux) ** 2 * matrix).sum())
    angular_second_moment = float((matrix**2).sum())
    inverse_difference_moment = float((matrix / (1.0 + diff**2)).sum())
    if sigx > 0 and sigy > 0:
        correlation = float(((i - mux) * (j - muy) * matrix).sum() / (sigx * sigy))
    else:
        correlation = 0.0

    sum_distribution = np.bincount((i + j).ravel(), weights=matrix.ravel(), minlength=2 * levels - 1)
    sum_indices = np.arange(sum_distribution.size)
    sum_average = float((sum_indices * sum_distribution).sum())
    sum_entropy = entropy_from_distribution(sum_distribution)
    sum_variance = float(((sum_indices - sum_average) ** 2 * sum_distribution).sum())

    diff_distribution = np.bincount(absdiff.ravel(), weights=matrix.ravel(), minlength=levels)
    diff_indices = np.arange(diff_distribution.size)
    diff_mean = float((diff_indices * diff_distribution).sum())
    difference_variance = float(((diff_indices - diff_mean) ** 2 * diff_distribution).sum())
    difference_entropy = entropy_from_distribution(diff_distribution)

    hx = entropy_from_distribution(px)
    hy = entropy_from_distribution(py)
    px_py = np.outer(px, py)
    hxy1 = -float((matrix * np.log2(px_py + eps)).sum())
    hxy2 = -float((px_py * np.log2(px_py + eps)).sum())
    info_meas_1 = (entropy - hxy1) / max(hx, hy) if max(hx, hy) > 0 else 0.0
    info_meas_2 = math.sqrt(max(0.0, 1.0 - math.exp(-2.0 * max(0.0, hxy2 - entropy))))

    return {
        "AngularSecondMoment": angular_second_moment,
        "Contrast": contrast,
        "Correlation": correlation,
        "Variance": variance,
        "InverseDifferenceMoment": inverse_difference_moment,
        "SumAverage": sum_average,
        "SumVariance": sum_variance,
        "SumEntropy": sum_entropy,
        "Entropy": entropy,
        "DifferenceVariance": difference_variance,
        "DifferenceEntropy": difference_entropy,
        "InfoMeas1": info_meas_1,
        "InfoMeas2": info_meas_2,
    }


def entropy_from_distribution(distribution: np.ndarray) -> float:
    positive = distribution[distribution > 0]
    if positive.size == 0:
        return 0.0
    return -float((positive * np.log2(positive)).sum())


def process_image_set(
    image_number: int,
    image_set: ImageSet,
    output_dir: Path,
    preserve_labels: bool,
    object_conversion: str,
    signal_core_percentile: float,
    signal_core_min_voxels: int,
    texture_levels: int,
    texture_distance: int,
    skip_texture: bool,
    skip_overlays: bool,
    object_limit: int | None,
) -> list[dict[str, object]]:
    print(f"[{image_number}] Loading {image_set.stem}")
    raw_stacks = {channel: load_stack(paths) for channel, paths in image_set.channel_files.items()}
    stacks = {channel: normalize_intensity(stack) for channel, stack in raw_stacks.items()}
    raw_mask = tifffile.imread(image_set.mask_file)

    assert_stack_shapes(image_set, stacks, raw_mask)
    _rescaled_mask = rescale_mask_intensity(raw_mask)
    # Preserve the source Cellpose label while ObjectNumber tracks the processed object.
    object_labels = convert_image_to_objects(
        raw_mask,
        preserve_labels=preserve_labels,
        object_conversion=object_conversion,
    )
    object_labels = object_labels.astype(np.int32, copy=False)

    if not skip_overlays:
        overlays_dir = output_dir / "overlays"
        save_overlay(
            stacks["green"],
            object_labels,
            OVERLAY_COLORS["green"],
            overlays_dir / f"{image_set.stem}_green_overlay.tif",
        )
        save_overlay(
            stacks["red"],
            object_labels,
            OVERLAY_COLORS["red"],
            overlays_dir / f"{image_set.stem}_red_overlay.tif",
        )

    metadata = make_metadata_row(image_number, image_set, stacks)
    edge_mask = segmentation.find_boundaries(object_labels, mode="inner")
    regions = measure.regionprops(object_labels)
    if object_limit is not None:
        regions = regions[:object_limit]

    quantized = {}
    if not skip_texture:
        quantized = {channel: quantize_image(stack, texture_levels) for channel, stack in stacks.items()}

    rows: list[dict[str, object]] = []
    for idx, region in enumerate(regions, start=1):
        if idx % 100 == 0:
            print(f"  measured {idx}/{len(regions)} objects")
        row = dict(metadata)
        row.update(measure_size_shape(region, raw_mask))
        row.update(measure_intensity(region, object_labels, edge_mask, stacks))
        row.update(
            measure_signal_core_intensity(
                region,
                stacks,
                percentile=signal_core_percentile,
                min_voxels=signal_core_min_voxels,
            )
        )
        if not skip_texture:
            row.update(measure_texture(region, object_labels, quantized, texture_levels, texture_distance))
        rows.append(row)

    print(f"  exported {len(rows)} objects from {image_set.stem}")
    return rows


def assert_stack_shapes(image_set: ImageSet, stacks: dict[str, np.ndarray], mask: np.ndarray) -> None:
    reference_shape = next(iter(stacks.values())).shape
    for channel, stack in stacks.items():
        if stack.shape != reference_shape:
            raise ValueError(f"{image_set.stem}: channel {channel} has shape {stack.shape}, expected {reference_shape}")
    if mask.shape != reference_shape:
        raise ValueError(f"{image_set.stem}: mask has shape {mask.shape}, expected {reference_shape}")


def export_tables(
    rows: list[dict[str, object]],
    output_dir: Path,
    write_xlsx: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    object_table = pd.DataFrame(rows)
    if not object_table.empty:
        cell_id_location = object_table.columns.get_loc("ObjectNumber") + 1
        object_table.insert(cell_id_location, "CellID", np.arange(1, len(object_table) + 1, dtype=np.int64))

    object_csv = output_dir / "object_features.csv"
    object_table.to_csv(object_csv, index=False)

    summary_table = make_image_summary(object_table)
    summary_csv = output_dir / "image_summary_features.csv"
    summary_table.to_csv(summary_csv, index=False)

    if write_xlsx:
        object_table.to_excel(output_dir / "object_features.xlsx", index=False)
        summary_table.to_excel(output_dir / "image_summary_features.xlsx", index=False)

    print(f"Wrote {object_csv}")
    print(f"Wrote {summary_csv}")


def make_image_summary(object_table: pd.DataFrame) -> pd.DataFrame:
    if object_table.empty:
        return object_table

    group_columns = ["ImageNumber", "ImageSet", "Metadata_Date", "Metadata_Field", "FoF", "Date"]
    excluded_numeric_columns = {
        "ImageNumber",
        "ObjectNumber",
        "CellID",
        "CellposeLabel",
        "Number_Object_Number",
    }
    numeric_columns = [
        column
        for column in object_table.select_dtypes(include=[np.number]).columns
        if column not in excluded_numeric_columns
    ]
    grouped = object_table.groupby(group_columns, dropna=False)[numeric_columns]
    mean_table = grouped.mean().add_prefix("Mean_")
    median_table = grouped.median().add_prefix("Median_")
    std_table = grouped.std(ddof=0).add_prefix("Std_")
    count_table = object_table.groupby(group_columns, dropna=False).size().rename("Count_object")
    return pd.concat([count_table, mean_table, median_table, std_table], axis=1).reset_index()


def main() -> None:
    args = parse_args()
    image_sets = discover_image_sets_by_layout(args.image_dir, args.mask_dir, args.fields, args.input_layout)
    if args.limit is not None:
        image_sets = image_sets[: args.limit]
    if not image_sets:
        raise SystemExit("No matching image sets were found.")

    print(f"Discovered {len(image_sets)} image set(s).")
    all_rows: list[dict[str, object]] = []
    for image_number, image_set in enumerate(image_sets, start=1):
        rows = process_image_set(
            image_number=image_number,
            image_set=image_set,
            output_dir=args.output_dir,
            preserve_labels=args.preserve_mask_labels,
            object_conversion=args.object_conversion,
            signal_core_percentile=args.signal_core_percentile,
            signal_core_min_voxels=args.signal_core_min_voxels,
            texture_levels=args.texture_levels,
            texture_distance=args.texture_distance,
            skip_texture=args.skip_texture,
            skip_overlays=args.skip_overlays,
            object_limit=args.object_limit,
        )
        all_rows.extend(rows)

    export_tables(
        all_rows,
        args.output_dir,
        write_xlsx=args.write_xlsx,
    )


if __name__ == "__main__":
    main()
