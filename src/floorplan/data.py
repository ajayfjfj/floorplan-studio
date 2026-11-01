"""CubiCasa SVG polygons, preserving the official floor-level split."""

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from .common import CLASSES, letterbox, save_json

NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"


def transform_matrix(value):
    result = np.eye(3)
    for kind, arg in re.findall(r"(\w+)\s*\(([^)]*)\)", value or ""):
        a = list(map(float, re.findall(NUMBER, arg)))
        m = np.eye(3)
        if kind == "matrix" and len(a) == 6:
            m = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1.0]])
        elif kind == "translate":
            m[:2, 2] = [a[0], a[1] if len(a) > 1 else 0]
        elif kind == "scale":
            m[0, 0], m[1, 1] = a[0], a[1] if len(a) > 1 else a[0]
        elif kind == "rotate":
            angle = np.deg2rad(a[0])
            c, s = np.cos(angle), np.sin(angle)
            m[:2, :2] = [[c, -s], [s, c]]
            if len(a) == 3:
                p = np.array(a[1:])
                m[:2, 2] = p - m[:2, :2] @ p
        elif kind in ("skewX", "skewY"):
            m[0 if kind == "skewX" else 1, 1 if kind == "skewX" else 0] = np.tan(
                np.deg2rad(a[0])
            )
        else:
            raise ValueError(f"Unsupported SVG transform: {kind}({arg})")
        result = result @ m
    return result


def svg_objects(path, image_size):
    root = ET.parse(path).getroot()
    w, h = image_size
    vb = list(map(float, re.findall(NUMBER, root.get("viewBox", ""))))
    if len(vb) == 4:
        _, _, sw, sh = vb
    else:
        sw = float(re.findall(NUMBER, root.get("width", str(w)))[0])
        sh = float(re.findall(NUMBER, root.get("height", str(h)))[0])
    if sw <= 0 or sh <= 0:
        raise ValueError("Invalid SVG dimensions")
    # CubiCasa's F1_scaled raster and polygon points already share pixel space.
    # The declared SVG viewport can differ from the raster; the official House
    # loader likewise uses polygon points directly instead of viewport scaling.
    base = np.eye(3)
    objects = []

    def visit(node, parent):
        tag = node.tag.split("}")[-1]
        if tag not in ("svg", "g"):
            return
        if "Dimension" in node.get("class", "").split():
            return
        matrix = parent @ transform_matrix(node.get("transform", ""))
        if tag == "g":
            identity = node.get("id", "")
            parts = node.get("class", "").split()
            kind = identity.lower() if identity in ("Wall", "Door", "Window") else None
            source = identity
            if "Space" in parts:
                i = parts.index("Space")
                source = parts[i + 1] if i + 1 < len(parts) else "Undefined"
                kind = None if source in ("Outdoor", "Background") else "room"
            if kind:
                polygons = [p for p in node if p.tag.split("}")[-1] == "polygon"]
                if not polygons:
                    raise ValueError(f"No direct polygon for {kind}: {path}")
                for p in polygons:
                    nums = np.array(
                        list(map(float, re.findall(NUMBER, p.get("points", ""))))
                    )
                    if len(nums) < 6 or len(nums) % 2:
                        raise ValueError(f"Invalid polygon in {path}")
                    points = nums.reshape(-1, 2)
                    points = (
                        matrix
                        @ transform_matrix(p.get("transform", ""))
                        @ np.c_[points, np.ones(len(points))].T
                    ).T[:, :2]
                    objects.append(
                        {
                            "class": kind,
                            "source_category": source,
                            "polygon": points.tolist(),
                        }
                    )
        for child in node:
            visit(child, matrix)

    visit(root, base)
    if not any(o["class"] == "wall" for o in objects):
        raise ValueError(f"No walls found: {path}")
    return objects


def rasterize(objects, size):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    # Higher class index wins: window > door > wall > room > background.
    for name in CLASSES[1:]:
        for obj in objects:
            if obj["class"] == name:
                draw.polygon(
                    [tuple(p) for p in obj["polygon"]], fill=CLASSES.index(name)
                )
    return mask


def prepare(raw, output, size=512):
    raw, output = Path(raw), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    splits = {}
    fingerprints = {}
    report = {
        "classes": CLASSES,
        "size": size,
        "splits": {},
        "original_splits": {},
        "duplicates": [],
        "duplicate_policy": "Keep first occurrence with split priority test > val > train; exclude exact RGB pixel duplicates from lower-priority splits.",
        "mapping": {},
        "pixel_counts": [0] * 5,
    }
    for split in ("train", "val", "test"):
        source = raw / f"{split}.txt"
        if not source.exists():
            raise FileNotFoundError(source)
        entries = [
            x.strip().strip("/") for x in source.read_text().splitlines() if x.strip()
        ]
        splits[split] = entries
        report["original_splits"][split] = len(entries)
        if len(entries) != len(set(entries)):
            raise ValueError(f"Duplicate sample IDs in {split}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(splits[a]) & set(splits[b]):
            raise ValueError(f"Split overlap: {a}, {b}")
    # Resolve image duplicates before learning: held-out data always wins.
    for split in ("test", "val", "train"):
        entries = splits[split]
        manifest = []
        for sample in tqdm(entries, desc=f"prepare {split}"):
            folder = (raw / sample).resolve()
            if not folder.is_relative_to(raw.resolve()):
                raise ValueError("Unsafe dataset path")
            image_path = folder / "F1_scaled.png"
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            fingerprint = hashlib.sha256(
                image.tobytes() + str(image.size).encode()
            ).hexdigest()
            annotation_hash = hashlib.sha256(
                (folder / "model.svg").read_bytes()
            ).hexdigest()
            if fingerprint in fingerprints:
                kept = fingerprints[fingerprint]
                report["duplicates"].append(
                    {
                        "excluded_split": split,
                        "excluded_id": sample,
                        "kept_split": kept["split"],
                        "kept_id": kept["id"],
                        "image_sha256": fingerprint,
                        "identical_svg_bytes": annotation_hash
                        == kept["annotation_sha256"],
                    }
                )
                continue
            fingerprints[fingerprint] = {
                "split": split,
                "id": sample,
                "annotation_sha256": annotation_hash,
            }
            objects = svg_objects(folder / "model.svg", image.size)
            mask = rasterize(objects, image.size)
            for obj in objects:
                report["mapping"][obj["source_category"]] = obj["class"]
            canvas, target, box = letterbox(image, size, mask)
            key = sample.replace("/", "__")
            dest = output / split / key
            dest.mkdir(parents=True, exist_ok=True)
            canvas.save(dest / "image.png")
            target.save(dest / "mask.png")
            # Store source-coordinate annotations, used for true room/object evaluation.
            save_json(
                dest / "annotations.json",
                {
                    "sample_id": sample,
                    "size": list(image.size),
                    "objects": objects,
                    "letterbox": box,
                },
            )
            if split == "train":
                counts = np.bincount(np.asarray(target).ravel(), minlength=256)[:5]
                report["pixel_counts"] = (
                    np.array(report["pixel_counts"]) + counts
                ).tolist()
            manifest.append(
                {
                    "id": sample,
                    "path": str(Path(split) / key),
                    "source_sha256": fingerprint,
                    "annotation_sha256": annotation_hash,
                }
            )
        if not manifest:
            raise ValueError(f"No unique examples remain in {split}")
        save_json(output / f"{split}.json", manifest)
        report["splits"][split] = len(manifest)
    save_json(output / "audit.json", report)
    return report
