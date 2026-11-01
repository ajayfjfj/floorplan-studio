import json
import time
from pathlib import Path
import jsonschema
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
from .common import CLASSES, device_for, save_json, overlay
from .model import predict
from .geometry import vectorize, components
from .metrics import confusion_matrix, pixel_metrics, instance_counts, count_metrics


def validate_json(value):
    schema = json.loads(Path(__file__).with_name("prediction.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(value)


def infer(checkpoint, image_path, output, device="auto", size=512):
    device = device_for(device)
    model = (
        SegformerForSemanticSegmentation.from_pretrained(checkpoint).to(device).eval()
    )
    image = Image.open(image_path).convert("RGB")
    if min(image.size) < 2:
        raise ValueError("Image must have width and height >=2")
    mask, conf = predict(model, image, device, size)
    value = vectorize(mask, conf, Path(image_path).stem)
    validate_json(value)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "prediction.json", value)
    Image.fromarray(mask).save(output / "mask.png")
    overlay(image, mask).save(output / "overlay.png")
    return value


def evaluate(data, checkpoint, output, split="test", device="auto", samples=8):
    device = device_for(device)
    root = Path(data)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((root / f"{split}.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    model = (
        SegformerForSemanticSegmentation.from_pretrained(checkpoint).to(device).eval()
    )
    cm = np.zeros((5, 5), np.int64)
    counts = {n: np.zeros(3, int) for n in CLASSES[1:]}
    records = []
    valid_json = 0
    for index, row in enumerate(tqdm(rows, desc=f"evaluate {split}")):
        folder = root / row["path"]
        image = Image.open(folder / "image.png").convert("RGB")
        truth = np.asarray(Image.open(folder / "mask.png"))
        ann = json.loads((folder / "annotations.json").read_text())
        started = time.perf_counter()
        pred, confidence = predict(model, image, device, audit["size"])
        elapsed = time.perf_counter() - started
        # Ignore padding in pixel evaluation, and clear it before instance extraction.
        valid = truth != 255
        pred[~valid] = 0
        plan_cm = confusion_matrix(truth, pred)
        cm += plan_cm
        geometry = vectorize(pred, confidence, row["id"])
        validate_json(geometry)
        valid_json += 1
        local = {}
        room_error = 0
        left, top, nw, nh = ann["letterbox"]
        source_w, source_h = ann["size"]
        for name in CLASSES[1:]:
            gt = []
            for obj in ann["objects"]:
                if obj["class"] != name:
                    continue
                coords = np.array(obj["polygon"]) * [nw / source_w, nh / source_h] + [
                    left,
                    top,
                ]
                m = Image.new("L", image.size, 0)
                ImageDraw.Draw(m).polygon([tuple(p) for p in coords], fill=1)
                binary = (
                    np.asarray(m).astype(bool) & valid & (truth == CLASSES.index(name))
                )
                if binary.any():
                    gt.append(binary)
            ps = components(pred == CLASSES.index(name), 12)
            c = instance_counts(gt, ps, 0.5)
            counts[name] += c
            local[name] = count_metrics(c)
            if name == "room":
                room_error = abs(len(gt) - len(ps))
        records.append(
            {
                "sample_id": row["id"],
                "source_sha256": row["source_sha256"],
                "foreground_miou": pixel_metrics(plan_cm)["mean_iou_foreground"],
                "room_count_absolute_error": room_error,
                "inference_seconds": elapsed,
                "instances": local,
            }
        )
        if index < samples:
            dest = output / "samples" / f"sample_{index+1:02d}"
            dest.mkdir(parents=True, exist_ok=True)
            image.save(dest / "floorplan.png")
            Image.fromarray(truth).save(dest / "ground_truth_mask.png")
            overlay(image, pred).save(dest / "prediction_overlay.png")
            overlay(image, np.where(valid, truth, 0)).save(
                dest / "ground_truth_overlay.png"
            )
            save_json(dest / "prediction.json", geometry)
            save_json(
                dest / "source.json",
                {
                    "sample_id": row["id"],
                    "source": "CubiCasa5K",
                    "license": "CC BY-NC 4.0",
                    "doi": "10.5281/zenodo.2613548",
                    "letterbox": ann["letterbox"],
                    "source_size": ann["size"],
                },
            )
    report = {
        "split": split,
        "plans": len(rows),
        "resolution": audit["size"],
        "device": str(device),
        "checkpoint": str(checkpoint),
        "pixel": pixel_metrics(cm),
        "instances_iou_0_5": {n: count_metrics(c) for n, c in counts.items()},
        "room_count_mae": float(
            np.mean([r["room_count_absolute_error"] for r in records])
        ),
        "json_valid_fraction": valid_json / max(1, len(rows)),
        "median_inference_seconds": float(
            np.median([r["inference_seconds"] for r in records])
        ),
        "instance_note": "True instances are individual SVG polygons; predicted instances are connected semantic-mask regions. Wall-instance scores are diagnostic because connected wall networks merge segments.",
    }
    save_json(output / "metrics.json", report)
    save_json(output / "per_plan.json", records)
    return report
