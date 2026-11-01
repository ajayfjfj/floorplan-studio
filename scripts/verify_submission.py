"""Validate deliverable presence, recorded training, model reload and sample JSON."""

import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation
from safetensors.torch import load_file
from floorplan.common import save_json
from floorplan.evaluate import validate_json
from floorplan.geometry import vectorize
from floorplan.model import predict


def main():
    root = Path(__file__).resolve().parents[1]
    required = [
        "README.md",
        "docs/DATASET.md",
        "docs/JSON_FORMAT.md",
        "notebooks/FloorPlan_Colab.ipynb",
        "artifacts/segformer/best/model.safetensors",
        "artifacts/segformer/best/config.json",
        "artifacts/segformer/history.json",
        "artifacts/segformer/finetuning_evidence.json",
        "results/test/metrics.json",
        "results/RESULTS.md",
        "presentation/FloorPlan_Detection_Final.pptx",
        "web/dist/index.html",
        "web/dist/models/floorplan-segformer.onnx",
        "web/README.md",
        "results/browser_export_verification.json",
        "results/browser_demo_verification.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete submission: {missing}")
    evidence = json.loads(
        (root / "artifacts/segformer/finetuning_evidence.json").read_text()
    )
    if (
        evidence["optimizer_steps"] < 1
        or evidence["encoder_max_absolute_change_this_run"] <= 0
    ):
        raise ValueError(
            "Fine-tuning evidence must demonstrate optimizer steps and encoder updates"
        )
    metrics = json.loads((root / "results/test/metrics.json").read_text())
    per_plan = json.loads((root / "results/test/per_plan.json").read_text())
    if metrics["plans"] != len(per_plan) or metrics["json_valid_fraction"] != 1:
        raise ValueError("Evaluation count or JSON validity mismatch")
    if len({r["source_sha256"] for r in per_plan}) != len(per_plan):
        raise ValueError("Duplicate images in the reported test set")
    for name in ["room", "wall", "door", "window"]:
        if name not in metrics["pixel"]["per_class"]:
            raise ValueError(f"Missing class: {name}")
    sample_dirs = sorted((root / "results/test/samples").glob("sample_*"))
    if len(sample_dirs) < 8:
        raise ValueError("Expected at least eight sample outputs")
    for folder in sample_dirs:
        validate_json(json.loads((folder / "prediction.json").read_text()))
        for name in [
            "floorplan.png",
            "prediction_overlay.png",
            "ground_truth_overlay.png",
        ]:
            Image.open(folder / name).verify()
    torch.set_num_threads(4)
    model = SegformerForSemanticSegmentation.from_pretrained(
        root / "artifacts/segformer/best"
    ).eval()
    initial = load_file(root / "artifacts/segformer/initial/model.safetensors")
    trained_state = model.state_dict()
    best_encoder_change = max(
        float((trained_state[k] - initial[k]).abs().max())
        for k in initial
        if k.startswith("segformer.")
    )
    if best_encoder_change <= 0:
        raise ValueError("Submitted encoder weights are unchanged from initialization")
    image = Image.open(sample_dirs[0] / "floorplan.png").convert("RGB")
    mask, conf = predict(model, image, torch.device("cpu"))
    validate_json(vectorize(mask, conf, "local_checkpoint_reload"))
    files = {}
    for name in required:
        p = root / name
        files[name] = {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    save_json(
        root / "results/submission_verification.json",
        {
            "complete": True,
            "trained_checkpoint_reload": "passed on local CPU",
            "sample_json_validated": len(sample_dirs),
            "test_plans": len(per_plan),
            "best_checkpoint_encoder_change": best_encoder_change,
            "files": files,
        },
    )
    print(
        "Submission checks passed; trained weights reload and produce valid JSON locally."
    )


if __name__ == "__main__":
    main()
