"""Export the selected checkpoint and check numerical parity on held-out samples."""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from floorplan.common import image_tensor, letterbox


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "web/dist/models/floorplan-segformer.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = Logits(
        SegformerForSemanticSegmentation.from_pretrained(
            root / "artifacts/segformer/best"
        ).eval()
    ).eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 512, 512),
        output,
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        external_data=False,
    )
    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    checks = []
    for folder in sorted((root / "results/test/samples").glob("sample_*")):
        with Image.open(folder / "floorplan.png") as image:
            canvas, _, _ = letterbox(image.convert("RGB"), 512)
            tensor = image_tensor(canvas).unsqueeze(0)
        with torch.inference_mode():
            reference = model(tensor).numpy()
        actual = session.run(None, {"pixel_values": tensor.numpy()})[0]
        max_error = float(np.max(np.abs(reference - actual)))
        agreement = float(np.mean(reference.argmax(1) == actual.argmax(1)))
        if max_error > 0.001 or agreement < 0.999:
            raise RuntimeError(
                f"ONNX parity failed: {folder.name}, {max_error}, {agreement}"
            )
        checks.append(
            {
                "sample": folder.name,
                "max_logit_error": max_error,
                "argmax_agreement": agreement,
            }
        )
    report = {
        "checkpoint": "artifacts/segformer/best",
        "opset": 17,
        "input": [1, 3, 512, 512],
        "output": [1, 5, 128, 128],
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "checks": checks,
        "passed": True,
    }
    (root / "results/browser_export_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
