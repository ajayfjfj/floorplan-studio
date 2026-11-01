"""Verify model identity, asset hashes, and image-only demo samples."""

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    dist = root / "web/dist"
    manifest = json.loads((dist / "asset-manifest.json").read_text())
    for name, record in manifest["files"].items():
        path = dist / name
        assert path.is_file(), f"Missing browser asset: {name}"
        assert path.stat().st_size == record["bytes"], f"Asset size changed: {name}"
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        ), f"Asset hash changed: {name}"
        assert (
            path.stat().st_size < 25 * 1024**2
        ), f"Static asset exceeds host limit: {name}"
    exported = json.loads(
        (root / "results/browser_export_verification.json").read_text()
    )
    model = manifest["files"]["models/floorplan-segformer.onnx"]
    assert exported["passed"] and exported["sha256"] == model["sha256"]
    assert len(exported["checks"]) == 8
    assert min(row["argmax_agreement"] for row in exported["checks"]) >= 0.999
    schema = json.loads((dist / "prediction.schema.json").read_text())
    assert schema == json.loads(
        (root / "src/floorplan/prediction.schema.json").read_text()
    )
    samples = json.loads((dist / "samples/index.json").read_text())
    assert len(samples) == 8 and len({s["id"] for s in samples}) == 8
    assert not list(
        (dist / "samples").rglob("prediction.json")
    ), "Saved predictions must not ship in the interface"
    for sample in samples:
        folder = dist / "samples" / sample["id"]
        with Image.open(folder / "floorplan.png") as image:
            image.verify()
        metadata = json.loads((folder / "source.json").read_text())
        assert metadata["sample_id"] == sample["source_id"]
        reference = root / "results/test/samples" / sample["id"] / "floorplan.png"
        assert (folder / "floorplan.png").read_bytes() == reference.read_bytes()

    gallery_ids = []

    class References(HTMLParser):
        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if tag == "button" and "sample-card" in attributes.get("class", "").split():
                gallery_ids.append(attributes.get("data-id"))
            for key, value in attrs:
                if key not in ("src", "href") or not value:
                    continue
                url = urlparse(value)
                if not url.scheme and url.path not in ("", "./"):
                    assert (
                        dist / url.path
                    ).is_file(), f"Broken local reference: {value}"

    for path in dist.glob("*.html"):
        References().feed(path.read_text())
    assert gallery_ids == [
        sample["id"] for sample in samples
    ], "Static HTML must include all eight sample controls"
    print(
        f"Web checks passed: {len(manifest['files'])} hashed assets, ONNX model parity, 8 image-only samples, local HTML links."
    )


if __name__ == "__main__":
    main()
