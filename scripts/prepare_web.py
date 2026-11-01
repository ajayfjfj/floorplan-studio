"""Copy held-out examples and prepare reproducible static demo assets."""

import argparse
import hashlib
import html
import io
import json
import re
import shutil
import tarfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "web/dist"
RUNTIMES = {
    "onnxruntime": {
        "version": "1.22.0",
        "url": "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.22.0.tgz",
        "sha256": "e293551f9d36d003e293d0f2937fee7b3dfdcef76d71d1449b55e2d4157c7aea",
    },
    "opencv": {
        "version": "4.13.0",
        "url": "https://docs.opencv.org/4.x/opencv.js",
        "sha256": "63366510248adf3a7eddf3e793dd825404efb7df3749f4d6f8557c7fa4ca8aa0",
    },
}


def download_runtimes():
    import requests

    def fetch(url, checksum=None):
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        data = response.content
        if checksum and hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(
                f"Upstream asset changed; refusing unverified bytes: {url}"
            )
        return data

    vendor = DIST / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    runtime = RUNTIMES["onnxruntime"]
    with tarfile.open(
        fileobj=io.BytesIO(fetch(runtime["url"], runtime["sha256"]))
    ) as archive:
        for filename in (
            "ort.wasm.min.js",
            "ort-wasm-simd-threaded.mjs",
            "ort-wasm-simd-threaded.wasm",
        ):
            (vendor / filename).write_bytes(
                archive.extractfile(f"package/dist/{filename}").read()
            )
    opencv = RUNTIMES["opencv"]
    (vendor / "opencv.js").write_bytes(fetch(opencv["url"], opencv["sha256"]))
    for target, url in {
        "ONNXRUNTIME-LICENSE.txt": "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/LICENSE",
        "ONNXRUNTIME-NOTICES.txt": "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/ThirdPartyNotices.txt",
        "OPENCV-LICENSE.txt": "https://raw.githubusercontent.com/opencv/opencv/4.13.0/LICENSE",
    }.items():
        (vendor / target).write_bytes(fetch(url))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-runtimes", action="store_true")
    args = parser.parse_args()
    if args.download_runtimes:
        download_runtimes()
    titles = {
        1: "Architectural plan",
        2: "Open layout",
        3: "Detailed drawing",
        4: "Narrow apartment",
        5: "Residential plan",
        6: "L-shaped layout",
        7: "Compact apartment",
        8: "Multi-unit plan",
    }
    samples = []
    for i in (4, 7, 2, 6, 1, 3, 5, 8):
        source = ROOT / f"results/test/samples/sample_{i:02d}"
        target = DIST / "samples" / source.name
        target.mkdir(parents=True, exist_ok=True)
        metadata = json.loads((source / "source.json").read_text())
        # Evaluation predictions stay in results/; the interface ships input images only.
        (target / "prediction.json").unlink(missing_ok=True)
        for name in ("floorplan.png", "source.json"):
            shutil.copy2(source / name, target / name)
        with Image.open(source / "floorplan.png") as image:
            image.resize((160, 160)).save(target / "thumbnail.webp", quality=80)
        samples.append(
            {"id": source.name, "title": titles[i], "source_id": metadata["sample_id"]}
        )
    (DIST / "samples/index.json").write_text(json.dumps(samples, indent=2) + "\n")
    cards = []
    for sample in samples:
        sample_id = html.escape(sample["id"], quote=True)
        title = html.escape(sample["title"], quote=True)
        source_id = html.escape(sample["source_id"], quote=True)
        cards.append(
            f'<button class="sample-card" data-id="{sample_id}" data-title="{title}" '
            f'data-source-id="{source_id}" aria-label="Load sample: {title}" '
            f'aria-pressed="false" title="{title} · CubiCasa5K · {source_id}">'
            f'<img src="samples/{sample_id}/thumbnail.webp" alt="" width="100" height="100">'
            f'<span class="sample-label">{title}</span></button>'
        )
    index = DIST / "index.html"
    markup, count = re.subn(
        r"<!-- SAMPLE_PLANS_START -->.*?<!-- SAMPLE_PLANS_END -->",
        "<!-- SAMPLE_PLANS_START -->\n"
        + "\n".join(cards)
        + "\n<!-- SAMPLE_PLANS_END -->",
        index.read_text(),
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("Expected exactly one static sample-gallery placeholder")
    index.write_text(markup)
    for source, target in (
        ("src/floorplan/prediction.schema.json", "prediction.schema.json"),
        ("artifacts/segformer/NVIDIA_LICENSE.txt", "models/NVIDIA_LICENSE.txt"),
        ("LICENSE", "PROJECT-LICENSE.txt"),
    ):
        shutil.copy2(ROOT / source, DIST / target)
    files = {}
    for path in sorted(DIST.rglob("*")):
        if path.is_file() and path.name != "asset-manifest.json":
            files[path.relative_to(DIST).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (DIST / "asset-manifest.json").write_text(
        json.dumps({"runtimes": RUNTIMES, "files": files}, indent=2) + "\n"
    )
    print(f"Prepared {len(samples)} samples and recorded {len(files)} static assets.")


if __name__ == "__main__":
    main()
