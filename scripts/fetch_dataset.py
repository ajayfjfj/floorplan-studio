"""Download the official archive with resumable, bounded parallel HTTP ranges."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
import requests

URL = "https://zenodo.org/api/records/2613548"


def fetch(output, archive, workers=4):
    output, archive = Path(output), Path(archive)
    output.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    record = response.json()
    item = next(f for f in record["files"] if f["key"] == "cubicasa5k.zip")
    url = item["links"]["self"]
    size = item["size"]
    chunk = 64 * 1024 * 1024
    if not archive.exists() or archive.stat().st_size != size:

        def get_part(index):
            start = index * chunk
            end = min(size, start + chunk) - 1
            path = archive.with_name(archive.name + f".part{index:03d}")
            if path.exists() and path.stat().st_size == end - start + 1:
                return path
            for attempt in range(6):
                try:
                    with requests.get(
                        url,
                        headers={"Range": f"bytes={start}-{end}"},
                        stream=True,
                        timeout=(30, 120),
                    ) as r:
                        r.raise_for_status()
                        if r.status_code != 206:
                            raise RuntimeError("Server does not honor HTTP ranges")
                        with path.open("wb") as f:
                            for block in r.iter_content(1024 * 1024):
                                f.write(block)
                    if path.stat().st_size != end - start + 1:
                        raise IOError("Incomplete range")
                    print(
                        f"downloaded chunk {index+1}/{(size+chunk-1)//chunk}",
                        flush=True,
                    )
                    return path
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(min(60, 2**attempt))

        count = (size + chunk - 1) // chunk
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(get_part, range(count)))
        with archive.open("wb") as dest:
            for part in parts:
                with part.open("rb") as src:
                    shutil.copyfileobj(src, dest)
        for part in parts:
            part.unlink()
    algo, expected = item["checksum"].split(":", 1)
    digest = hashlib.new(algo)
    with archive.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise IOError("Dataset checksum mismatch; remove the corrupt archive and retry")
    print("Archive checksum verified", flush=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename)
            if name.name not in (
                "F1_scaled.png",
                "model.svg",
                "train.txt",
                "val.txt",
                "test.txt",
            ):
                continue
            # The official archive has a cubicasa5k/ root directory.
            parts = name.parts[1:] if name.parts[0] == "cubicasa5k" else name.parts
            dest = output.joinpath(*parts).resolve()
            if not dest.is_relative_to(output.resolve()):
                raise ValueError("Unsafe ZIP member")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, dest.open("wb") as target:
                shutil.copyfileobj(src, target)
    # Some archive versions omit split lists; obtain those from the official repo.
    for split in ("train", "val", "test"):
        path = output / f"{split}.txt"
        if not path.exists():
            r = requests.get(
                f"https://raw.githubusercontent.com/CubiCasa/CubiCasa5k/master/data/{split}.txt",
                timeout=60,
            )
            r.raise_for_status()
            path.write_text(r.text)
    (output / "download_provenance.json").write_text(
        json.dumps(
            {
                "record": URL,
                "archive": item["key"],
                "checksum": item["checksum"],
                "license": "CC BY-NC 4.0",
            },
            indent=2,
        )
    )
    print(f"Dataset ready: {output}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/raw/cubicasa5k")
    p.add_argument("--archive", default="data/raw/cubicasa5k.zip")
    p.add_argument("--workers", type=int, default=4)
    fetch(**vars(p.parse_args()))
