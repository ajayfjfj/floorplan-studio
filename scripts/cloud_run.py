"""End-to-end Colab job. Run in a fresh runtime after uploading project_source.zip."""

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/content/floorplan_project")


def run(args):
    print("RUN", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def main():
    run([sys.executable, "-m", "pip", "install", "-e", "."])
    run(
        [
            sys.executable,
            "scripts/fetch_dataset.py",
            "--output",
            "/content/data/cubicasa5k",
            "--archive",
            "/content/cubicasa5k.zip",
            "--workers",
            "8",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "floorplan.cli",
            "prepare",
            "--raw",
            "/content/data/cubicasa5k",
            "--output",
            "/content/data/prepared",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "floorplan.cli",
            "train",
            "--data",
            "/content/data/prepared",
            "--output",
            "artifacts/segformer",
            "--epochs",
            "30",
            "--batch-size",
            "16",
            "--device",
            "cuda",
            "--resume",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "floorplan.cli",
            "evaluate",
            "--data",
            "/content/data/prepared",
            "--checkpoint",
            "artifacts/segformer/best",
            "--output",
            "results/test",
            "--device",
            "cuda",
        ]
    )
    run([sys.executable, "scripts/report_results.py"])
    # Preserve provenance and audit without copying the dataset into the submission.
    import shutil

    shutil.copy(
        "/content/data/prepared/audit.json", ROOT / "results/dataset_audit.json"
    )
    shutil.copy(
        "/content/data/cubicasa5k/download_provenance.json",
        ROOT / "results/download_provenance.json",
    )
    run([sys.executable, "-m", "pip", "freeze"])
    environment_dir = ROOT / "docs/environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    (environment_dir / "environment_colab.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    dest = Path("/content/floorplan_trained_artifacts.zip")
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for base in (ROOT / "artifacts/segformer", ROOT / "results", environment_dir):
            for path in base.rglob("*"):
                if (
                    path.is_file()
                    and "last" not in path.parts
                    and path.name != "optimizer.pt"
                ):
                    z.write(path, path.relative_to(ROOT))
    print("JOB_COMPLETE", dest, flush=True)


if __name__ == "__main__":
    main()
