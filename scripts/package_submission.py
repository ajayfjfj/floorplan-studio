"""Create the local review bundle after verification has passed."""

import json
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    verification = json.loads(
        (root / "results/submission_verification.json").read_text()
    )
    if not verification.get("complete"):
        raise RuntimeError("Run verify_submission.py first")
    target = root / "dist/FloorPlan_Detection_Submission.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for name in (
            "src",
            "scripts",
            "tests",
            "docs",
            "notebooks",
            "artifacts",
            "results",
            "presentation",
            "web",
            ".github",
        ):
            for p in sorted((root / name).rglob("*")):
                if (
                    not p.is_file()
                    or "__pycache__" in p.parts
                    or any(x.endswith(".egg-info") for x in p.parts)
                    or any(x in (".git", ".openai", "node_modules") for x in p.parts)
                    or p.name == ".DS_Store"
                ):
                    continue
                if "last" in p.parts or p.name == "optimizer.pt":
                    continue
                z.write(p, p.relative_to(root))
        for name in (
            "README.md",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "requirements.txt",
            "pyproject.toml",
            ".gitignore",
            ".gitattributes",
        ):
            z.write(root / name, name)
    print(f"Saved {target} ({target.stat().st_size/1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
