"""Create a small source-only archive to upload to Colab."""

import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / ".build/project_source.zip"
target.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("src", "scripts", "tests", "docs", "notebooks"):
        for path in (root / name).rglob("*"):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and not any(p.endswith(".egg-info") for p in path.parts)
                and path.suffix != ".pyc"
            ):
                z.write(path, path.relative_to(root))
    for name in ("pyproject.toml", "README.md", "THIRD_PARTY_NOTICES.md", "LICENSE"):
        z.write(root / name, name)
print(target)
