# GitHub setup

The source, trained weights, browser export, sample predictions, evaluation records, and final PowerPoint are ready to version together. Large training data and local tooling are excluded by `.gitignore`.

## Before the first push

Run these checks from the repository root:

```bash
python -m unittest discover -s tests -v
node tests/test_browser_geometry.cjs
python scripts/verify_web.py
python scripts/verify_submission.py
git status --short --untracked-files=all
```

The expected source tree includes `src/`, `scripts/`, `tests/`, `docs/`, `notebooks/`, `artifacts/`, `results/`, `presentation/`, `web/`, and `.github/`. Do not force-add `.build/`, `.venv/`, raw datasets, credentials, or local hosting state. The review zip is a downloadable local deliverable and is intentionally ignored by Git.

All current model and runtime assets are below 25 MiB per file. Keep the browser ONNX model and WASM files as normal Git blobs so static hosting serves the actual bytes. Replacing them with unresolved Git LFS pointers breaks inference.

## Publish the source

Create an empty repository in the intended GitHub account, then set that repository as the remote and push the reviewed local branch. No GitHub remote is configured in the delivered project.

## GitHub Pages hosting

1. Open repository **Settings → Pages** and choose **GitHub Actions** as the source.
2. In **Actions**, run **Publish demo to GitHub Pages** manually.
3. Open the deployment URL returned by that workflow.

The workflow publishes `web/dist/` directly. It requires no model service, build step, or secret. Relative URLs support both a repository subpath and a custom domain. Source pushes run checks; they do not trigger Pages publication automatically.

Refer to [GitHub's custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) for repository settings and deployment permissions.
