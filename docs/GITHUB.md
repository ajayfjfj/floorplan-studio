# Repository maintenance

The source, trained weights, browser export, sample predictions, evaluation records, and final PowerPoint are versioned in [ajayfjfj/floorplan-studio](https://github.com/ajayfjfj/floorplan-studio). The repository is **private**, its default branch is `main`, and GitHub Pages is disabled. Large training data and local tooling are excluded by `.gitignore`.

## Before pushing changes

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

## Push reviewed source changes

The local `origin` remote points to `https://github.com/ajayfjfj/floorplan-studio.git`. Commit reviewed changes and push `main` using an authenticated GitHub account with access to this private repository. Source pushes run the project checks automatically.

## GitHub Pages hosting

Website publication is separate from repository access and is currently disabled. Do not run the publishing workflow until a website deployment is requested. Confirm the account's Pages eligibility and intended website audience before enabling it.

1. Open repository **Settings → Pages** and choose **GitHub Actions** as the source.
2. In **Actions**, run **Publish demo to GitHub Pages** manually.
3. Open the deployment URL returned by that workflow.

The workflow publishes `web/dist/` directly. It requires no model service, build step, or secret. Relative URLs support both a repository subpath and a custom domain. Source pushes run checks; they do not trigger Pages publication automatically.

Refer to [GitHub's custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) for repository settings and deployment permissions.
