# Floorplan Studio

A static browser interface for the trained floor-plan model. No API keys, Python server, or Colab session are needed to use the demo.

## Run locally

From the repository root:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory web/dist
```

Open http://localhost:8765. Opening `index.html` directly as a file will not work: the model and sample catalog use HTTP fetches.

## What it does

- Eight CubiCasa5K test images can be selected as inputs for live detection.
- The **Sample floor plans** gallery is included directly in the HTML. **Download sample image** saves the selected input as PNG.
- PNG, JPEG, and WebP uploads run the fine-tuned model locally through ONNX Runtime Web.
- Class filters, overlay opacity, and region inspection help review the output.
- JSON output can be viewed, copied, and downloaded. **Download visual overview (PNG)** saves the plan with the current overlay, a class legend, region counts, and the image name.
- The layout supports desktop and mobile screens. Speed depends on the visitor's device.

Selecting a sample or uploading an image shows the input image only. Results, region counts, and downloads stay empty until **Run detection** finishes. Each run executes the trained model; no saved predictions are loaded or distributed with the interface. Switching images or starting another run clears previous output. The worker downloads roughly 36 MB of model and runtime assets on its first run; later runs reuse the model session but recompute predictions. There is no image-upload endpoint or analytics script.

## Browser pipeline

1. Decode the image and composite transparency on white.
2. Letterbox to 512 × 512, scale RGB to 0–1, and apply ImageNet normalization.
3. Run the fixed-shape FP32 ONNX model in a Web Worker with single-threaded WASM.
4. Bilinearly resize logits, remove padding, apply softmax, and choose classes.
5. Trace connected components, polygon holes, bounding boxes, and opening spans with OpenCV.js.
6. Map geometry to normalized coordinates and keep the original image dimensions in JSON.

Browser geometry uses a raster capped at a 1024-pixel longest edge. Native canvas resizing can differ slightly from PIL. The browser is therefore an interactive demo of the same weights, not an exact replacement for the Python evaluation pipeline. Reported 400-plan metrics come from the Python implementation; evaluation examples remain under `results/` and are not loaded by the interface. Probabilities are not calibrated confidence scores.

Uploads are limited to 10 MB and 16 megapixels. Use a recent browser with WebAssembly SIMD support. A slow device may take several seconds or longer. PDF, SVG, real-world scale inference, and 3D rendering are outside this interface.

## Rebuild model and assets

The repository includes the browser model and vendor assets, so a normal checkout can be served immediately. To export again:

```bash
python -m pip install -e '.[browser]'
python scripts/export_browser_model.py
python scripts/prepare_web.py
python scripts/verify_web.py
node tests/test_browser_geometry.cjs
```

The export script checks ONNX against PyTorch on all eight samples. `results/browser_export_verification.json` records numerical error, argmax agreement, model size, and SHA-256. The current export has 100% argmax agreement on all eight images.

`python scripts/prepare_web.py --download-runtimes` can restore pinned runtime files. The OpenCV download uses the official rolling URL with an exact checksum; if upstream changes, it fails rather than silently substituting a different build. The committed copy remains usable.

## Static hosting

The interface is published with GitHub Pages: [Open Floorplan Studio](https://ajayfjfj.github.io/floorplan-studio/).

Publish the contents of `web/dist/` on any static host that serves `.wasm` as `application/wasm` and `.mjs` as JavaScript. All application paths are relative, including model and worker URLs, so repository subpaths work. No build command or environment variables are required.

For GitHub Pages, select **GitHub Actions** under repository Settings → Pages, then run the manually triggered **Publish demo to GitHub Pages** workflow. The Pages workflow is manual so pushing the repository does not publish an additional site automatically. [GitHub's custom workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

Bundled runtimes follow [ONNX Runtime Web's deployment guidance](https://onnxruntime.ai/docs/tutorials/web/deploy.html). See the demo's license page and the repository's third-party notices for dataset, model, and runtime terms.
