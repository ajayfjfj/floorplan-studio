# Third-party sources

## CubiCasa5K

Copyright 2019, CubiCasa5K authors. Licensed under [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/). Official source: https://github.com/CubiCasa/CubiCasa5k . Dataset record: https://zenodo.org/records/2613548 . Cite Kalervo, Ylioinas, Häikiö, Karhu, Kannala (2019).

Dataset-derived examples and trained artifacts are identified separately from project-authored source code. The MIT license for project code does not replace upstream restrictions or grant commercial rights to data, model weights, or derived artifacts.

## SegFormer

Pretrained checkpoint: https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512 . Architecture by Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers* (2021). Upstream license and terms: https://github.com/NVlabs/SegFormer/blob/master/LICENSE . The upstream model license must be retained with redistributed checkpoints.

The Transformers, PyTorch, NumPy, Pillow, OpenCV, SciPy, Matplotlib, Requests, tqdm, and jsonschema Python packages retain their own licenses. They are installed as dependencies rather than vendored into the project.

## Browser runtimes

The static demo bundles ONNX Runtime Web 1.22.0 (Microsoft and contributors, MIT) and OpenCV.js 4.13.0 (OpenCV contributors, Apache 2.0). Their license files and ONNX Runtime third-party notices are included under `web/dist/vendor/`. Runtime download URLs and content hashes are recorded in `web/dist/asset-manifest.json` and `scripts/prepare_web.py`.

