# Dataset and preprocessing

Source: [CubiCasa5K official repository](https://github.com/CubiCasa/CubiCasa5k), [Zenodo record](https://zenodo.org/records/2613548), DOI `10.5281/zenodo.2613548`. The dataset has 5,000 floor-plan samples. The code reads official train/validation/test lists rather than randomly re-splitting floor plans. The generated audit records the observed split counts and category mapping; the exact manifests are retained in the prepared cache.

## Source annotation mapping

| Source selector | Project category |
|---|---|
| `g` with class token `Space`, excluding Outdoor and Background | room (1) |
| `g id="Wall"` with direct polygon | wall (2) |
| `g id="Door"` with direct polygon | door (3) |
| `g id="Window"` with direct polygon | window (4) |
| Outside labeled architecture, furnishings, railings | background (0), unless covered by a room polygon |

Furniture inside room polygons remains room. Outdoor regions are excluded from the room ontology. SVG polygon geometry, including inherited affine transforms, is rasterized in aligned `F1_scaled.png` pixel coordinates. **Do not rescale polygon coordinates using the SVG width/viewBox**: a checked official sample has a declared SVG viewport different from the raster dimensions. The official `House` parser also uses those polygon points directly. This project uses a new compact parser with the same relevant selectors rather than executing the legacy training code.

Rasterization order: room, wall, door, window. Later categories overwrite earlier ones. This represents openings explicitly where they overlap a wall. A 512 × 512 letterbox preserves aspect ratio; label resizing uses nearest-neighbor interpolation and padding uses ignore index 255. No text OCR, test annotations, or source SVG is used at inference.

## Leakage controls and audit

Preparation rejects repeated IDs within a split and overlapping IDs across splits. An actual cross-split image duplicate was found in the official lists (`high_quality_architectural/642` was encountered in validation after an identical training raster). The final preparation removes exact RGB pixel duplicates using **test > validation > training** priority, retaining the first occurrence within a split. This preserves held-out examples and removes copies from lower-priority splits before training. No model predictions or scores are used to choose exclusions.

The audit records original and retained split counts, every excluded ID and retained counterpart, image hashes, whether the SVG bytes also match, source-to-project mappings, and training-only pixel frequencies used for loss weighting. Source-image hashes include RGB pixel bytes and dimensions. The check does not establish building-level independence where the public dataset lacks building identifiers. Near-duplicate plans remain a possible limitation.

The completed audit retained **4,179 training / 400 validation / 400 test** plans. It removed 15 repeated training images, four training copies of validation images, and two training copies of test images. All 21 exclusions came from training. Sixteen excluded copies also had identical SVG bytes; a different SVG byte hash alone does not establish different semantic labels.

## Licensing

CubiCasa5K is released under CC BY-NC 4.0. Download the dataset separately; do not commit it. Locally saved example images and outputs must retain attribution and the non-commercial license notice. A company assessment is not permission for commercial deployment. Dataset usage and sharing should be reviewed against the upstream terms before external publication.

Reference: Kalervo et al., *CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis* (2019), [paper](https://arxiv.org/abs/1904.01920).
