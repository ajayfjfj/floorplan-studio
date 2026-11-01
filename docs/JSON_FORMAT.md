# Spatial output contract

`prediction.json` conforms to `src/floorplan/prediction.schema.json`, version 1.0. All geometry is derived from model predictions; annotation files are only used for training/evaluation.

- `image`: source image identifier, width, and height in pixels.
- `coordinates`: normalized x/y, origin at the top left; positive x points right and positive y points down. Convert to source pixels with `x_px = x * (width - 1)`, `y_px = y * (height - 1)`.
- `meters_per_pixel`: null unless a consumer supplies calibrated scale.
- `elements`: room, wall, door, or window regions with stable per-image IDs; polygon rings have implicit closure and may contain interior holes. Ring winding is not guaranteed. `bbox` is `[xmin, ymin, xmax, ymax]`.
- `area_fraction`: number of region pixels divided by image pixel count. `confidence`: average winning class probability for the region; this is not calibrated confidence. Simplification may change polygon area slightly.
- `opening_span`: estimated major-axis endpoints of an opening's minimum-area rectangle. It is a geometric estimate, not a detected hinge or swing direction.

Rooms are generic room regions. The model does not predict room function, dimensions, structural engineering properties, or building-code compliance. IDs are generated in deterministic contour order for a given mask; they are not persistent IDs across different model outputs.

For 3D reconstruction, convert polygons to source pixels first, then apply the physical scale equally on both axes. Since x and y are normalized independently, normalized Euclidean distance alone is not a physical length. Extrude wall footprints while respecting holes; apply separately supplied wall/opening heights. Treat region merges and uncertain opening placement as explicit review cases. This project supplies perception geometry and does not implement the downstream 3D engine.
