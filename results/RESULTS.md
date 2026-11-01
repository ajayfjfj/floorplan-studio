# Measured evaluation results

Evaluated **400 unseen test plans** at a 512 × 512 inference canvas.

- Pixel accuracy: 0.9471
- Foreground mIoU: 0.6979
- Mean room-count absolute error: 4.715
- Schema-valid JSON: 100.0%
- Median inference time: 0.029 seconds

| Class | IoU | Dice | Instance F1 at IoU ≥ 0.5 |
|---|---:|---:|---:|
| room | 0.9193 | 0.9580 | 0.3887 |
| wall | 0.7005 | 0.8239 | 0.0748 |
| door | 0.4966 | 0.6637 | 0.4584 |
| window | 0.6750 | 0.8060 | 0.5863 |

## Evidence of actual fine-tuning

Completed 7857 optimizer steps over 30 epochs. Maximum absolute change in the first encoder parameter during the run: 0.01448383.

The starting checkpoint uses a pretrained backbone and a newly initialized five-class head. The saved initial validation result measures that unadapted task head; it is not the original ADE20K model evaluated with its original labels.

## Interpretation and limitations

Pixel accuracy can be inflated by the large background area; foreground and class IoU are the primary segmentation measures. Instance metrics compare individual source SVG polygons with connected regions in the predicted semantic mask. Connected wall networks and adjoining rooms can merge, so semantic segmentation does not guarantee correct topology or room counts. Small openings can disappear during resizing. Normalized geometry requires external physical scale and height information for 3D construction.

The first eight test examples are exported in manifest order, without selecting for quality. Sample images and derived annotations originate from CubiCasa5K under CC BY-NC 4.0.

## Lowest-scoring test cases

These are selected by per-plan foreground mIoU, including every architectural category present in either ground truth or prediction. They are reported to make failure analysis reproducible.

| Sample ID | Foreground mIoU | Absolute room-count error |
|---|---:|---:|
| high_quality_architectural/7705 | 0.1069 | 6 |
| high_quality_architectural/5135 | 0.2585 | 20 |
| high_quality_architectural/4813 | 0.2772 | 13 |
| high_quality_architectural/6652 | 0.2963 | 4 |
| high_quality_architectural/9543 | 0.3298 | 11 |
