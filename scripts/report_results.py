"""Generate plots and a factual Markdown report from measured artifacts."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results"
    output.mkdir(exist_ok=True)
    report = json.loads((output / "test/metrics.json").read_text())
    history = json.loads((root / "artifacts/segformer/history.json").read_text())
    evidence = json.loads(
        (root / "artifacts/segformer/finetuning_evidence.json").read_text()
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    axes[0].plot(
        [x["epoch"] for x in history],
        [x["train_loss"] for x in history],
        color="#34495e",
    )
    axes[0].set(xlabel="Epoch", ylabel="Training loss", title="Fine-tuning loss")
    axes[1].plot(
        [x["epoch"] for x in history],
        [x["val_miou_foreground"] for x in history],
        color="#268495",
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="Foreground mIoU",
        title="Validation selection",
        ylim=(0, 1),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)
    per = report["pixel"]["per_class"]
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    names = ["room", "wall", "door", "window"]
    ax.bar(
        names,
        [per[x]["iou"] or 0 for x in names],
        color=["#80bcc7", "#34495e", "#e7a269", "#72a887"],
    )
    ax.set(ylabel="IoU", ylim=(0, 1), title="Held-out test segmentation")
    fig.savefig(output / "test_class_iou.png", dpi=180)
    plt.close(fig)
    lines = [
        "# Measured evaluation results",
        "",
        f"Evaluated **{report['plans']} unseen {report['split']} plans** at a {report['resolution']} × {report['resolution']} inference canvas.",
        "",
        f"- Pixel accuracy: {report['pixel']['pixel_accuracy']:.4f}",
        f"- Foreground mIoU: {report['pixel']['mean_iou_foreground']:.4f}",
        f"- Mean room-count absolute error: {report['room_count_mae']:.3f}",
        f"- Schema-valid JSON: {report['json_valid_fraction']:.1%}",
        f"- Median inference time: {report['median_inference_seconds']:.3f} seconds",
        "",
        "| Class | IoU | Dice | Instance F1 at IoU ≥ 0.5 |",
        "|---|---:|---:|---:|",
    ]
    for name in names:
        p = per[name]
        ins = report["instances_iou_0_5"][name]
        iou = "N/A" if p["iou"] is None else f"{p['iou']:.4f}"
        lines.append(f"| {name} | {iou} | {p['dice']:.4f} | {ins['f1']:.4f} |")
    lines += [
        "",
        "## Evidence of actual fine-tuning",
        "",
        f"Completed {evidence['optimizer_steps']} optimizer steps over {evidence['epochs_completed']} epochs. Maximum absolute change in the first encoder parameter during the run: {evidence['encoder_max_absolute_change_this_run']:.8f}.",
        "",
        "The starting checkpoint uses a pretrained backbone and a newly initialized five-class head. The saved initial validation result measures that unadapted task head; it is not the original ADE20K model evaluated with its original labels.",
        "",
        "## Interpretation and limitations",
        "",
        "Pixel accuracy can be inflated by the large background area; foreground and class IoU are the primary segmentation measures. Instance metrics compare individual source SVG polygons with connected regions in the predicted semantic mask. Connected wall networks and adjoining rooms can merge, so semantic segmentation does not guarantee correct topology or room counts. Small openings can disappear during resizing. Normalized geometry requires external physical scale and height information for 3D construction.",
        "",
        "The first eight test examples are exported in manifest order, without selecting for quality. Sample images and derived annotations originate from CubiCasa5K under CC BY-NC 4.0.",
    ]
    per_plan = json.loads((output / "test/per_plan.json").read_text())
    ranked = sorted(
        [r for r in per_plan if "foreground_miou" in r],
        key=lambda r: r["foreground_miou"],
    )
    if ranked:
        lines += [
            "",
            "## Lowest-scoring test cases",
            "",
            "These are selected by per-plan foreground mIoU, including every architectural category present in either ground truth or prediction. They are reported to make failure analysis reproducible.",
            "",
            "| Sample ID | Foreground mIoU | Absolute room-count error |",
            "|---|---:|---:|",
        ]
        for r in ranked[:5]:
            lines.append(
                f"| {r['sample_id']} | {r['foreground_miou']:.4f} | {r['room_count_absolute_error']} |"
            )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
