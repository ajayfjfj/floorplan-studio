import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from .common import BASE_MODEL, BASE_REVISION, device_for, save_json, seed_everything
from .model import Plans, new_model, segmentation_loss
from .metrics import confusion_matrix, pixel_metrics


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    cm = np.zeros((5, 5), dtype=np.int64)
    for x, y in loader:
        logits = model(pixel_values=x.to(device)).logits
        p = (
            F.interpolate(
                logits, size=y.shape[-2:], mode="bilinear", align_corners=False
            )
            .argmax(1)
            .cpu()
            .numpy()
        )
        cm += confusion_matrix(y.numpy(), p)
    return pixel_metrics(cm)


def train(
    data,
    output,
    epochs=30,
    batch_size=4,
    lr=6e-5,
    seed=42,
    patience=6,
    device="auto",
    resume=False,
    workers=2,
):
    if epochs < 1 or batch_size < 1 or lr <= 0 or patience < 1 or workers < 0:
        raise ValueError(
            "Epochs, batch size, learning rate and patience must be positive; workers must be nonnegative"
        )
    seed_everything(seed)
    device = device_for(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    audit = json.loads((Path(data) / "audit.json").read_text())
    config = {
        "base_model": BASE_MODEL,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "patience": patience,
        "device": str(device),
        "size": audit["size"],
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "loss": "weighted cross entropy + foreground soft Dice",
        "splits": audit["splits"],
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    config["base_revision"] = BASE_REVISION
    save_json(output / "training_config.json", config)
    trainset, valset = Plans(data, "train", True), Plans(data, "val")
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    valloader = DataLoader(valset, batch_size=batch_size, num_workers=workers)
    counts = np.maximum(np.array(audit["pixel_counts"], dtype=float), 1)
    weights = 1 / np.sqrt(counts / counts.sum())
    weights /= weights.mean()
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    save_json(output / "class_weights.json", weights.cpu().tolist())
    last = output / "last"
    model = (
        SegformerForSemanticSegmentation.from_pretrained(last)
        if resume and last.exists()
        else new_model()
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = max(1, epochs * len(trainloader))
    warmup = max(1, int(total * 0.05))
    schedule = lambda step: (
        max(0.0, step / warmup)
        if step < warmup
        else max(0.0, 1 - (step - warmup) / max(1, total - warmup)) ** 0.9
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start = 0
    best = -1.0
    bad = 0
    history = []
    global_step = 0
    if resume and (output / "optimizer.pt").exists():
        state = torch.load(
            output / "optimizer.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start = state["epoch"] + 1
        best = state["best"]
        bad = state["bad"]
        global_step = state["global_step"]
        history = json.loads((output / "history.json").read_text())
    if not history:
        before = validate(model, valloader, device)
        save_json(output / "before_finetuning_val.json", before)
        model.save_pretrained(output / "initial")
    initial_encoder = next(model.segformer.parameters()).detach().cpu().clone()
    started = time.time()
    for epoch in range(start, epochs):
        model.train()
        running = 0.0
        seen = 0
        for step, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
                dtype=torch.float16,
            ):
                logits = model(pixel_values=x).logits
                loss = segmentation_loss(logits, y, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
                global_step += 1
            running += loss.item() * len(x)
            seen += len(x)
            if step % 100 == 0:
                print(
                    f"epoch={epoch+1} batch={step}/{len(trainloader)} loss={loss.item():.4f}",
                    flush=True,
                )
        val = validate(model, valloader, device)
        score = val["mean_iou_foreground"]
        improved = score > best
        if improved:
            best = score
            bad = 0
            model.save_pretrained(output / "best")
        else:
            bad += 1
        record = {
            "epoch": epoch + 1,
            "train_loss": running / max(1, seen),
            "val_miou_foreground": score,
            "val_pixel_accuracy": val["pixel_accuracy"],
            "elapsed_seconds": time.time() - started,
            "global_step": global_step,
        }
        history.append(record)
        save_json(output / "history.json", history)
        save_json(output / "validation_latest.json", val)
        model.save_pretrained(last)
        state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best": best,
            "bad": bad,
            "global_step": global_step,
        }
        torch.save(state, output / "optimizer.pt")
        delta = float(
            (next(model.segformer.parameters()).detach().cpu() - initial_encoder)
            .abs()
            .max()
        )
        save_json(
            output / "finetuning_evidence.json",
            {
                "optimizer_steps": global_step,
                "encoder_max_absolute_change_this_run": delta,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "best_validation_miou": best,
                "epochs_completed": epoch + 1,
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if device.type == "cuda"
                    else str(device)
                ),
            },
        )
        print(json.dumps(record), flush=True)
        if bad >= patience:
            break
    return history
