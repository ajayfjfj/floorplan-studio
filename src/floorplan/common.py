import json
import random
from pathlib import Path
import numpy as np
from PIL import Image

CLASSES = ["background", "room", "wall", "door", "window"]
COLORS = np.array(
    [[255, 255, 255], [171, 219, 227], [51, 65, 85], [238, 139, 72], [86, 169, 113]],
    dtype=np.uint8,
)
BASE_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
BASE_REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name="auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def letterbox(image, size=512, mask=None):
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
    target = None
    if mask is not None:
        target = Image.new("L", (size, size), 255)
        target.paste(mask.resize((nw, nh), Image.Resampling.NEAREST), (left, top))
    return canvas, target, (left, top, nw, nh)


def image_tensor(image):
    import torch

    arr = np.asarray(image, dtype=np.float32) / 255
    arr = (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def overlay(image, mask):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    colored = COLORS[np.minimum(mask, 4)]
    out = np.where((mask > 0)[..., None], 0.55 * rgb + 0.45 * colored, rgb)
    return Image.fromarray(np.uint8(np.clip(out, 0, 255)))
