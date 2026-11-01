import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from transformers import SegformerForSemanticSegmentation
from .common import CLASSES, BASE_MODEL, BASE_REVISION, image_tensor, letterbox


class Plans(Dataset):
    def __init__(self, root, split, augment=False):
        self.root = Path(root)
        self.rows = json.loads((self.root / f"{split}.json").read_text())
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        folder = self.root / self.rows[index]["path"]
        with Image.open(folder / "image.png") as f:
            image = f.convert("RGB")
        with Image.open(folder / "mask.png") as f:
            mask = f.copy()
        if self.augment:
            k = random.randrange(4)
            if k:
                image = image.rotate(k * 90)
                mask = mask.rotate(k * 90)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        return image_tensor(image), torch.from_numpy(
            np.asarray(mask, dtype=np.int64).copy()
        )


def new_model():
    return SegformerForSemanticSegmentation.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        num_labels=5,
        id2label=dict(enumerate(CLASSES)),
        label2id={n: i for i, n in enumerate(CLASSES)},
        ignore_mismatched_sizes=True,
    )


def segmentation_loss(logits, target, weights):
    logits = F.interpolate(
        logits, size=target.shape[-2:], mode="bilinear", align_corners=False
    ).float()
    ce = F.cross_entropy(logits, target, weight=weights, ignore_index=255)
    valid = target != 255
    truth = (
        F.one_hot(target.clamp(max=4), 5).permute(0, 3, 1, 2).float() * valid[:, None]
    )
    probs = logits.softmax(1) * valid[:, None]
    axes = (0, 2, 3)
    dice = (2 * (probs * truth).sum(axes) + 1) / (probs.sum(axes) + truth.sum(axes) + 1)
    return ce + (1 - dice[1:]).mean()


@torch.inference_mode()
def predict(model, image, device, size=512):
    canvas, _, (left, top, w, h) = letterbox(image, size)
    tensor = image_tensor(canvas).unsqueeze(0).to(device)
    logits = model(pixel_values=tensor).logits
    logits = F.interpolate(
        logits, size=(size, size), mode="bilinear", align_corners=False
    )
    logits = logits[:, :, top : top + h, left : left + w]
    logits = F.interpolate(
        logits, size=(image.height, image.width), mode="bilinear", align_corners=False
    )
    probabilities = logits.softmax(1)[0]
    confidence, mask = probabilities.max(0)
    return mask.cpu().numpy().astype(np.uint8), confidence.cpu().numpy()
