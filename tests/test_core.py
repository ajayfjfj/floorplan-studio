import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from floorplan.common import letterbox
from floorplan.data import svg_objects, rasterize, prepare
from floorplan.geometry import vectorize
from floorplan.evaluate import validate_json
from floorplan.metrics import confusion_matrix, pixel_metrics, instance_counts
from floorplan.model import segmentation_loss


class CoreTests(unittest.TestCase):
    def test_svg_mapping_transform_and_precedence(self):
        svg = '<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg"><g class="Space Bedroom"><polygon points="0,0 100,0 100,100 0,100"/></g><g transform="translate(10,0)"><g id="Wall"><polygon points="10,0 20,0 20,100 10,100"/></g><g id="Door"><polygon points="10,40 20,40 20,60 10,60"/></g></g><g class="Space Outdoor"><polygon points="70,70 90,70 90,90 70,90"/></g></svg>'
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.svg"
            p.write_text(svg)
            objs = svg_objects(p, (100, 100))
            mask = np.asarray(rasterize(objs, (100, 100)))
        self.assertEqual(mask[10, 25], 2)
        self.assertEqual(mask[50, 25], 3)
        self.assertEqual(mask[50, 50], 1)
        self.assertEqual(len(objs), 3)

    def test_letterbox_ignores_padding(self):
        _, mask, box = letterbox(
            Image.new("RGB", (100, 50)), 64, Image.new("L", (100, 50), 2)
        )
        self.assertEqual(box, (0, 16, 64, 32))
        self.assertEqual(np.asarray(mask)[0, 0], 255)
        cm = confusion_matrix(np.asarray(mask), np.full((64, 64), 2))
        self.assertEqual(cm.sum(), 64 * 32)
        self.assertEqual(pixel_metrics(cm)["pixel_accuracy"], 1)

    def test_cubicasa_viewport_does_not_rescale_aligned_points(self):
        svg = '<svg width="50" height="50" xmlns="http://www.w3.org/2000/svg"><g id="Wall"><polygon points="10,10 20,10 20,20 10,20"/></g></svg>'
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.svg"
            p.write_text(svg)
            points = svg_objects(p, (100, 100))[0]["polygon"]
        self.assertEqual(points[0], [10.0, 10.0])

    def test_holes_opening_span_and_schema(self):
        mask = np.zeros((100, 200), np.uint8)
        mask[10:90, 10:190] = 2
        mask[20:80, 20:180] = 1
        mask[40:50, 10:15] = 3
        value = vectorize(mask, np.ones_like(mask, dtype=float))
        validate_json(value)
        self.assertTrue(
            any(e["class"] == "wall" and e["holes"] for e in value["elements"])
        )
        door = next(e for e in value["elements"] if e["class"] == "door")
        self.assertEqual(len(door["opening_span"]), 2)
        validate_json(vectorize(np.zeros((10, 10), np.uint8)))

    def test_instance_matching_penalizes_merges(self):
        a = np.zeros((10, 10), bool)
        a[1:3, 1:3] = True
        b = np.zeros((10, 10), bool)
        b[6:8, 6:8] = True
        self.assertEqual(instance_counts([a, b], [a, b]).tolist(), [2, 0, 0])
        self.assertEqual(instance_counts([a, b], [a | b]).tolist(), [1, 0, 1])

    def test_loss_backpropagates_and_ignores_padding(self):
        torch.manual_seed(1)
        logits = torch.randn(2, 5, 8, 8, requires_grad=True)
        target = torch.randint(0, 5, (2, 32, 32))
        target[:, :2] = 255
        loss = segmentation_loss(logits, target, torch.ones(5))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_duplicate_images_keep_test_then_validation_priority(self):
        import json

        svg = '<svg width="32" height="32" xmlns="http://www.w3.org/2000/svg"><g id="Wall"><polygon points="2,2 8,2 8,28 2,28"/></g></svg>'
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "raw"
            root.mkdir()
            for split, color in [("train", "red"), ("val", "blue"), ("test", "green")]:
                ids = []
                for name, c in [
                    (split + "_unique", color),
                    (split + "_duplicate", "white"),
                ]:
                    folder = root / name
                    folder.mkdir()
                    ids.append(name)
                    Image.new("RGB", (32, 32), c).save(folder / "F1_scaled.png")
                    (folder / "model.svg").write_text(svg)
                (root / f"{split}.txt").write_text("\n".join(ids))
            report = prepare(root, Path(d) / "prepared", 32)
            self.assertEqual(report["splits"], {"test": 2, "val": 1, "train": 1})
            self.assertEqual(len(report["duplicates"]), 2)
            self.assertTrue(
                all(x["kept_split"] == "test" for x in report["duplicates"])
            )


if __name__ == "__main__":
    unittest.main()
