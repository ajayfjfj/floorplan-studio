import unittest

import numpy as np

from floorplan.evaluate import validate_json
from floorplan.geometry import vectorize


class GeometryEdgeTests(unittest.TestCase):
    def convert(self, mask, confidence=None):
        value = vectorize(mask, confidence, "geometry-edge-case")
        validate_json(value)
        return value

    def test_empty_mask_returns_no_elements(self):
        value = self.convert(np.zeros((32, 48), np.uint8))
        self.assertEqual(value["elements"], [])

    def test_wall_ring_preserves_interior_hole(self):
        mask = np.zeros((64, 64), np.uint8)
        mask[8:56, 8:56] = 2
        mask[16:48, 16:48] = 0
        walls = self.convert(mask)["elements"]
        self.assertEqual(len(walls), 1)
        self.assertEqual(walls[0]["class"], "wall")
        self.assertEqual(len(walls[0]["holes"]), 1)
        self.assertGreaterEqual(len(walls[0]["holes"][0]), 3)

    def test_normalized_coordinates_include_edges_and_stay_in_bounds(self):
        mask = np.zeros((64, 96), np.uint8)
        mask[:16, :24] = 1
        mask[28:58, 34:62] = 2
        mask[34:52, 40:56] = 0
        mask[5:10, 60:84] = 3
        mask[48:, 90:] = 4
        confidence = np.full(mask.shape, 0.8, dtype=np.float32)
        elements = self.convert(mask, confidence)["elements"]
        self.assertEqual(
            {e["class"] for e in elements}, {"room", "wall", "door", "window"}
        )
        coordinates = []
        for element in elements:
            points = element["polygon"] + element.get("opening_span", [])
            points += [point for hole in element["holes"] for point in hole]
            coordinates.extend(value for point in points for value in point)
            coordinates.extend(element["bbox"])
        self.assertIn(0.0, coordinates)
        self.assertIn(1.0, coordinates)
        self.assertTrue(
            all(np.isfinite(value) and 0 <= value <= 1 for value in coordinates)
        )

    def test_doors_and_windows_have_nonzero_opening_spans(self):
        for class_id, name in [(3, "door"), (4, "window")]:
            with self.subTest(element=name):
                mask = np.zeros((64, 80), np.uint8)
                mask[20:28, 15:55] = class_id
                elements = self.convert(mask)["elements"]
                self.assertEqual(len(elements), 1)
                self.assertEqual(elements[0]["class"], name)
                span = elements[0]["opening_span"]
                self.assertEqual(len(span), 2)
                self.assertNotEqual(span[0], span[1])

    def test_speck_below_default_area_threshold_is_dropped(self):
        mask = np.zeros((512, 512), np.uint8)
        # Four pixels form a valid contour but are below the default area limit of 12.
        mask[20:22, 20:22] = 2
        self.assertEqual(self.convert(mask)["elements"], [])


if __name__ == "__main__":
    unittest.main()
