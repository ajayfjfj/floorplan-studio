import cv2
import numpy as np
from .common import CLASSES


def components(binary, min_area=12):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), 8
    )
    return [
        (labels == i) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]


def vectorize(mask, confidence=None, image_id="plan", min_area_at_512=12):
    h, w = mask.shape
    area_limit = max(2, round(min_area_at_512 * w * h / (512 * 512)))

    def point(p):
        return [
            round(float(p[0]) / max(1, w - 1), 6),
            round(float(p[1]) / max(1, h - 1), 6),
        ]

    def ring(contour):
        approx = cv2.approxPolyDP(contour, max(0.5, 0.001 * max(w, h)), True).reshape(
            -1, 2
        )
        return [point(p) for p in approx]

    elements = []
    for cls in range(1, 5):
        for region in components(mask == cls, area_limit):
            contours, hierarchy = cv2.findContours(
                region.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            if hierarchy is None:
                continue
            for i, contour in enumerate(contours):
                if hierarchy[0, i, 3] != -1:
                    continue
                polygon = ring(contour)
                if len(polygon) < 3:
                    continue
                holes = [
                    ring(c)
                    for j, c in enumerate(contours)
                    if hierarchy[0, j, 3] == i and len(ring(c)) >= 3
                ]
                x, y, bw, bh = cv2.boundingRect(contour)
                item = {
                    "id": f"{CLASSES[cls]}_{sum(e['class']==CLASSES[cls] for e in elements)+1:03d}",
                    "class": CLASSES[cls],
                    "polygon": polygon,
                    "holes": holes,
                    "bbox": [
                        *point((x, y)),
                        *point((min(w - 1, x + bw - 1), min(h - 1, y + bh - 1))),
                    ],
                    "area_fraction": round(float(region.sum()) / (w * h), 8),
                    "confidence": (
                        None
                        if confidence is None
                        else round(float(confidence[region].mean()), 5)
                    ),
                }
                if cls in (3, 4):
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    edges = np.roll(box, -1, axis=0) - box
                    j = int(np.argmax(np.linalg.norm(edges, axis=1)))
                    # Midpoints of opposing short sides define the opening span.
                    direction = edges[j] / max(1e-6, np.linalg.norm(edges[j]))
                    half = float(np.linalg.norm(edges[j])) / 2
                    center = np.array(rect[0])
                    ends = np.array(
                        [center - half * direction, center + half * direction]
                    )
                    ends[:, 0] = np.clip(ends[:, 0], 0, w - 1)
                    ends[:, 1] = np.clip(ends[:, 1], 0, h - 1)
                    item["opening_span"] = [point(p) for p in ends]
                elements.append(item)
    return {
        "schema_version": "1.0",
        "image": {"id": image_id, "width": w, "height": h},
        "coordinates": {
            "system": "normalized",
            "origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "normalization": "x/(width-1), y/(height-1)",
        },
        "meters_per_pixel": None,
        "elements": elements,
        "limitations": [
            "Room categories are generic; room function is not predicted.",
            "Connected mask regions can merge adjacent instances.",
            "Physical dimensions and 3D heights require separate calibration.",
        ],
    }
