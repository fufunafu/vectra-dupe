"""Sanity checks for vectra3d.face_regions on a frontal portrait's landmarks."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vectra3d.face_regions import REGIONS, region_polygons, region_masks, _canonical_mirror_map, MIDLINE

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "portrait_landmarks.json")


def _L():
    return np.asarray(json.load(open(FIX))["landmarks"], float)


def test_mirror_table_is_involution():
    m = _canonical_mirror_map()
    assert len(m) == 478
    assert all(m[m[i]] == i for i in range(478))
    assert all(m[i] == i for i in MIDLINE)
    for a, b in [(33, 263), (133, 362), (61, 291), (234, 454), (50, 280)]:
        assert m[a] == b and m[b] == a


def test_every_region_yields_polygons_with_area():
    L = _L()
    polys = region_polygons(L)
    assert set(polys) == set(REGIONS)
    for rid, ps in polys.items():
        assert len(ps) == (2 if REGIONS[rid]["bilateral"] else 1), rid
        for p in ps:
            x, y = p[:, 0], p[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            assert area > 500, (rid, area)


def test_bilateral_regions_are_symmetric_on_symmetric_face():
    L = _L()
    xm = L[MIDLINE, 0].mean()
    for rid, ps in region_polygons(L).items():
        if len(ps) == 2:
            a, b = ps[0][:, 0].mean(), ps[1][:, 0].mean()
            assert abs((a - xm) + (b - xm)) < 12, (rid, a, b)  # px, portrait is ~symmetric


def test_masks_cover_face_without_gross_overlap():
    L = _L()
    masks = region_masks(L, (1152, 896))
    total = sum(m.sum() for m in masks.values())
    union = np.zeros((1152, 896), bool)
    for m in masks.values():
        union |= m
    assert union.sum() > 150_000
    assert total / union.sum() < 1.08          # neighbours share edges, not areas


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
