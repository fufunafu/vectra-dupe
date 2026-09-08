"""Filler-response regions defined on MediaPipe FaceLandmarker (478-point) indices.

Companion to docs/filler-region-rules.md and docs/filler-regions.json. Regions are
authored on the subject's RIGHT side (viewer's left in a frontal image, the low
MediaPipe indices) and mirrored to the left side through a symmetric index map.

    from vectra3d.face_regions import region_polygons
    polys = region_polygons(landmarks)          # {region_id: [np.ndarray (N,2 or 3), ...]}

`landmarks` is any (478, 2) or (478, 3) array (pixels or metric mm); polygons come back
in the same space, one per side for bilateral regions. The mirror map is derived from
the canonical topology, not from any particular face, so it works on asymmetric subjects.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Region definitions (subject's right side). Index order traces the polygon.
# Boundaries follow the anatomy in docs/filler-region-rules.md §5.
# --------------------------------------------------------------------------- #
REGIONS: dict[str, dict] = {
    # MediaPipe's mesh ends at the mid-forehead; (index, anchor, k) extrapolates a
    # vertex away from `anchor` by factor k so the region reaches the hairline.
    "forehead": dict(
        bilateral=False, cls="P",
        idx=[(10, 151, 2.7), (109, 108, 2.6), (67, 69, 2.4), (103, 104, 1.7), 54,
             68, 63, 105, 66, 107, 9, 336, 296, 334, 293, 298,
             284, (332, 333, 1.7), (297, 299, 2.4), (338, 337, 2.6)]),
    "glabella": dict(
        bilateral=False, cls="M",
        idx=[107, 9, 336, 285, 417, 168, 193, 55]),
    "brow": dict(                       # brow + upper-lid sulcus (ROOF / preseptal)
        bilateral=True, cls="T",
        idx=[70, 63, 105, 66, 107, 55, 221, 222, 223, 224, 46]),
    "temple": dict(                     # temporal fossa: hairline to lateral orbital rim
        bilateral=True, cls="D",
        idx=[54, 71, 70, 156, 124, 143, 34, 127, 162, (21, 70, 1.6)]),
    "tear_trough": dict(                # medial canthus to mid-pupil, below the rim
        bilateral=True, cls="T",
        idx=[133, 243, 112, 26, 22, 23, 230, 231, 232, 128, 245, 244]),
    "palpebromalar": dict(              # mid-pupil to lateral canthus along the ORL
        bilateral=True, cls="T",
        idx=[23, 24, 110, 25, 130, 226, 31, 228, 229, 230]),
    "zygomatic": dict(                  # malar eminence / prezygomatic space
        bilateral=True, cls="P",
        idx=[111, 228, 229, 230, 119, 101, 36, 50, 123, 116]),
    "deep_medial": dict(                # anterior midface triangle, medial border = NLF
        bilateral=True, cls="D",
        idx=[231, 232, 128, 188, 174, 236, 134, 131, 49, 203, 206, 36, 101, 119, 120]),
    "lateral_cheek": dict(              # buccal / submalar down to the jawline band
        bilateral=True, cls="D",
        idx=[123, 50, 36, 206, 216, 186, 57, 43, 202, 210, 135, 138, 215, 213, 147]),
    "nose": dict(
        bilateral=False, cls="P",
        idx=[168, 193, 122, 196, 3, 236, 134, 131, 49, 102, 64, 98, 97, 2,
             326, 327, 294, 331, 279, 360, 363, 456, 248, 419, 351, 417]),
    "nasolabial": dict(                 # band astride the fold, ala to commissure
        bilateral=True, cls="P",
        idx=[49, 64, 98, 165, 40, 185, 61, 57, 186, 92, 206, 203]),
    "lips": dict(
        bilateral=False, cls="M",
        idx=[61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
             375, 321, 405, 314, 17, 84, 181, 91, 146]),
    "marionette": dict(                 # commissure down the DAO to the prejowl
        bilateral=True, cls="M",
        idx=[61, 43, 202, 210, 211, 170, 149, 140, 32, 194, 182, 106]),
    "chin": dict(
        bilateral=False, cls="P",
        idx=[18, 83, 182, 194, 32, 140, 176, 171, 148, 152,
             377, 400, 369, 262, 418, 406, 313]),
    "jawline": dict(                    # band along the mandibular border, gonion to prejowl
        bilateral=True, cls="P",
        idx=[58, 172, 136, 150, 149, 170, 211, 210, 135, 138]),
    "preauricular": dict(
        bilateral=True, cls="N",
        idx=[127, 234, 93, 132, 177, 147, 123, 116, 34]),
}

# Midline vertices of the canonical mesh (x == 0 in the model). Used to build the
# mirror map and to define the sagittal plane for any landmark set.
MIDLINE = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12,
           13, 14, 15, 16, 17, 18, 200, 199, 175, 152]

_MIRROR: dict[int, int] | None = None


def _canonical_mirror_map() -> dict[int, int]:
    """Index -> mirrored index, from the canonical face-mesh topology.

    Built lazily from the MediaPipe canonical model when available, otherwise from
    a bundled table. Falls back to nearest-neighbour matching on a reflected
    landmark set if neither is present (see `mirror_map_from_landmarks`)."""
    global _MIRROR
    if _MIRROR is not None:
        return _MIRROR
    try:
        from . import _facemesh_mirror  # generated table, see tools/build_mirror_map.py
        _MIRROR = dict(_facemesh_mirror.PAIRS)
    except ImportError:
        _MIRROR = {}
    return _MIRROR


def mirror_map_from_landmarks(L: np.ndarray) -> dict[int, int]:
    """Nearest-neighbour mirror pairs from one near-symmetric landmark set.

    Reflect every point across the sagittal plane fitted to MIDLINE and pair each
    index with the closest reflected index. Good enough to bootstrap the table on
    a frontal, symmetric face; verified pairs are then frozen in _facemesh_mirror."""
    P = np.asarray(L, float)[:, :2]
    mid = P[MIDLINE]
    # sagittal line: through the midline centroid, direction = principal axis
    c = mid.mean(0)
    u, s, vt = np.linalg.svd(mid - c)
    d = vt[0] / np.linalg.norm(vt[0])
    n = np.array([-d[1], d[0]])
    dist = (P - c) @ n
    R = P - 2 * dist[:, None] * n[None, :]
    cost = ((R[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    # one-to-one assignment so the map is an involution even where landmarks
    # nearly coincide (closed inner-lip line, iris rings)
    j = np.argmin(cost, axis=1)
    for i in MIDLINE:                      # midline vertices are their own mirror
        j[i] = i
    # repair non-mutual matches by pairing unresolved indices with each other
    for _ in range(10):
        unresolved = [i for i in range(len(P)) if j[j[i]] != i]
        if not unresolved:
            break
        used = set()
        for i in unresolved:
            if i in used:
                continue
            cands = [k for k in unresolved if k != i and k not in used]
            if not cands:
                break
            k = min(cands, key=lambda k: cost[i, k] + cost[k, i])
            j[i], j[k] = k, i
            used.update((i, k))
    m = {int(i): int(j[i]) for i in range(len(P))}
    return m


def mirror_indices(idx: list, L: np.ndarray | None = None) -> list:
    m = _canonical_mirror_map()
    if not m:
        if L is None:
            raise RuntimeError("No mirror table; pass landmarks to derive one.")
        m = mirror_map_from_landmarks(L)
    return [(m[i[0]], m[i[1]], i[2]) if isinstance(i, tuple) else m[i] for i in idx]


def _vertices(L: np.ndarray, idx: list) -> np.ndarray:
    """Resolve plain indices and (index, anchor, k) extrapolation specs to points."""
    pts = []
    for i in idx:
        if isinstance(i, tuple):
            j, a, k = i
            pts.append(L[a] + k * (L[j] - L[a]))
        else:
            pts.append(L[i])
    return np.asarray(pts)


def region_polygons(L: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Return {region_id: [polygon, ...]} in the landmark coordinate space."""
    L = np.asarray(L, float)
    if L.shape[0] != 478:
        raise ValueError(f"Expected 478 landmarks, got {L.shape}")
    out = {}
    for rid, r in REGIONS.items():
        polys = [_vertices(L, r["idx"])]
        if r["bilateral"]:
            polys.append(_vertices(L, mirror_indices(r["idx"], L)))
        out[rid] = polys
    return out


def region_masks(L: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Rasterise regions to boolean masks (H, W) for 2D landmark sets in pixels."""
    from PIL import Image, ImageDraw
    H, W = shape
    masks = {}
    for rid, polys in region_polygons(L).items():
        im = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(im)
        for p in polys:
            dr.polygon([tuple(map(float, q[:2])) for q in p], fill=255)
        masks[rid] = np.asarray(im) > 0
    return masks
