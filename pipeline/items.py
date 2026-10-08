"""Detection-derived non-book attributes. No pricing or metric size inference."""


def detection_attributes(observations):
    """Keep decor separate; text-bearing signs remain signs and art needs review."""
    best = max(observations,key=lambda o:o.get("confidence") or 0)
    categories = sorted({o.get("category","unknown") for o in observations})
    category = best.get("category","unknown")
    ambiguous = best["kind"] == "ambiguous"
    issues = []
    if ambiguous:
        issues.append("Object category uncertain; excluded from definite book count")
    if len(categories) > 1:
        issues.append("Conflicting observed non-book categories: " + ", ".join(categories))
    if any(w in category.lower() for w in ["art","portrait","painting","picture"]):
        issues.append("Art/portrait requires appraisal unless claimant confirms a print")
    return {"category":category,"description":best.get("evidence_notes") or None,
        "brand_model":None,"material":None,"dimensions_cm":{"w":None,"h":None,"d":None},
        "classification":"ambiguous" if ambiguous else "non_book","issues":issues}


# --- Consolidation of item instances that are one physical object (pure function, not yet wired into dedup) -----------------------------------------
# Measured on the YT1 dense run against a visual inventory (a development run): the detector emits several categories for one object (vase + pot + plant on
# a potted plant; frame + book_or_frame on a print) and re-detects an object under new ids after tracking breaks. dedup.associate cannot merge the first kind (same-frame
# exclusion) and the second is not reached by cross-frame item matching.
ITEM_FAMILIES = {"plant": "potted", "pot": "potted", "vase": "potted", "frame": "framed", "book_or_frame": "framed", "framed_sign": "framed"}
CONSOLIDATE_DEFAULTS = {
    "colocated_iou": 0.7, "colocated_containment": 0.8, "colocated_min_frac": 0.5,     # rule 1: same frame, near-identical boxes, any categories of one family
    "plane_iou": 0.5, "plane_containment": 0.7, "plane_min_anchor_obs": 3,             # rule 2: same plane position, time-disjoint re-detection; one group must have >= 3 sightings
    "plane_min_appearance": 0.4,                                                       # rule 2 with an appearance callback: same value as cross_view item_min_appearance
    "distinct_max_containment": 0.5,                                                   # two groups that co-occur in a frame and NEVER overlap this much are distinct objects (veto)
}


def _box_stats(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); inter = w * h
    aa, bb = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9), inter / (min(aa, bb) + 1e-9)


def consolidate_item_instances(instances, observations, transforms=None, config=None, appearance=None):
    """Return (groups, links): groups = lists of instance_ids that are one object; links = evidence per merge. Inputs are never modified.

    instances: dedup item instances (instance_id, observation_ids, category). observations: {observation_id: obs} or a list of obs.
    transforms: optional {frame_id: 3x3 frame->plane} (e.g. the LK chain); enables rule 2.
    appearance: optional callable (observation_id_a, observation_id_b) -> similarity in [-1, 1] (e.g. cross_view warped-patch correlation); when given, rule 2 also needs
      plane_min_appearance between the two groups' most confident observations. Rule 2 without it is NOT recommended (chain drift on fast footage joins neighbouring objects).
    Rule 1 (same frame, near-identical boxes, one category family) never needs transforms. Rule 2 (same plane position, time-disjoint) needs one well-supported group.
    Veto: groups that co-occur in a frame and never overlap by distinct_max_containment are distinct objects, and no merge (direct or through a weak bridging group) may put
    them in one component. Titles/category text are never used beyond the family map."""
    import collections, itertools
    cfg = {**CONSOLIDATE_DEFAULTS, **(config or {})}
    obs = observations if isinstance(observations, dict) else {o["observation_id"]: o for o in observations}
    by_frame = {}
    for it in instances:
        d = collections.defaultdict(list)
        for oid in it["observation_ids"]:
            o = obs.get(oid)
            if o and o.get("bbox_xyxy"):
                d[o["frame_id"]].append(o["bbox_xyxy"])
        by_frame[it["instance_id"]] = d
    plane = {}
    if transforms:
        import numpy as np
        for it in instances:
            boxes = []
            for oid in it["observation_ids"]:
                o = obs.get(oid)
                if not o or not o.get("bbox_xyxy") or o["frame_id"] not in transforms:
                    continue
                x0, y0, x1, y1 = o["bbox_xyxy"]
                q = np.hstack([np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), np.ones((4, 1))]) @ np.asarray(transforms[o["frame_id"]], float).T
                q = q[:, :2] / q[:, 2:3]
                boxes.append([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()])
            if boxes:
                plane[it["instance_id"]] = [float(v) for v in np.median(np.array(boxes), axis=0)]
    best = {it["instance_id"]: max(it["observation_ids"], key=lambda i: (obs.get(i) or {}).get("confidence") or 0) for it in instances}
    fam_inst = [it for it in instances if ITEM_FAMILIES.get(it.get("category"))]
    distinct, candidates = set(), []
    for a, b in itertools.combinations(fam_inst, 2):
        ia, ib = a["instance_id"], b["instance_id"]
        shared = set(by_frame[ia]) & set(by_frame[ib])
        stats = {f: [_box_stats(x, y) for x in by_frame[ia][f] for y in by_frame[ib][f]] for f in shared}
        # one clear co-occurring view with the two boxes apart proves two objects; later colliding boxes (blur, re-detection) cannot erase that
        if shared and any(max(s[1] for s in v) < cfg["distinct_max_containment"] for v in stats.values()):
            distinct.add(frozenset((ia, ib)))
        if ITEM_FAMILIES[a["category"]] != ITEM_FAMILIES[b["category"]]:
            continue
        if shared:  # rule 1
            good = sum(1 for f in shared if any(s[0] >= cfg["colocated_iou"] and s[1] >= cfg["colocated_containment"] for s in stats[f]))
            if good / len(shared) >= cfg["colocated_min_frac"]:
                candidates.append((0, 0.0, {"instances": [ia, ib], "rule": "same_frame_colocated", "shared_frames": len(shared), "colocated_frames": good}))
                continue
        if not shared and ia in plane and ib in plane and max(len(a["observation_ids"]), len(b["observation_ids"])) >= cfg["plane_min_anchor_obs"]:  # rule 2: time-disjoint only
            iou, cont = _box_stats(plane[ia], plane[ib])
            if iou >= cfg["plane_iou"] and cont >= cfg["plane_containment"]:
                ev = {"instances": [ia, ib], "rule": "same_plane_position", "plane_iou": round(iou, 3), "plane_containment": round(cont, 3)}
                if appearance is not None:
                    app = float(appearance(best[ia], best[ib]))
                    if app < cfg["plane_min_appearance"]:
                        continue
                    ev["appearance"] = round(app, 3)
                candidates.append((1, -iou, ev))
    candidates.sort(key=lambda c: (c[0], c[1], c[2]["instances"]))
    members = {it["instance_id"]: {it["instance_id"]} for it in instances}
    where = {i: i for i in members}
    links = []
    for _, _, ev in candidates:
        ra, rb = where[ev["instances"][0]], where[ev["instances"][1]]
        if ra == rb:
            links.append(ev)
            continue
        if any(frozenset((x, y)) in distinct for x in members[ra] for y in members[rb]):
            continue  # would join two groups known to be distinct objects
        members[ra] |= members[rb]
        for i in members.pop(rb):
            where[i] = ra
        links.append(ev)
    return sorted(sorted(g) for g in members.values()), links


# --- Non-book robustness: scene gate + neighbour rule ---------------------------------------------------------------------------------------------------------
# Live sweeps start while the user aims the camera: the spine model then fires on a face, two text lines of a wall sign, or two rows of laptop keys (measured on the
# IESTest1 live sweep and the 21:14 keyboard sweep). Real shelf views show rows of parallel spines sharing a baseline; those false positives never do. Nothing is
# deleted: a demoted group becomes `ambiguous` (kept for review, excluded from book_count). A group is demoted only when EVERY sighting fails, so a real book seen
# once in a sparse frame and once in a proper shelf view stays a book.
SCENE_GATE_DEFAULTS = {
    "min_row": 3,             # a frame whose largest co-aligned spine row has fewer members is "not a shelf view" (IESTest1 face/sign frames: 1-2; shelf frames: 3+)
    "angle_deg": 15.0,        # two spines are parallel when their long axes differ by at most this
    "max_gap_widths": 3.0,    # centre distance across the spine axis, in spine widths (of the box named by row_gap_box)
    # Measure that gap in widths of the NARROWER box. With the wider box a ~400 px square vase/flower box/orchid was "in a row" with every spine
    # within 1200 px (YugoTest11). Two real spines side by side sit about one (wider) width apart, which
    # is still within 3 narrower widths unless one is over 3x as wide; and only groups under isolated_max_aspect in every sighting can be demoted for it.
    "row_gap_box": "narrower",  # neighbour rule only ("wider" restores the original wider-box rule); the scene gate's frame_max_row always uses the wider box
    "max_shift_lengths": 0.5,  # centre offset along the spine axis, in spine lengths (shared baseline/row)
    # Neighbour rule: an isolated group is demoted only if it is also never spine-shaped. On YT1 every isolated group under 2.5 was a stool, sign or framed record
    # (aspect 1.0-1.7) and every isolated real book (lone book on a decor shelf, a book lying flat) was 3.0 or more.
    "isolated_max_aspect": 2.5,
}


def _spine_geometry(o):
    import numpy as np
    p = o.get("polygon_xy")
    p = np.asarray(p, float) if p is not None and len(p) == 4 else None
    if p is None:
        x0, y0, x1, y1 = o["bbox_xyxy"]
        p = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    edges = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(edges, axis=1)
    i = int(np.argmax(lengths))
    return p.mean(axis=0), edges[i] / max(float(lengths[i]), 1e-6), float(lengths.max()), float(lengths.min())


def _row_neighbours(a, b, cfg, gap_box=None):
    import numpy as np
    ca, ua, la, wa = a
    cb, ub, lb, wb = b
    if abs(float(ua @ ub)) < np.cos(np.radians(cfg["angle_deg"])):
        return False
    d = cb - ca
    along = abs(float(d @ ua))
    across = abs(float(d @ np.array([-ua[1], ua[0]])))
    width = min(wa, wb) if (gap_box or cfg.get("row_gap_box", "narrower")) == "narrower" else max(wa, wb)
    return across <= cfg["max_gap_widths"] * width and along <= cfg["max_shift_lengths"] * max(la, lb)


def _rows(geom, cfg, gap_box):
    import collections
    parent = list(range(len(geom)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(geom)):
        for j in range(i + 1, len(geom)):
            if _row_neighbours(geom[i], geom[j], cfg, gap_box):
                parent[root(i)] = root(j)
    size = collections.Counter(root(i) for i in range(len(geom)))
    return [size[root(i)] for i in range(len(geom))]


def spine_row_support(observations, config=None):
    """{observation_id: {"row": size of its co-aligned row, "frame_max_row": largest row in its frame}} for kind == 'book' observations.

    A row = connected parallel spines that are side by side (small across-axis gap) and share a baseline (small along-axis shift). Pure geometry of one frame.
    "row" (the neighbour rule) measures the gap in the box chosen by row_gap_box (narrower by default); "frame_max_row" (the scene gate: is this frame a
    shelf view at all) keeps the wider-box gap, so the scene gate is unchanged."""
    import collections
    cfg = {**SCENE_GATE_DEFAULTS, **(config or {})}
    by_frame = collections.defaultdict(list)
    for o in observations:
        if o.get("kind") == "book" and o.get("bbox_xyxy"):
            by_frame[o["frame_id"]].append(o)
    support = {}
    for books in by_frame.values():
        geom = [_spine_geometry(o) for o in books]
        own = _rows(geom, cfg, cfg.get("row_gap_box", "narrower"))
        frame_max = max(_rows(geom, cfg, "wider"))
        for i, o in enumerate(books):
            support[o["observation_id"]] = {"row": own[i], "frame_max_row": frame_max,
                                            "aspect": round(geom[i][2] / max(geom[i][3], 1e-6), 2)}
    return support


def non_book_demotion(observation_ids, support, config=None):
    """Reason dict when a book group must be demoted to ambiguous, else None. Unknown observations count as passing (never demote on missing evidence)."""
    cfg = {**SCENE_GATE_DEFAULTS, **(config or {})}
    rows = [support.get(oid) for oid in observation_ids]
    if not rows or any(r is None for r in rows):
        return None
    best_frame = max(r["frame_max_row"] for r in rows)
    if best_frame < cfg["min_row"]:
        return {"rule": "scene_gate", "category": "book_or_non_shelf_object", "sightings": len(rows), "best_frame_row": best_frame,
                "reason": f"Every sighting ({len(rows)}) is in a frame with fewer than {cfg['min_row']} co-aligned spines (not a shelf view); "
                          "likely a non-book shape seen while aiming; kept for review, not counted as a book"}
    aspect = max(r.get("aspect", 99) for r in rows)
    if max(r["row"] for r in rows) < 2 and aspect < cfg["isolated_max_aspect"]:
        return {"rule": "no_neighbour", "category": "book_or_isolated_object", "sightings": len(rows), "best_frame_row": best_frame, "max_aspect": aspect,
                "reason": f"No adjacent parallel spine in any of its {len(rows)} sighting(s) and never spine-shaped (aspect {aspect:.1f} < {cfg['isolated_max_aspect']}); "
                          "kept for review, not counted as a book"}
    return None


# --- Item over-trigger ------------------------------------------------------------------------------------------------------------------------------------------
# Live Yugo sweeps listed 185 / 146 item lines for a shelf with ~30 decorations (hand check). Three causes, measured there:
# (1) room structure as item lines (`desk` = the shelf board, `shelving` = the bookcase); (2) one object split into many lines: a fast hand-held sweep breaks the
# tracked chain every 2-3 s (35 segments in 119 s), so the same pot is re-detected under a new id per segment and under several labels (plant / vase / cup / basket);
# (3) one-sighting fragments (blurred re-views, half-visible objects). Nothing below touches book groups: book_count is identical by construction.
ITEM_CONTEXT_ONLY = {"desk", "shelving", "floor", "door", "window", "radiator", "cable"}   # conflict evidence for spines, never an inventory line
ITEM_DEDUP_FAMILIES = {**dict.fromkeys(["plant", "pot", "vase", "cup", "basket", "jar", "pot_or_vase"], "potted"),
                       **dict.fromkeys(["frame", "framed_sign", "record", "sign"], "framed")}
ITEM_DEDUP_DEFAULTS = {
    "distinct_max_containment": 0.2,   # co-occurring boxes this far apart (and not plant-over-pot stacked) are two objects: veto, also through bridging groups
    "colocated_iou": 0.5, "colocated_containment": 0.7, "nest_iou": 0.25, "nest_containment": 0.85, "colocated_min_frac": 0.5,
    "plane_iou": 0.3, "plane_containment": 0.8, "plane_min_iou_contained": 0.2, "plane_min_appearance": 0.3,   # same tracked segment: plane position
    # Different segments (no shared geometry): appearance + time only. Within one segment these rules never run (YT1: 9 wrong merges when they did).
    "near_gap_s": 1.5, "near_min_hist": 0.45,
    "gap_s": 4.0, "min_hist": 0.6, "min_ncc": 0.45, "hist_alone": 0.85,
    "sift_gap_s": 15.0, "sift_min_inliers": 12,                                         # same family; Yugo: same frame 12-55 inliers, different frames 3-8
    "cross_family_gap_s": 4.0, "cross_family_min_inliers": 20,                          # e.g. one yellow vase read as `cup` and `cushion`
    # Same family, nearest frames registered directly: strong overlap only (YT1: two neighbouring cream frames overlap at IoU .36; the weak path is off)
    "register_gap_s": 2.0, "register_iou": 0.45, "register_weak_iou": 1.01, "register_weak_hist": 0.7, "register_weak_ncc": 0.5, "register_distinct": 0.1,
    "register_min_hist": 0.5,                                                           # registration costs ~0.1 s per frame pair: only for look-alikes
    "min_sightings": 2,                                                                 # fewer -> unconfirmed candidate (review), not an item line
    "demoted_min_sightings": True,                                                      # the same for a demoted (ambiguous) spine group
    # Per-category floor on a line's best confidence (one item_conf .25 serves every prompt). YugoTest6, filmed rotated: books lying sideways read as `box`
    # at .31-.43, a blurred light as `lamp` .37-.42; real boxes .53-.77 on all three runs (YT1 included). Other categories showed no separating confidence.
    "min_confidence": {"box": 0.45, "lamp": 0.45},
}
_ITEM_FEATURES = {}   # (run root, observation_id, box) -> descriptors; the live engine re-merges the whole run every few seconds


# Detector labels that the fixed item review taxonomy lacks (plant pot vase frame cup box basket lamp ornament sign record cushion sculpture
# shelf_part book other; scripts/item_sheets.py), mapped to the taxonomy entry that contains them. A label already in the taxonomy is never changed: the
# detector's pot/vase/cup/basket confusions are disagreements, not synonyms (YugoTest11 dev: `vase` lines read as pot 2, vase 1, basket 1). Applied in dedup
# after consolidation and the line filter, so families and confidence floors still see the detector label (kept on the line as `detector_category`).
ITEM_TAXONOMY = {"framed_sign": "sign", "picture_frame": "frame", "potted_plant": "plant", "decorative_letters": "ornament",
                 "chair": "other", "sofa": "other", "table": "other", "curtain": "other", "electronics": "other"}


def item_family(category):
    c = category[8:] if category.startswith("book_or_") else category
    return ITEM_DEDUP_FAMILIES.get(c, c)


def _stacked(a, b):
    """Plant leaves above its pot: horizontally aligned boxes, the lower one reaching lower, touching or overlapping."""
    if a[1] > b[1]:
        a, b = b, a
    xo = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / max(1e-6, min(a[2] - a[0], b[2] - b[0]))
    return xo >= 0.6 and b[3] >= a[3] and b[1] - a[3] <= 0.15 * max(a[3] - a[1], b[3] - b[1])


def _item_features(key, crop):
    """Colour histogram + 24x24 grey thumbnail now; SIFT keypoints only when a rule asks for them (_sift)."""
    import cv2
    import numpy as np
    if key in _ITEM_FEATURES:
        return _ITEM_FEATURES[key]
    if len(_ITEM_FEATURES) > 4000:
        _ITEM_FEATURES.clear()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    g = cv2.cvtColor(cv2.resize(crop, (24, 24), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(float)
    g -= g.mean()
    s = 320 / max(crop.shape[:2])
    small = cv2.resize(crop, None, fx=s, fy=s) if s < 1 else crop.copy()
    _ITEM_FEATURES[key] = f = {"hist": hist, "thumb": g / (np.linalg.norm(g) + 1e-6), "crop": small}
    return f


def _sift(f):
    import cv2
    import numpy as np
    if "desc" not in f:
        kp, desc = cv2.SIFT_create(nfeatures=500).detectAndCompute(cv2.cvtColor(f.pop("crop"), cv2.COLOR_BGR2GRAY), None)
        f.update(pts=np.float32([k.pt for k in kp]) if kp else None, desc=desc)
    return f


def _sift_inliers(a, b):
    import cv2
    a, b = _sift(a), _sift(b)
    if a["desc"] is None or b["desc"] is None or len(a["desc"]) < 8 or len(b["desc"]) < 8:
        return 0
    m = [x[0] for x in cv2.BFMatcher().knnMatch(a["desc"], b["desc"], k=2) if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(m) < 8:
        return 0
    _, mask = cv2.findHomography(a["pts"][[x.queryIdx for x in m]], b["pts"][[x.trainIdx for x in m]], cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def _project_box(box, H):
    import numpy as np
    x0, y0, x1, y1 = box
    q = np.hstack([np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), np.ones((4, 1))]) @ np.asarray(H, float).T
    q = q[:, :2] / q[:, 2:3]
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def consolidate_items_v2(instances, observations, timestamps, load_image=None, transforms=None, segment_of=None, config=None, cache_key="", register=None, trace=None):
    """(groups, links): item instances that are one physical object. Inputs are never modified; categories in ITEM_CONTEXT_ONLY are left alone.

    timestamps {frame_id: s}; load_image(frame_id) -> BGR image (enables the appearance rules); transforms {frame_id: 3x3 frame->plane} with segment_of
    {frame_id: tracked segment} (enables the plane rule); register(frame_a, frame_b) -> 3x3 a->b homography or None (direct registration across a broken chain).
    Rules, strongest first: same-frame co-location (incl. plant-over-pot stacking) -> same plane position in a shared segment -> same position after registering
    their nearest frames directly (the chained plane drifts on fast sweeps) -> only when neither geometry exists: near in time + colour histogram, or SIFT inliers
    between their best crops. Veto: two groups that co-occur in a frame without overlapping (and not stacked), or whose registered boxes are apart, are distinct,
    and no chain of merges may join them."""
    import collections, itertools
    import numpy as np
    cfg = {**ITEM_DEDUP_DEFAULTS, **(config or {})}
    obs = observations if isinstance(observations, dict) else {o["observation_id"]: o for o in observations}
    segment_of = segment_of or {}
    insts = [i for i in instances if i.get("category") not in ITEM_CONTEXT_ONLY and not i.get("non_book_demotion")]
    best = {}   # instance -> its two most confident observations: the crops the appearance rules compare
    if load_image is not None:
        for it in insts:
            os_ = [obs[o] for o in it["observation_ids"] if o in obs and obs[o].get("bbox_xyxy")]
            best[it["instance_id"]] = sorted(os_, key=lambda o: -(o.get("confidence") or 0))[:2]
    feats = collections.defaultdict(list)
    by_frame_needed = collections.defaultdict(list)
    for iid, os_ in best.items():
        for rank, o in enumerate(os_):
            by_frame_needed[o["frame_id"]].append((iid, rank, o))
    for fid in sorted(by_frame_needed):   # each frame is read once, and only when one of its crops is not cached yet
        im = None if all((cache_key, o["observation_id"], tuple(o["bbox_xyxy"])) in _ITEM_FEATURES for _, _, o in by_frame_needed[fid]) else load_image(fid)
        for iid, rank, o in by_frame_needed[fid]:
            key = (cache_key, o["observation_id"], tuple(o["bbox_xyxy"]))
            if key not in _ITEM_FEATURES:
                if im is None:
                    continue
                x0, y0, x1, y1 = (int(round(v)) for v in o["bbox_xyxy"])
                c = im[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
                if not c.size or min(c.shape[:2]) < 8:
                    continue
            else:
                c = None
            feats[iid].append((rank, _item_features(key, c)))
    info = {}
    for it in insts:
        os_ = [obs[o] for o in it["observation_ids"] if o in obs and obs[o].get("bbox_xyxy")]
        if not os_:
            continue
        by_frame, plane = collections.defaultdict(list), collections.defaultdict(list)
        for o in os_:
            by_frame[o["frame_id"]].append(o["bbox_xyxy"])
            if transforms and o["frame_id"] in transforms:
                x0, y0, x1, y1 = o["bbox_xyxy"]
                q = np.hstack([np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), np.ones((4, 1))]) @ np.asarray(transforms[o["frame_id"]], float).T
                q = q[:, :2] / q[:, 2:3]
                plane[segment_of.get(o["frame_id"], 0)].append([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()])
        info[it["instance_id"]] = {"family": item_family(it["category"]), "frames": by_frame, "t": sorted(timestamps.get(o["frame_id"], 0.0) for o in os_),
                                   "plane": {s: [float(v) for v in np.median(np.array(b), axis=0)] for s, b in plane.items()}, "feats": [f for _, f in sorted(feats.get(it["instance_id"], []), key=lambda x: x[0])]}
    ids = sorted(info)
    distinct, candidates = set(), []
    for ia, ib in itertools.combinations(ids, 2):
        A, B = info[ia], info[ib]
        same_family = A["family"] == B["family"]
        potted = same_family and A["family"] == "potted"
        shared = set(A["frames"]) & set(B["frames"])
        stats = {f: [(*_box_stats(x, y), potted and _stacked(x, y)) for x in A["frames"][f] for y in B["frames"][f]] for f in shared}
        if shared and any(max(s[1] for s in v) < cfg["distinct_max_containment"] and not any(s[2] for s in v) for v in stats.values()):
            distinct.add(frozenset((ia, ib)))
            if trace is not None:
                trace.append((ia, ib, "distinct_cooccurring"))
            continue
        geo = set(A["plane"]) & set(B["plane"])
        gap = min(abs(x - y) for x in A["t"] for y in B["t"])
        hist = ncc = None
        if A["feats"] and B["feats"]:
            import cv2
            hist = float(cv2.compareHist(A["feats"][0]["hist"], B["feats"][0]["hist"], cv2.HISTCMP_CORREL))
            ncc = float((A["feats"][0]["thumb"] * B["feats"][0]["thumb"]).sum())
        ev = None
        if same_family and shared:
            good = sum(1 for f in shared if any((s[0] >= cfg["colocated_iou"] and s[1] >= cfg["colocated_containment"]) or
                                               (s[0] >= cfg["nest_iou"] and s[1] >= cfg["nest_containment"]) or s[2] for s in stats[f]))
            if good / len(shared) >= cfg["colocated_min_frac"]:
                ev = (0, {"rule": "same_frame_colocated", "shared_frames": len(shared), "colocated_frames": good})
        if ev is None and same_family and geo and (hist is None or hist >= cfg["plane_min_appearance"]):
            for s in sorted(geo):
                iou, cont = _box_stats(A["plane"][s], B["plane"][s])
                if iou >= cfg["plane_iou"] or (cont >= cfg["plane_containment"] and iou >= cfg["plane_min_iou_contained"]):
                    ev = (1, {"rule": "same_plane_position", "plane_iou": round(iou, 3), "plane_containment": round(cont, 3)})
                    break
        H = None
        if ev is None and same_family and register is not None and gap <= cfg["register_gap_s"] and (hist is None or hist >= cfg["register_min_hist"]):
            fa, fb = min(((x, y) for x in A["frames"] for y in B["frames"] if x != y), key=lambda p: abs(timestamps.get(p[0], 0) - timestamps.get(p[1], 0)), default=(None, None))
            H = register(fa, fb) if fa else None
            if H is not None:
                pa = _project_box(A["frames"][fa][0], H)
                iou, cont = max((_box_stats(pa, b) for b in B["frames"][fb]), key=lambda x: x[0] + x[1])
                if iou >= cfg["register_iou"] or (iou >= cfg["register_weak_iou"] and hist is not None and hist >= cfg["register_weak_hist"] and ncc >= cfg["register_weak_ncc"]):
                    ev = (1, {"rule": "registered_position", "frames": [fa, fb], "iou": round(iou, 3), "containment": round(cont, 3)})
                elif cont < cfg["register_distinct"]:
                    distinct.add(frozenset((ia, ib)))
                    if trace is not None:
                        trace.append((ia, ib, "distinct_registered", round(iou, 3), round(cont, 3)))
                    continue
        if ev is None and not geo and H is None and hist is not None:   # no geometry at all: appearance + time
            if same_family and gap <= cfg["near_gap_s"] and hist >= cfg["near_min_hist"]:
                ev = (2, {"rule": "near_in_time_appearance", "gap_s": round(gap, 2), "hist": round(hist, 3)})
            elif same_family and gap <= cfg["gap_s"] and hist >= cfg["min_hist"] and (ncc >= cfg["min_ncc"] or hist >= cfg["hist_alone"]):
                ev = (2, {"rule": "time_appearance", "gap_s": round(gap, 2), "hist": round(hist, 3), "ncc": round(ncc, 3)})
            elif gap <= (cfg["sift_gap_s"] if same_family else cfg["cross_family_gap_s"]):
                k = max(_sift_inliers(x, y) for x in A["feats"] for y in B["feats"])
                if k >= (cfg["sift_min_inliers"] if same_family else cfg["cross_family_min_inliers"]):
                    ev = (3, {"rule": "sift_inliers" if same_family else "sift_inliers_cross_family", "gap_s": round(gap, 2), "inliers": k})
        if trace is not None:
            trace.append((ia, ib, ev[1]["rule"] if ev else None, round(gap, 2), bool(geo), hist and round(hist, 2), ncc and round(ncc, 2)))
        if ev:
            candidates.append((ev[0], ia, ib, {"instances": [ia, ib], **ev[1]}))
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    members = {i["instance_id"]: {i["instance_id"]} for i in instances}
    where = {i: i for i in members}
    links = []
    for _, ia, ib, ev in candidates:
        ra, rb = where[ia], where[ib]
        if ra == rb:
            links.append(ev)
            continue
        if any(frozenset((x, y)) in distinct for x in members[ra] for y in members[rb]):
            continue
        members[ra] |= members[rb]
        for i in members.pop(rb):
            where[i] = ra
        links.append(ev)
    return sorted(sorted(g) for g in members.values()), links
