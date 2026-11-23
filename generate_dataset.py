#!/usr/bin/env python3
"""3D-SpatialViewBench dataset generator.

Generates 60 deterministic multiple-choice questions probing 3D mental rotation
and perspective-taking in text-only LLMs. Answers are computed with exact
integer rotation matrices (90-degree increments only), so every ground truth is
a whole number with no floating-point ambiguity.

Coordinate system (right-handed):
    +X = right, +Y = up, -Z = forward (camera initially faces -Z)

Usage:
    py generate_dataset.py          # writes dataset.json next to this script
    py generate_dataset.py out.json # writes to an explicit path
"""

import json
import random
import sys
from pathlib import Path

SEED = 20261122
N_PER_LEVEL = 20

DIRECTIONS = ("left", "right", "front", "behind", "above", "below")

# Relative-position vectors in the camera frame: +X right, +Y up, -Z forward.
VECTORS = {
    "right": (1, 0, 0), "left": (-1, 0, 0),
    "up": (0, 1, 0), "down": (0, -1, 0),
    "forward": (0, 0, -1), "back": (0, 0, 1),
    "forward-right": (1, 0, -1), "forward-left": (-1, 0, -1),
    "back-right": (1, 0, 1), "back-left": (-1, 0, 1),
    "up-right": (1, 1, 0), "up-left": (-1, 1, 0),
    "down-right": (1, -1, 0), "down-left": (-1, -1, 0),
    "forward-up-right": (1, 1, -1), "forward-up-left": (-1, 1, -1),
    "forward-down-right": (1, -1, -1), "forward-down-left": (-1, -1, -1),
    "back-up-right": (1, 1, 1), "back-up-left": (-1, 1, 1),
    "back-down-right": (1, -1, 1), "back-down-left": (-1, -1, 1),
    "forward-up": (0, 1, -1), "forward-down": (0, -1, -1),
    "back-up": (0, 1, 1), "back-down": (0, -1, 1),
    "right-up": (1, 1, 0), "right-down": (1, -1, 0),
    "left-up": (-1, 1, 0), "left-down": (-1, -1, 0),
}


# --------------------------------------------------------------------------
# Exact rotation matrices (integers, 90-degree increments)
# --------------------------------------------------------------------------
def rot_y(deg):
    """Yaw about +Y. Positive = counterclockwise seen from above."""
    deg %= 360
    if deg == 0:
        return ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    if deg == 90:
        return ((0, 0, 1), (0, 1, 0), (-1, 0, 0))
    if deg == 180:
        return ((-1, 0, 0), (0, 1, 0), (0, 0, -1))
    if deg == 270:
        return ((0, 0, -1), (0, 1, 0), (1, 0, 0))
    raise ValueError(f"rot_y requires a multiple of 90, got {deg}")


def rot_x(deg):
    """Pitch about +X. Positive = nose up."""
    deg %= 360
    if deg == 0:
        return ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    if deg == 90:
        return ((1, 0, 0), (0, 0, -1), (0, 1, 0))
    if deg == 180:
        return ((1, 0, 0), (0, -1, 0), (0, 0, -1))
    if deg == 270:
        return ((1, 0, 0), (0, 0, 1), (0, -1, 0))
    raise ValueError(f"rot_x requires a multiple of 90, got {deg}")


def matmul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def apply_mat(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def transpose(m):
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


def rotate(m, v):
    """Rotate a point/vector in world coordinates: p' = M p."""
    return apply_mat(m, v)


def to_cam(m, p):
    """Express world point p in the camera frame whose basis is M (columns)."""
    return apply_mat(transpose(m), p)


def to_world(m, c):
    return apply_mat(m, c)


def fmt_vec(v):
    return f"({v[0]}, {v[1]}, {v[2]})"


# --------------------------------------------------------------------------
# Question helpers
# --------------------------------------------------------------------------
def direction_word(v):
    """Describe a camera-frame vector using world-style axis words."""
    z = 0 if v[2] == 0 else (1 if v[2] > 0 else -1)
    x = 0 if v[0] == 0 else (1 if v[0] > 0 else -1)
    y = 0 if v[1] == 0 else (1 if v[1] > 0 else -1)
    parts = []
    if z == -1:
        parts.append("forward")
    elif z == 1:
        parts.append("back")
    if x == 1:
        parts.append("right")
    elif x == -1:
        parts.append("left")
    if y == 1:
        parts.append("up")
    elif y == -1:
        parts.append("down")
    if not parts:
        return "centered"
    return "-".join(parts)  # ordered forward/back, right/left, up/down


def make_options(correct_word, pool, rng):
    """Build 4 unique options, shuffle, return (options dict, correct letter)."""
    distractors = [w for w in pool if w != correct_word]
    rng.shuffle(distractors)
    chosen = distractors[:3]
    opts = [correct_word] + chosen
    rng.shuffle(opts)
    letters = "ABCD"
    options = {letters[i]: opts[i] for i in range(4)}
    correct_letter = letters[opts.index(correct_word)]
    return options, correct_letter


ROTATION_DESC = {
    90: "90° clockwise",
    180: "180°",
    270: "270° clockwise (= 90° counter-clockwise)",
}


# --------------------------------------------------------------------------
# Level 1: single-axis yaw rotation, 3 objects
# --------------------------------------------------------------------------
def level1(rng, idx, scene):
    objects = scene["objects"]
    yaw_deg = scene["yaw"]
    question_form = scene["form"]

    m = rot_y(yaw_deg)
    new_cam = m  # camera basis after yaw rotation (was identity)

    if question_form == "view":
        # Which object is now in the named relative position?
        target_word = scene["target_word"]
        pool_words = list(VECTORS.keys())
        correct_letter = None
        for obj in objects:
            if direction_word(to_cam(new_cam, obj["pos"])) == target_word:
                correct_letter = obj["label"]
                break
        # Build options from labels
        if correct_letter is None:
            return None
        others = [o["label"] for o in objects if o["label"] != correct_letter]
        letters = "ABCD"
        pool = [correct_letter] + others
        # need 4 options; add decoy labels not in scene
        decoys = ["Object E", "Object F", "Object G"]
        while len(pool) < 4:
            pool.append(decoys.pop(0))
        rng.shuffle(pool)
        options = {letters[i]: pool[i] for i in range(4)}
        correct_answer = letters[pool.index(correct_letter)]
        question = (
            f"The camera at the origin then performs a yaw turn of "
            f"{ROTATION_DESC[yaw_deg]} about the vertical axis. "
            f"From the camera's new viewpoint, which object lies {target_word}?"
        )
        gt = {
            "initial_positions": {o["label"]: list(o["pos"]) for o in objects},
            "camera_rotation_y_deg": yaw_deg,
            "camera_basis_after": [list(r) for r in new_cam],
            "positions_in_new_camera_frame": {
                o["label"]: list(to_cam(new_cam, o["pos"])) for o in objects
            },
            "target_relative_vector": list(VECTORS[target_word]),
            "answer_label": correct_letter,
        }
        return {
            "id": f"L1-{idx:03d}",
            "difficulty": "Level 1",
            "scene_description": scene["desc"],
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "ground_truth_math": gt,
        }

    # question_form == "relative": what is the relative position of X from Y,
    # after the camera rotates.
    a, b = scene["pair"]
    pa = objects[a]["pos"]
    pb = objects[b]["pos"]
    rel_world = tuple(pa[i] - pb[i] for i in range(3))
    rel_cam = to_cam(new_cam, rel_world)
    word = direction_word(rel_cam)
    options, correct_answer = make_options(word, list(VECTORS.keys()), rng)
    question = (
        f"The camera at the origin performs a yaw turn of "
        f"{ROTATION_DESC[yaw_deg]} about the vertical axis. "
        f"From the camera's new viewpoint, in which direction is "
        f"{objects[a]['label']} relative to {objects[b]['label']}?"
    )
    gt = {
        "initial_positions": {o["label"]: list(o["pos"]) for o in objects},
        "camera_rotation_y_deg": yaw_deg,
        "camera_basis_after": [list(r) for r in new_cam],
        "world_relative_vector": list(rel_world),
        "camera_frame_relative_vector": list(rel_cam),
        "description": (
            "v_cam = R_y(-yaw)^T * (p_a - p_b); "
            "x=right, y=up, z=back (forward is -z)"
        ),
    }
    return {
        "id": f"L1-{idx:03d}",
        "difficulty": "Level 1",
        "scene_description": scene["desc"],
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "ground_truth_math": gt,
    }


# --------------------------------------------------------------------------
# Level 2: translation + rotation, 4 objects
# --------------------------------------------------------------------------
def level2(rng, idx, scene):
    objects = scene["objects"]
    shift = scene["camera_shift"]
    yaw_deg = scene["yaw"]

    # Camera moves (translation in world), then yaws. New camera basis = R_y.
    m = rot_y(yaw_deg)
    # Camera position is shift, facing direction is m * (0,0,-1) before yaw,
    # so new basis is m (applied to initial identity basis).
    shift_vec = shift
    rel_cam = {}
    cam_in_cam = to_cam(m, shift_vec)
    for o in objects:
        p = to_cam(m, o["pos"])
        rel_cam[o["label"]] = tuple(p[i] - cam_in_cam[i] for i in range(3))
    target_word = scene["target_word"]
    matches = [lbl for lbl, v in rel_cam.items() if direction_word(v) == target_word]
    if not matches:
        return None
    correct_label = matches[0]
    others = [o["label"] for o in objects if o["label"] != correct_label]
    pool = [correct_label] + others
    decoys = ["Object E", "Object F", "Object G"]
    while len(pool) < 4:
        pool.append(decoys.pop(0))
    rng.shuffle(pool)
    letters = "ABCD"
    options = {letters[i]: pool[i] for i in range(4)}
    correct_answer = letters[pool.index(correct_label)]

    question = (
        f"The camera starts at the origin facing forward. It first translates by "
        f"{fmt_vec(shift_vec)} (world axes: +X right, +Y up, -Z forward), then "
        f"yaws {ROTATION_DESC[yaw_deg]} about the vertical axis. "
        f"From the camera's new position and orientation, which object is "
        f"{target_word}?"
    )
    gt = {
        "initial_positions": {o["label"]: list(o["pos"]) for o in objects},
        "camera_translation_world": list(shift_vec),
        "camera_yaw_deg": yaw_deg,
        "camera_basis_after": [list(r) for r in m],
        "positions_in_new_camera_frame": {k: list(v) for k, v in rel_cam.items()},
        "target_relative_vector": list(VECTORS[target_word]),
        "answer_label": correct_label,
    }
    return {
        "id": f"L2-{idx:03d}",
        "difficulty": "Level 2",
        "scene_description": scene["desc"],
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "ground_truth_math": gt,
    }


# --------------------------------------------------------------------------
# Level 3: sequential yaw + pitch, then relative position from new viewpoint
# --------------------------------------------------------------------------
def level3(rng, idx, scene):
    objects = scene["objects"]
    yaw_deg = scene["yaw"]
    pitch_deg = scene["pitch"]
    a, b = scene["pair"]

    ryaw = rot_y(yaw_deg)
    rpitch = rot_x(pitch_deg)
    # Camera first yaws, then pitches in its new frame: basis = R_yaw * R_pitch.
    m = matmul(ryaw, rpitch)

    pa = objects[a]["pos"]
    pb = objects[b]["pos"]
    rel_world = tuple(pa[i] - pb[i] for i in range(3))
    rel_cam = to_cam(m, rel_world)
    word = direction_word(rel_cam)
    # Pool: direction words plus axis combos; filter to distinct
    pool = list(VECTORS.keys())
    options, correct_answer = make_options(word, pool, rng)

    question = (
        f"The camera starts at the origin facing forward. It yaws "
        f"{ROTATION_DESC[yaw_deg]} about the vertical axis, then pitches "
        f"{pitch_deg}° about its local right axis (positive pitch = nose up). "
        f"From the camera's final viewpoint, in which direction is "
        f"{objects[a]['label']} relative to {objects[b]['label']}?"
    )
    gt = {
        "initial_positions": {o["label"]: list(o["pos"]) for o in objects},
        "camera_yaw_deg": yaw_deg,
        "camera_pitch_deg": pitch_deg,
        "rotation_order": "R_final = R_y(yaw) @ R_x(pitch)",
        "camera_basis_after": [list(r) for r in m],
        "world_relative_vector": list(rel_world),
        "camera_frame_relative_vector": list(rel_cam),
        "naming_convention": (
            "camera-frame axes: x=right, y=up, z=back; "
            "forward is -z; combined directions use '-'"
        ),
    }
    return {
        "id": f"L3-{idx:03d}",
        "difficulty": "Level 3",
        "scene_description": scene["desc"],
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "ground_truth_math": gt,
    }


# --------------------------------------------------------------------------
# Scene sampling
# --------------------------------------------------------------------------
LABELS = ["Object A", "Object B", "Object C", "Object D"]
LABEL_NOUNS = [
    "a red cube", "a blue sphere", "a green pyramid", "a yellow cylinder",
    "an orange cone", "a purple torus",
]


def rand_pos(rng, used, max_r=6):
    while True:
        p = (
            rng.randint(-max_r, max_r),
            rng.randint(0, 4),
            rng.randint(-max_r, max_r),
        )
        if p == (0, 0, 0):
            continue
        if p in used:
            continue
        return p


def sample_scene(rng, n_objects, need_obstructions=True):
    """Sample objects with integer coordinates, ensuring at least one
    unambiguous single-axis relative relationship for legibility."""
    used = set()
    objects = []
    nouns = rng.sample(LABEL_NOUNS, n_objects)
    for i in range(n_objects):
        pos = None
        if need_obstructions and i >= 1:
            # Force one axis to be shared with a previous object half the time
            # so a clean single-axis relative relation exists.
            prev = objects[rng.randrange(len(objects))]["pos"]
            if rng.random() < 0.6:
                axis = rng.choice([0, 1, 2])
                pos = []
                for ax in range(3):
                    if ax == axis:
                        delta = rng.choice([-1, 1]) * rng.randint(2, 5)
                        pos.append(prev[ax] + delta)
                    else:
                        pos.append(prev[ax])
                pos = tuple(pos)
                if pos == (0, 0, 0) or pos in used:
                    pos = None
        if pos is None:
            pos = rand_pos(rng, used)
        used.add(pos)
        objects.append({"label": LABELS[i], "pos": pos, "noun": nouns[i]})
    return objects


DIRECTION_POOL_WORDS = [
    "right", "left", "forward", "back", "up", "down",
    "front-right", "front-left", "back-right", "back-left",
    "up-right", "up-left", "down-right", "down-left",
]


def scene_description(objects):
    parts = [f"{o['label']} is {o['noun']} at world coordinates {fmt_vec(o['pos'])}"
             for o in objects]
    return "; ".join(parts) + ". World axes: +X right, +Y up, -Z forward."


# --------------------------------------------------------------------------
# Main generation
# --------------------------------------------------------------------------
def generate():
    rng = random.Random(SEED)
    items = []

    # Level 1: 20 items, single-axis yaw, 3 objects
    l1_count = 0
    attempts = 0
    while l1_count < N_PER_LEVEL and attempts < 5000:
        attempts += 1
        objects = sample_scene(rng, 3)
        yaw_deg = rng.choice([90, 180, 270])
        m = rot_y(yaw_deg)
        form = "view" if l1_count % 2 == 0 else "relative"
        if form == "view":
            # choose an object and a target direction it will occupy
            obj = objects[rng.randrange(3)]
            new_cam = m
            c = to_cam(new_cam, obj["pos"])
            word = direction_word(c)
            # Only accept simple axis-aligned single-word directions for L1 View
            if word not in ("right", "left", "forward", "back", "up", "down"):
                continue
            # Deduplicate: another object at same direction => ambiguous
            if sum(1 for o in objects if direction_word(to_cam(m, o["pos"])) == word) > 1:
                continue
            target_word = word
            scene = {
                "objects": objects,
                "yaw": yaw_deg,
                "form": "view",
                "target_word": target_word,
                "desc": scene_description(objects),
            }
            item = level1(rng, l1_count + 1, scene)
            if item is None:
                continue
            items.append(item)
            l1_count += 1
        else:
            a, b = rng.sample(range(3), 2)
            rel_world = tuple(objects[a]["pos"][i] - objects[b]["pos"][i] for i in range(3))
            rel_cam = to_cam(m, rel_world)
            word = direction_word(rel_cam)
            if word not in ("right", "left", "forward", "back", "up", "down",
                            "front-right", "front-left", "back-right", "back-left",
                            "up-right", "up-left", "down-right", "down-left"):
                continue
            scene = {
                "objects": objects,
                "yaw": yaw_deg,
                "form": "relative",
                "pair": (a, b),
                "desc": scene_description(objects),
            }
            items.append(level1(rng, l1_count + 1, scene))
            l1_count += 1

    # Level 2: 20 items, translation + rotation, 4 objects
    l2_count = 0
    attempts = 0
    while l2_count < N_PER_LEVEL and attempts < 5000:
        attempts += 1
        objects = sample_scene(rng, 4)
        yaw_deg = rng.choice([90, 180, 270])
        shift = (rng.randint(-3, 3), rng.randint(0, 2), rng.randint(-3, 3))
        m = rot_y(yaw_deg)
        rel = {o["label"]: tuple(to_cam(m, o["pos"])[i] - to_cam(m, shift)[i] for i in range(3)) for o in objects}
        # pick a target object with a clean single-axis direction
        cands = []
        for o in objects:
            w = direction_word(rel[o["label"]])
            if w in ("right", "left", "forward", "back", "above", "below"):
                if sum(1 for v in rel.values() if direction_word(v) == w) == 1:
                    cands.append((o["label"], w))
        if not cands:
            continue
        label, word = rng.choice(cands)
        scene = {
            "objects": objects,
            "camera_shift": shift,
            "yaw": yaw_deg,
            "target_word": word,
            "desc": scene_description(objects),
        }
        item = level2(rng, l2_count + 1, scene)
        if item is None:
            continue
        items.append(item)
        l2_count += 1

    # Level 3: 20 items, sequential yaw + pitch, 4 objects
    l3_count = 0
    attempts = 0
    while l3_count < N_PER_LEVEL and attempts < 5000:
        attempts += 1
        objects = sample_scene(rng, 4)
        yaw_deg = rng.choice([90, 180, 270])
        pitch_deg = rng.choice([90, 180, 270])
        m = matmul(rot_y(yaw_deg), rot_x(pitch_deg))
        a, b = rng.sample(range(4), 2)
        rel_world = tuple(objects[a]["pos"][i] - objects[b]["pos"][i] for i in range(3))
        rel_cam = to_cam(m, rel_world)
        word = direction_word(rel_cam)
        if word == "at the same height and depth":
            continue
        scene = {
            "objects": objects,
            "yaw": yaw_deg,
            "pitch": pitch_deg,
            "pair": (a, b),
            "desc": scene_description(objects),
        }
        items.append(level3(rng, l3_count + 1, scene))
        l3_count += 1

    return items


def self_check(items):
    """Re-derive each answer from ground truth and verify consistency."""
    for it in items:
        opts = it["options"]
        assert it["correct_answer"] in opts, it["id"]
        assert len(opts) == 4 and len(set(opts.values())) == 4, it["id"]
        gt = it["ground_truth_math"]
        if "camera_frame_relative_vector" in gt:
            v = gt["camera_frame_relative_vector"]
            assert direction_word(v) == opts[it["correct_answer"]], (
                it["id"], direction_word(v), opts[it["correct_answer"]]
            )
        if "target_relative_vector" in gt and "positions_in_new_camera_frame" in gt:
            v = gt["target_relative_vector"]
            label = gt["answer_label"]
            assert direction_word(gt["positions_in_new_camera_frame"][label]) == direction_word(v), it["id"]
    # Answer key balance
    dist = {}
    for it in items:
        dist[it["correct_answer"]] = dist.get(it["correct_answer"], 0) + 1
    return dist


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("dataset.json")
    items = generate()
    levels = {}
    for it in items:
        levels[it["difficulty"]] = levels.get(it["difficulty"], 0) + 1
    assert levels == {"Level 1": N_PER_LEVEL, "Level 2": N_PER_LEVEL, "Level 3": N_PER_LEVEL}, levels
    assert len(items) == 60, len(items)
    for it in items:
        for k in ("id", "difficulty", "scene_description", "question",
                  "options", "correct_answer", "ground_truth_math"):
            assert k in it, (it["id"], k)
    dist = self_check(items)
    out_path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(items)} items to {out_path}")
    print(f"Levels: {levels}")
    print(f"Answer key distribution: {dist}")


if __name__ == "__main__":
    main()
