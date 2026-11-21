"""Heuristic visual-activity scoring for speech-independent gameplay clips."""
import math


def rank_visual_windows(samples, duration, count, min_length, max_length, step=5.0):
    """Rank fixed-length windows using sampled motion and scene-change signals.

    samples are dictionaries with timestamp, motion (0..1), and cut (0..1).
    This is a candidate generator, not semantic understanding of gameplay.
    """
    values = []
    for sample in samples:
        try:
            timestamp = float(sample["timestamp"])
            motion = float(sample.get("motion", 0))
            cut = float(sample.get("cut", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in (timestamp, motion, cut)):
            if 0 <= timestamp <= duration:
                values.append((timestamp, max(0.0, motion), max(0.0, cut)))
    values.sort()
    if not values or duration < min_length or count < 1 or step <= 0:
        return []

    length = min(max_length, max(min_length, (min_length + max_length) / 2))
    last_start = max(0.0, duration - length)
    starts = []
    position = 0.0
    while position < last_start:
        starts.append(position)
        position += step
    starts.append(last_start)

    scored = []
    for start in starts:
        end = min(duration, start + length)
        window = [(motion, cut) for timestamp, motion, cut in values if start <= timestamp < end]
        if not window:
            continue
        motions = sorted(item[0] for item in window)
        # The upper quartile rewards sustained activity without letting one spike dominate.
        upper = motions[max(0, math.ceil(len(motions) * 0.75) - 1)]
        average_cut = sum(item[1] for item in window) / len(window)
        raw = 0.75 * upper + 0.25 * min(1.0, average_cut * 2.0)
        if raw > 0.015:
            scored.append((raw, start, end))

    if not scored:
        return []
    peak = max(item[0] for item in scored)
    ranked = sorted(scored, reverse=True)
    selected = []
    for raw, start, end in ranked:
        if any(start < old["end"] and old["start"] < end for old in selected):
            continue
        score = max(1, min(100, round(raw / peak * 100)))
        selected.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "title": "High visual activity",
            "hook": "Gameplay/action segment detected from frame changes",
            "reason": "Heuristic motion and scene-change score; review manually.",
            "score": score,
            "source": "visual",
        })
        if len(selected) >= count:
            break
    return sorted(selected, key=lambda item: item["start"])
