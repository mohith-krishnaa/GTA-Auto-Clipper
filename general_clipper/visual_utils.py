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


# GTA HUD text cues: high-confidence end-state / objective overlays, not generic dialogue.
GTA_HUD_CUES = {
    "WASTED": 1.0,
    "BUSTED": 1.0,
    "MISSION PASSED": 1.0,
    "MISSION FAILED": 0.9,
    "RACE WINNER": 1.0,
    "NEW RECORD": 0.9,
    "CHECKPOINT": 0.45,
    "LAP": 0.25,
}


def detect_gta_hud_cues(ocr_samples):
    """Return timestamped GTA HUD cue hits from OCR text samples."""
    hits = []
    for sample in ocr_samples:
        try:
            timestamp = float(sample["timestamp"])
            text = " ".join(str(sample.get("text", "")).upper().split())
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(timestamp) or timestamp < 0 or not text:
            continue
        matched = [(cue, weight) for cue, weight in GTA_HUD_CUES.items() if cue in text]
        if matched:
            cue, weight = max(matched, key=lambda item: item[1])
            hits.append({"timestamp": timestamp, "cue": cue, "weight": weight, "text": text[:240]})
    return hits


def rank_gta_event_windows(hits, duration, count, clip_length, lead_in=6.0, merge_gap=8.0):
    """Build candidate clips around OCR-recognized GTA HUD events."""
    clean = []
    for hit in hits:
        try:
            timestamp = float(hit["timestamp"])
            weight = float(hit.get("weight", 0.5))
            cue = str(hit["cue"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(timestamp) and math.isfinite(weight) and 0 <= timestamp <= duration:
            clean.append((timestamp, max(0.0, min(1.0, weight)), cue))
    clean.sort()
    if not clean or duration < clip_length or count < 1 or clip_length <= 0:
        return []
    clusters = []
    for timestamp, weight, cue in clean:
        if clusters and timestamp - clusters[-1]["last"] <= merge_gap:
            clusters[-1]["last"] = timestamp
            clusters[-1]["score"] = max(clusters[-1]["score"], weight)
            if cue not in clusters[-1]["cues"]:
                clusters[-1]["cues"].append(cue)
        else:
            clusters.append({"first": timestamp, "last": timestamp, "score": weight, "cues": [cue]})
    ranked = sorted(clusters, key=lambda item: (item["score"], item["last"] - item["first"]), reverse=True)
    selected = []
    for cluster in ranked:
        anchor = cluster["first"]
        start = max(0.0, min(duration - clip_length, anchor - lead_in))
        end = start + clip_length
        if any(start < old["end"] and old["start"] < end for old in selected):
            continue
        cue_text = ", ".join(cluster["cues"])
        selected.append({
            "start": round(start, 2), "end": round(end, 2),
            "title": f"GTA event: {cluster['cues'][0]}",
            "hook": f"HUD cue detected: {cue_text}",
            "reason": "OCR matched a GTA HUD event cue. Verify the overlay and surrounding gameplay.",
            "score": round(cluster["score"] * 100),
            "source": "gta-hud-ocr",
        })
        if len(selected) >= count:
            break
    return sorted(selected, key=lambda item: item["start"])
