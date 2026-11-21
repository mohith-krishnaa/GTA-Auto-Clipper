"""Validation helpers for AI-selected clip candidates."""
import math


def normalize_clips(candidates, duration, count, min_length, max_length):
    """Validate, rank and de-overlap untrusted model-generated clip candidates."""
    if not isinstance(candidates, list):
        raise ValueError("Gemini response must contain a clips array.")

    valid = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
            score = float(item.get("score", 0))
        except (KeyError, TypeError, ValueError):
            continue

        if not all(math.isfinite(v) for v in (start, end, score)):
            continue
        if start < 0 or end <= start or end > duration:
            continue
        if not min_length <= end - start <= max_length:
            continue

        valid.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "title": str(item.get("title") or "Untitled moment")[:120],
            "hook": str(item.get("hook") or "")[:300],
            "reason": str(item.get("reason") or "")[:500],
            "score": max(0, min(100, round(score))),
        })

    valid.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    for candidate in valid:
        overlaps = any(
            candidate["start"] < existing["end"]
            and existing["start"] < candidate["end"]
            for existing in selected
        )
        if not overlaps:
            selected.append(candidate)
        if len(selected) >= count:
            break

    return sorted(selected, key=lambda item: item["start"])
