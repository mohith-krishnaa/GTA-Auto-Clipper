"""Validation, transcript chunking and FFmpeg framing helpers."""
import math


def chunk_transcript(segments, max_chars=18000):
    """Group timestamped Whisper segments without dropping later parts of a video."""
    if max_chars < 1000:
        raise ValueError("max_chars must be at least 1000.")
    chunks, current, current_size = [], [], 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        try:
            timestamp = float(segment["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(timestamp) or timestamp < 0:
            continue
        line = f"[{timestamp:.1f}s] {text}"
        while len(line) > max_chars:
            if current:
                chunks.append("\n".join(current))
                current, current_size = [], 0
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        if not line:
            continue
        extra = len(line) + (1 if current else 0)
        if current and current_size + extra > max_chars:
            chunks.append("\n".join(current))
            current, current_size, extra = [], 0, len(line)
        current.append(line)
        current_size += extra
    if current:
        chunks.append("\n".join(current))
    return chunks


def normalize_clips(candidates, duration, count, min_length, max_length):
    """Validate, rank and de-overlap untrusted model-generated clip candidates."""
    if not isinstance(candidates, list):
        raise ValueError("Gemini response must contain a clips array.")
    valid = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        try:
            start, end = float(item["start"]), float(item["end"])
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
            "start": round(start, 2), "end": round(end, 2),
            "title": str(item.get("title") or "Untitled moment")[:120],
            "hook": str(item.get("hook") or "")[:300],
            "reason": str(item.get("reason") or "")[:500],
            "score": max(0, min(100, round(score))),
        })
    valid.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    for candidate in valid:
        if not any(candidate["start"] < old["end"] and old["start"] < candidate["end"] for old in selected):
            selected.append(candidate)
        if len(selected) >= count:
            break
    return sorted(selected, key=lambda item: item["start"])


def crop_x_expression(points, crop_width, source_width):
    """Build a time-varying FFmpeg crop-x expression from (seconds, center_x) points."""
    max_x = max(0.0, float(source_width) - float(crop_width))
    if max_x == 0:
        return "0"
    clean = []
    for timestamp, center in points:
        timestamp, center = float(timestamp), float(center)
        if math.isfinite(timestamp) and math.isfinite(center) and timestamp >= 0:
            clean.append((timestamp, min(max(center - crop_width / 2, 0), max_x)))
    clean.sort()
    if not clean:
        clean = [(0.0, max_x / 2)]
    deduped = []
    for point in clean:
        if deduped and point[0] == deduped[-1][0]:
            deduped[-1] = point
        else:
            deduped.append(point)
    if len(deduped) == 1:
        return f"{deduped[0][1]:.2f}"
    expression = f"{deduped[-1][1]:.2f}"
    for (t0, x0), (t1, x1) in reversed(list(zip(deduped, deduped[1:]))):
        span = max(t1 - t0, 0.001)
        interpolation = f"({x0:.2f}+({x1-x0:.2f})*(t-{t0:.3f})/{span:.3f})"
        expression = f"if(lt(t\\,{t1:.3f})\\,{interpolation}\\,{expression})"
    return expression
