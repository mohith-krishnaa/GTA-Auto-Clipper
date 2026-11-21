import json
import os
import re
import shutil
import subprocess
import tempfile

import cv2
from pathlib import Path

import requests
import streamlit as st
from faster_whisper import WhisperModel
from clip_utils import chunk_transcript, crop_x_expression, group_caption_words, normalize_clips
from visual_utils import rank_visual_windows

st.set_page_config(page_title="MK AI Clipper", page_icon="✂️", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem;}
[data-testid="stAppViewContainer"] {background: #08090d;}
[data-testid="stSidebar"] {background: #10121a;}
h1, h2, h3 {letter-spacing: -0.04em;}
.stButton>button {border-radius: 10px; min-height: 2.8rem; font-weight: 650;}
</style>
""", unsafe_allow_html=True)

st.title("✂️ MK AI Clipper")
st.caption("Turn long videos into ranked vertical clips. Local transcription · AI moment detection · face-aware vertical framing")

with st.sidebar:
    st.subheader("AI settings")
    api_key = st.text_input("Gemini API key", type="password", help="Used only for transcript analysis. It is not saved.")
    model_size = st.selectbox("Whisper model", ["tiny", "base", "small"], index=1,
                              help="Tiny/base are better for low-memory computers. First run downloads the model.")
    clip_count = st.slider("Clips to generate", 1, 10, 5)
    detect_visual = st.checkbox("Find gameplay/action moments (experimental)", value=True)
    min_len, max_len = st.slider("Clip duration (seconds)", 15, 90, (25, 55))
    st.caption("Requires FFmpeg installed and available in PATH.")

st.markdown("### 1. Add your source video")
uploaded = st.file_uploader("Upload a video you own or have permission to edit", type=["mp4", "mov", "mkv", "webm", "m4v"])

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def extract_audio(video_path, audio_path):
    p = run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", audio_path])
    if p.returncode:
        raise RuntimeError(p.stderr[-1800:])

def transcribe(audio_path, size):
    model = WhisperModel(size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)
    rows, words = [], []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            rows.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
        for word in (seg.words or []):
            token = word.word.strip()
            if token:
                words.append({"start": float(word.start), "end": float(word.end), "text": token})
    return rows, words, info.language or "unknown"

def ask_gemini(key, transcript, count, low, high):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    prompt = f"""You are an expert short-form video editor. Select up to {count} self-contained, compelling moments from this transcript.
Each clip must be {low}-{high} seconds, start and end at natural sentence boundaries, and make sense without missing context.
Prefer strong hooks, surprising insights, humour, emotion, useful advice or complete stories. Avoid duplicate/overlapping moments and avoid inventing dialogue.
Return ONLY JSON: {{\"clips\":[{{\"start\": number, \"end\": number, \"title\": string, \"hook\": string, \"reason\": string, \"score\": integer}}]}}
Score from 0 to 100. Timestamps are seconds from the start of the source.
TRANSCRIPT:
{transcript[:90000]}"""
    response = requests.post(
        url, headers={"x-goog-api-key": key},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Gemini returned an unexpected response. Try again or reduce the transcript size.") from exc
    return parsed.get("clips", [])

def ass_time(seconds):
    cs = max(0, round(seconds * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02}:{s:02}.{c:02}"

def ass_escape(value):
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")

def make_ass(words, start, end, path):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,76,&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,1,0,1,5,2,2,70,70,310,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for group in group_caption_words(words, start, end, max_words=5, max_gap=0.55):
        karaoke = []
        for word in group:
            token = ass_escape(word["text"])
            duration_cs = max(1, round((word["end"] - word["start"]) * 100))
            karaoke.append("{" + chr(92) + f"kf{duration_cs}" + "}" + token)
        event_start = group[0]["start"] - start
        event_end = max(event_start + 0.12, group[-1]["end"] - start)
        events.append(
            f"Dialogue: 0,{ass_time(event_start)},{ass_time(event_end)},Default,,0,0,0,,{' '.join(karaoke)}"
        )
    Path(path).write_text(header + "\n".join(events), encoding="utf-8")


def scan_visual_activity(video, duration, sample_seconds=1.0):
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video for visual analysis.")
    samples, previous_gray, previous_hist = [], None, None
    timestamp = 0.0
    try:
        while timestamp < duration:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if ok:
                small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                cv2.normalize(hist, hist)
                motion, cut = 0.0, 0.0
                if previous_gray is not None:
                    motion = float(cv2.absdiff(gray, previous_gray).mean()) / 255.0
                    cut = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                samples.append({"timestamp": timestamp, "motion": motion, "cut": cut})
                previous_gray, previous_hist = gray, hist
            timestamp += sample_seconds
    finally:
        capture.release()
    return samples

def detect_face_crop_points(video, start, end, sample_seconds=0.75):
    """Sample the clip for faces and return crop centers in source-pixel coordinates."""
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video for face tracking.")
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if not width or not height:
        capture.release()
        raise RuntimeError("Could not read source video dimensions.")
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    points = []
    timestamp = start
    try:
        while timestamp < end:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
                )
                if len(faces):
                    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                    points.append((timestamp - start, x + w / 2))
            timestamp += sample_seconds
    finally:
        capture.release()
    return points, int(width), int(height)


def render_clip(video, words, clip, out_path, temp_dir):
    start, end = float(clip["start"]), float(clip["end"])
    ass = os.path.join(temp_dir, "captions.ass")
    make_ass(words, start, end, ass)
    points, source_width, source_height = detect_face_crop_points(video, start, end)
    scaled_width = source_width * 1920 / source_height
    crop_width = min(1080, scaled_width)
    scaled_points = [(t, center * 1920 / source_height) for t, center in points]
    crop_x = crop_x_expression(scaled_points, 1080, scaled_width)
    # Track the largest detected face; fall back to a centered crop when none is found.
    escaped = ass.replace("\\", "/").replace(":", "\\:")
    vf = ("split=2[bg][fg];"
          "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=22:10[bg];"
          f"[fg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:x='{crop_x}':y=(ih-1920)/2[fg];"
          f"[bg][fg]overlay=(W-w)/2:(H-h)/2,subtitles='{escaped}'")
    p = run(["ffmpeg", "-y", "-ss", str(start), "-i", video, "-t", str(end-start),
             "-filter_complex", vf, "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "21", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out_path])
    if p.returncode:
        raise RuntimeError(p.stderr[-2200:])

if uploaded:
    st.video(uploaded)
    if st.button("Analyze video and find clips", type="primary", use_container_width=True):
        if not api_key and not detect_visual:
            st.error("Enter a Gemini API key or enable visual activity detection.")
        elif not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            st.error("FFmpeg and FFprobe must both be installed and available in PATH.")
        else:
            previous_job = st.session_state.pop("clip_job", None)
            st.session_state.pop("rendered", None)
            if previous_job:
                shutil.rmtree(previous_job.get("workdir", ""), ignore_errors=True)
            with st.spinner("Transcribing locally. The first run may download the Whisper model..."):
                workdir = tempfile.mkdtemp(prefix="mkclip_")
                safe_name = Path(uploaded.name).name
                source = os.path.join(workdir, safe_name)
                Path(source).write_bytes(uploaded.getbuffer())
                audio = os.path.join(workdir, "audio.wav")
                try:
                    extract_audio(source, audio)
                    segments, words, language = transcribe(audio, model_size)
                    transcript_chunks = chunk_transcript(segments)
                    st.session_state["clip_job"] = {
                        "workdir": workdir, "source": source, "words": words,
                        "language": language, "transcript_chunks": transcript_chunks,
                    }
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")
                    shutil.rmtree(workdir, ignore_errors=True)
            job = st.session_state.get("clip_job")
            if job and job.get("source") == source:
                with st.spinner("AI is ranking the strongest moments..."):
                    try:
                        probe = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "default=noprint_wrappers=1:nokey=1", job["source"]])
                        if probe.returncode:
                            raise RuntimeError("FFprobe could not read the uploaded video: " + probe.stderr[-800:])
                        duration = float(probe.stdout.strip())
                        candidates = []
                        failures = []
                        total_chunks = len(job["transcript_chunks"])
                        if total_chunks and api_key:
                            progress = st.progress(0, text="Analyzing transcript chunks...")
                            for index, transcript_chunk in enumerate(job["transcript_chunks"], 1):
                                try:
                                    candidates.extend(ask_gemini(api_key, transcript_chunk, clip_count, min_len, max_len))
                                except Exception as exc:
                                    failures.append(f"Chunk {index}: {exc}")
                                progress.progress(index / total_chunks, text=f"Analyzed chunk {index} of {total_chunks}")
                        elif total_chunks and not api_key:
                            st.warning("Speech-based AI ranking skipped because no Gemini key was supplied.")
                        if detect_visual:
                            with st.spinner("Scanning sampled frames for motion and scene changes..."):
                                visual_samples = scan_visual_activity(job["source"], duration)
                            candidates.extend(rank_visual_windows(visual_samples, duration, clip_count, min_len, max_len))
                        clean = normalize_clips(candidates, duration, clip_count, min_len, max_len)
                        if not clean and failures and not detect_visual:
                            raise RuntimeError("Gemini analysis failed for every transcript chunk. " + failures[0])
                        if failures:
                            st.warning(f"{len(failures)} transcript chunk(s) could not be analyzed; results may be incomplete.")
                        if not clean:
                            st.warning("No useful clips were detected. This can happen with static footage or low motion.")
                        st.session_state["clip_job"]["clips"] = clean
                        st.session_state["clip_job"]["duration"] = duration
                        st.session_state["clip_job"]["transcript_chunks"] = []
                    except Exception as exc:
                        st.error(f"AI analysis failed: {exc}")

job = st.session_state.get("clip_job")
if job and job.get("clips"):
    st.markdown("### 2. Review AI-selected moments")
    st.caption(f"Detected language: {job['language']} · {len(job['clips'])} candidate clips")
    chosen = []
    for i, clip in enumerate(job["clips"], 1):
        with st.container(border=True):
            left, right = st.columns([5, 1])
            with left:
                st.markdown(f"**{i}. {clip.get('title', 'Untitled moment')}**")
                st.write(f"{clip['start']:.1f}s – {clip['end']:.1f}s · Score {clip.get('score', 0)}/100")
                st.caption(clip.get("reason", ""))
                st.caption("Hook: " + clip.get("hook", ""))
            with right:
                if st.checkbox("Render", value=True, key=f"pick_{i}"):
                    chosen.append(clip)
    if st.button("Render selected clips", type="primary"):
        rendered = []
        with st.spinner("Rendering vertical clips with captions..."):
            for i, clip in enumerate(chosen, 1):
                out = os.path.join(job["workdir"], f"clip_{i:02}.mp4")
                try:
                    render_clip(job["source"], job["words"], clip, out, job["workdir"])
                    rendered.append((clip, out))
                except Exception as exc:
                    st.error(f"Clip {i} failed: {exc}")
        st.session_state["rendered"] = rendered

if st.session_state.get("rendered"):
    st.markdown("### 3. Download your clips")
    for clip, path in st.session_state["rendered"]:
        st.markdown(f"**{clip.get('title', 'Clip')}**")
        st.video(path)
        with open(path, "rb") as handle:
            st.download_button("Download MP4", handle.read(), file_name=Path(path).name,
                               mime="video/mp4", key=f"download_{Path(path).name}")
    st.info("Review every clip before publishing. Only use footage you own or are authorized to repurpose.")
