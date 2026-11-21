# MK AI Clipper

A general-purpose, local-first AI clipping MVP for podcasts, interviews, gaming videos, tutorials and other speech-led content.

## Features

- Upload a local video (MP4, MOV, MKV, WebM or M4V).
- Transcribe speech locally with faster-whisper (CPU int8), then analyze the complete transcript in bounded chunks.
- Ask Gemini 2.5 Flash to identify and rank self-contained moments.
- Validate model timestamps, reject malformed candidates and remove overlapping selections.
- Review and select candidate clips before rendering.
- Export 9:16 MP4 clips with blurred background fill, timed phrase captions with word-by-word karaoke highlighting, and sampled face-aware horizontal framing.
- Gemini key is entered at runtime and sent in an HTTP header; it is not saved by the app.

## Requirements

- Python 3.10+
- FFmpeg and FFprobe installed and available in PATH
- Gemini API key (used for moment selection)
- 4 GB+ RAM recommended; choose Whisper `tiny` or `base` on lower-memory machines. OpenCV face detection adds CPU work during rendering.

## Windows setup

```powershell
cd general_clipper
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The CI workflow also compiles the Python modules to catch syntax errors before running tests.

Run validation tests:

```powershell
python -m unittest discover -s tests -v
```

## Workflow

1. Upload a video you own or have permission to edit.
2. Enter a Gemini API key in the sidebar.
3. Choose a Whisper model and clip length.
4. Analyze, review the ranked moments, and render selected clips.
5. Download and manually review each MP4 before publishing.

## Notes and limitations

- This MVP focuses on spoken content. Silent action/gameplay highlight detection is not implemented.
- Local Whisper transcription can be slow on CPU. The first run downloads the selected model.
- Gemini receives transcript text only, not the uploaded video or audio. Long transcripts are split into bounded requests so later portions are analyzed too; API quotas and processing time increase with video length.
- Vertical output uses sampled Haar face detection to keep the largest visible face near the crop center, with a centered fallback when no face is detected. It is not identity-aware multi-speaker tracking and does not track gameplay objects.
- No automatic publishing, campaign scraping, or view/earnings guarantees.
- AI scores are suggestions, not predictions of virality.
- Respect source ownership, creator permissions, campaign rules and platform terms. Credit alone is not permission.

## Privacy

Video processing and transcription happen locally. Only transcript text and timestamps are sent to Google's Gemini API for moment selection. The API key is held in the current app session and is not written to disk.

## License

MIT
