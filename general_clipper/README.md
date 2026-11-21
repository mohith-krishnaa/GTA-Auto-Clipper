# MK AI Clipper

A general-purpose, local-first AI clipping MVP for podcasts, interviews, gaming videos, tutorials and other speech-led content.

## Features

- Upload a local video (MP4, MOV, MKV, WebM or M4V).
- Transcribe speech locally with faster-whisper (CPU int8).
- Ask Gemini 2.5 Flash to identify and rank self-contained moments.
- Review and select candidate clips before rendering.
- Export 9:16 MP4 clips with blurred background fill and burned-in word captions.
- Gemini key is entered at runtime and is not stored by the app.

## Requirements

- Python 3.10+
- FFmpeg and FFprobe installed and available in PATH
- Gemini API key (used for moment selection)
- 4 GB+ RAM recommended; choose Whisper `tiny` or `base` on lower-memory machines.

## Windows setup

```powershell
cd general_clipper
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## Workflow

1. Upload a video you own or have permission to edit.
2. Enter a Gemini API key in the sidebar.
3. Choose a Whisper model and clip length.
4. Analyze, review the ranked moments, and render selected clips.
5. Download and manually review each MP4 before publishing.

## Notes and limitations

- This is an MVP focused on spoken content. Silent action/gameplay highlight detection is not implemented yet.
- Local Whisper transcription can be slow on CPU. The first run downloads the selected model.
- Gemini receives transcript text only, not the uploaded video or audio.
- Vertical output uses a blurred background and centered full-frame foreground; speaker/face tracking is not implemented yet.
- No automatic publishing, campaign scraping, or view/earnings guarantees.
- AI scores are suggestions, not predictions of virality.
- Respect source ownership, creator permissions, campaign rules and platform terms. Credit alone is not permission.

## Privacy

Video processing and transcription happen locally. Only the transcript text and timestamps are sent to Google's Gemini API for moment selection. The API key is held in the current app session and is not written to disk.

## License

MIT
