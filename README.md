
# AI Highlight Generator

Turns a long video — lecture, sports match, vlog, interview — into a short highlight
reel, and shows *why* every clip was chosen. Four signals are fused: what was said
(scored by an LLM), how it sounded (loudness, pitch, spectral flux, with a speech/music
gate), where the editor cut (scene detection), and what the frame shows (a vision model
on the top candidates). Everything runs on free tiers and open-source tools: ₹0 to run.

```
video ─► transcribe (Whisper) ─► LLM meaning score ─┐
                              ├► audio excitement ───┼► fuse (domain weights) ─► select ─► refine cuts ─► reel.mp4
                              ├► scene-cut proximity ─┤                                                    captions, chapters,
                              └► VLM frame score ────┘                                                    manifest, labels
```

## Setup

Prerequisites: Python 3.10+, FFmpeg on PATH (`ffmpeg -version`), a free Groq key from
https://console.groq.com/keys (no card).

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # paste GROQ_API_KEY=gsk_... into it
```

## Web app

```bash
uvicorn app.server:app --port 8000     # open http://localhost:8000
```

- Upload a file or paste a YouTube/other URL (yt-dlp).
- Pick the domain preset, target length, transcription backend, vision on/off, caption burn-in.
- Watch stage-by-stage progress. Results: LLM title + summary, a **score timeline** (hover any
  segment for its meaning/audio/scene/visual/final breakdown, the LLM's reason and the VLM caption),
  the reel, downloads (video, .srt, chapters, manifest, labels), chapters, and a clickable clip list.
- **Re-cut in seconds**: drag the four weight sliders or the length slider; click timeline bars to
  force a segment in (green) or out (red); press *Re-cut reel*. Only fusion + rendering re-run.
- **Save picks as labels** exports your hand-edited selection as a ground-truth file for evaluation
  and weight learning.
- Earlier reels persist across restarts (left panel); delete from there.

## Command line

```bash
python -m src.pipeline --input data/input/talk.mp4 --domain lecture --duration 90
python -m src.pipeline --input data/input/match.mp4 --domain sports --duration 120 --burn-subtitles
python -m src.pipeline --input data/input/x.mp4 --asr local --no-visual     # fully offline except LLM scoring
```

## Evaluation and research tools

```bash
# time-overlap precision / recall / F1 / hit-rate vs. hand labels (+ inter-annotator agreement)
python -m src.evaluate --manifest data/output/<job>/x_manifest.json --truth data/truth/x.json [--truth2 data/truth/x_b.json]

# ablations: full, no_<signal>, only_<signal> — from the saved analysis, no API calls
python -m src.ablate --analysis data/jobs/x.analysis.json --truth data/truth/x.json --duration 120

# fit per-domain fusion weights from every labelled video (writes data/learned_weights.json, picked up automatically)
python -m src.learn_weights

# process a whole folder and write results.csv (evaluates wherever a truth file exists)
python -m src.batch --input data/input --domain sports --duration 120
```

Truth file format: `{"highlights": [{"start": 62.0, "end": 78.5}, ...]}` — the UI's
*Save picks as labels* produces exactly this.

## Configuration

`src/config.py` holds every knob. Presets (overridden by learned weights when present):

| preset  | meaning | audio | scene | visual |
|---------|---------|-------|-------|--------|
| lecture | 0.65    | 0.10  | 0.10  | 0.15   |
| sports  | 0.30    | 0.40  | 0.10  | 0.20   |
| vlog    | 0.45    | 0.25  | 0.10  | 0.20   |
| general | 0.50    | 0.25  | 0.10  | 0.15   |

Other knobs: clip min/max length, padding, merge gap, `visual_top_k`, `use_speech_gate`,
`refine_boundaries`, `llm_batch_size`, `asr_chunk_seconds`, and the prompts in `src/llm_scorer.py`.

## Layout

```
app/server.py           FastAPI: upload / URL jobs, progress, re-cut, files, history, delete; persists to data/jobs
app/static/index.html   single-file UI
src/config.py           presets, learned-weight loading, PipelineConfig
src/transcribe.py       ffmpeg extraction; Groq Whisper (chunked, word timestamps) or faster-whisper
src/llm_scorer.py       LLM scoring w/ backoff, reel title+summary, chapters
src/audio_features.py   loudness / pitch / flux + speech-vs-music gate
src/scene_detect.py     PySceneDetect cuts → proximity score
src/visual_scorer.py    Groq VLM scores frames of top-k candidates
src/fusion.py           4-signal fusion, re-fuse, greedy selection, overrides, merging
src/boundaries.py       snap cuts to word edges and silences
src/clip_extractor.py   ffmpeg cut/concat, SRT, chapters, manifests, label export
src/pipeline.py         analyse() [slow, cached] + render() [fast]; CLI
src/evaluate.py         P/R/F1/hit-rate, inter-annotator agreement
src/ablate.py           ablation table from a saved analysis
src/learn_weights.py    non-negative least squares → learned presets
src/batch.py            folder → results.csv
src/download.py         yt-dlp wrapper
```

## Troubleshooting

- `GROQ_API_KEY not set` → `.env` must sit in the directory you launch from.
- HTTP 429 → free-tier rate limit; already retried with backoff. Raise `llm_batch_size`, lower `visual_top_k`, or untick vision.
- Upload-size error from Groq ASR on very long videos → lower `asr_chunk_seconds`.
- `yt-dlp` not found → `pip install yt-dlp` inside the venv.
- Slow local ASR → `local_asr_model = "base"`.
