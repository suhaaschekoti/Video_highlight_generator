# AI-Driven Video Highlight Generator
## Explainable Multimodal Highlight and Trailer Creation from Long-Form Video

**Course:** Introduction to Artificial Intelligence
**Institution:** Indian Institute of Information Technology Kottayam
**Team:** Suhaas Chekoti (2024BCS0267), [add teammates]
**Date:** September 2026

---

## 1. Abstract

Long-form video is cheap to record and expensive to watch. This project delivers a complete, zero-cost system that converts a long video — lecture, sports match, vlog or interview — into a short highlight reel while exposing the reasoning behind every cut.

Four independent signals are computed for every spoken segment and fused with domain-specific weights: (i) **meaning**, from a large language model scoring the Whisper transcript; (ii) **acoustic excitement**, from loudness, pitch variability and spectral flux with a speech-versus-music gate; (iii) **editorial structure**, from scene-cut detection; and (iv) **visual content**, from a vision-language model captioning the frames of the strongest candidates. A budgeted greedy selector, boundary refinement on word timestamps and silences, and FFmpeg rendering produce the reel together with captions, chapter markers, a JSON/text manifest, and a label file.

A web application makes the process interactive: a score timeline shows every segment's four signals, the user can re-weight the fusion or force segments in and out and re-cut in seconds, and their edits can be saved as ground truth. Supporting tools provide time-overlap evaluation, inter-annotator agreement, ablations, batch processing and learned fusion weights. All components run on free tiers (Groq) or open-source software.

---

## 2. Problem Statement

### 2.1 Definition
Given a video *V* of length *T* and a target length *L* ≪ *T*, choose non-overlapping intervals whose total length is ≈ *L* and which maximise "highlight value", then present them in order as a coherent short video, with a human-readable justification for each interval.

### 2.2 Why it is hard
1. **Subjectivity.** Highlight is a judgement, not a measurement; different viewers and domains disagree.
2. **Multimodality.** Speech, sound, editing and picture each carry part of the signal; none is sufficient alone.
3. **Constrained selection.** The best set of clips under a length budget must also be watchable: no mid-word cuts, no stuttering between adjacent clips.
4. **Evaluation.** With no single truth, evaluation must be relative to human labels and must report how much humans agree with each other.

### 2.3 Applications
Lecture recaps and revision, sports and e-sports clipping, interview and podcast trailers, meeting recaps, social-media repurposing, accessibility for time-constrained viewers.

---

## 3. Related Work

| Work | What we took from it |
|---|---|
| Unsupervised Video Highlight Detection by Learning from Audio and Visual Recurrence (arXiv:2407.13933, 2024) | Audio is an under-used, highly informative cue → dedicated acoustic signal, audio–visual fusion. |
| Video Summarization with Large Language Models, LLMVS (arXiv:2504.11199, 2025) | An LLM judging importance from text tracks human annotation → LLM as the primary judge. |
| Unsupervised Transcript-assisted Video Summarization and Highlight Detection (arXiv:2505.23268, 2025) | Transcripts localise highlights better than vision alone but miss brief peaks → audio and visual signals fill that gap. |
| Adaptive Video Highlight Detection by Learning from User History (ECCV 2020); PHD-GIFs (Google/NTU) | Preference is personal → domain presets, manual overrides, learned weights from a user's own labels. |
| Mr. HiSum (NeurIPS 2023) | Terminology: highlight detection (most engaging parts) vs. summarisation (coverage). We do the former. |
| VT-SSum, YTSEG (lecture transcript summarisation and chaptering) | Spoken content needs sentence-level segmentation; chaptering is a natural by-product → chapter generation. |

Our contribution is engineering rather than a new model: an explainable, interactive fusion of an LLM judge, a VLM, and classical signal processing, with an evaluation protocol that takes subjectivity seriously.

---

## 4. System Architecture

### 4.1 Overview

```
                       ┌──────────────────────────────────────────────────────────┐
 video / URL ────────► │ 1. Transcribe: ffmpeg → 16 kHz FLAC → Whisper             │
 (upload or yt-dlp)    │    segments + word timestamps, cached                     │
                       └──────────────┬───────────────────────────────────────────┘
          ┌───────────────┬───────────┼───────────────┬──────────────────┐
          ▼               ▼           ▼               ▼                  │
 ┌────────────────┐ ┌───────────┐ ┌──────────┐ ┌────────────────────┐   │
 │ 2. Meaning     │ │ 3. Audio  │ │ 4. Scene │ │ 5. Visual (VLM on  │   │
 │ LLM 0–1 +      │ │ loudness, │ │ cut      │ │ top-k candidates   │   │
 │ reason         │ │ pitch,    │ │ proximity│ │ from a preliminary │   │
 │                │ │ flux,gate │ │          │ │ fusion) 0–1+caption│   │
 └───────┬────────┘ └─────┬─────┘ └────┬─────┘ └─────────┬──────────┘   │
         └────────────────┴────────────┴─────────────────┘              │
                                       ▼                                │
                  ┌──────────────────────────────────────────┐          │
                  │ 6. Fuse (domain / learned weights)        │  ANALYSIS (slow, saved to data/jobs)
                  │    + chapters (LLM)                       │          │
                  └───────────────────┬──────────────────────┘          │
  ────────────────────────────────────┼──────────────────────────────────┘
                                      ▼
                  ┌──────────────────────────────────────────┐
                  │ 7. Select: greedy budget, overrides, pad, │
                  │    order, merge                           │  RENDER (seconds; re-run on every re-cut)
                  │ 8. Refine cuts: word edges + silences     │
                  │ 9. ffmpeg cut/concat, SRT, chapters,      │
                  │    title/summary, manifests, labels       │
                  └──────────────────────────────────────────┘
                                      ▼
            Web UI: timeline · sliders · picks · player · chapters · clip list · history
```

The split into **analysis** (expensive, cached) and **render** (cheap) is the key architectural decision: it is what makes interactive re-weighting, manual overrides and ablations practical.

### 4.2 Transcription
FFmpeg extracts mono 16 kHz FLAC. Two interchangeable backends:
- **Groq API**, `whisper-large-v3-turbo` — no local model or PyTorch; audio is chunked into 10-minute pieces to respect the upload limit and timestamps are shifted back; word-level timestamps are requested alongside segments.
- **faster-whisper** — fully offline, with voice-activity filtering and word timestamps.

Both return identical `Segment` objects (id, start, end, text, confidence, words). Transcripts are cached per video, so re-runs skip this stage.

### 4.3 Meaning (LLM) — primary signal
Segments are batched (20 per request) to `llama-3.3-70b-versatile` on Groq with a domain-aware prompt and JSON mode. Each returns a score in [0, 1] and a ≤12-word reason. The prompt instructs the model to use the full range and to treat highlights as rare, which counters the tendency to cluster at 0.5. Rate limits are handled with exponential backoff; skipped segments default to 0.1.

### 4.4 Acoustic excitement + speech gate
For each segment: RMS loudness, pitch variability (YIN f0 standard deviation) and spectral flux. Each is robust-z-scored (median/MAD) and logistic-squashed so one applause burst cannot flatten the rest; they are combined 0.5/0.25/0.25.

**Speech gate.** Speech has strong amplitude modulation at the syllable rate (2–8 Hz); sustained music does not. The fraction of the envelope spectrum's energy in 2–8 Hz maps to a gate in [0.3, 1] that multiplies the excitement score, so background music cannot impersonate a cheering crowd. It is cheap, dependency-free and tunable.

### 4.5 Scene-cut proximity
PySceneDetect's content detector (HSV frame difference, threshold 27) finds cuts. A segment scores `max(0, 1 − d/2 s)` for distance *d* to the nearest cut. It is a weak supporting signal: cuts mark boundaries an editor chose, not importance. It fails soft to zeros for audio-only input.

### 4.6 Visual re-ranking (VLM)
Scoring every frame would cost hundreds of requests. Instead a **preliminary fusion of the other three signals** picks the top-*k* (default 40) candidates; one mid-segment JPEG each is sent to `meta-llama/llama-4-scout-17b-16e-instruct` on Groq, which returns a caption and a 0–1 visual-interest score. Non-candidates score 0. The visual weight is deliberately modest so this **refines** the ranking rather than dominating it, and the captions appear in the UI and manifest.

### 4.7 Fusion
`final = Σ_k w_k · signal_k` over {meaning, audio, scene, visual}. Weights come from domain presets, overridden by learned weights (4.11) when available, and are re-normalised whenever a signal is disabled. Because raw signals are stored, `refuse()` recomputes `final` for new weights without any API call.

| preset | meaning | audio | scene | visual | rationale |
|---|---|---|---|---|---|
| lecture | 0.65 | 0.10 | 0.10 | 0.15 | flat delivery; meaning carries the signal |
| sports | 0.30 | 0.40 | 0.10 | 0.20 | crowd/commentator energy; action is visible |
| vlog | 0.45 | 0.25 | 0.10 | 0.20 | reactions and content both matter |
| general | 0.50 | 0.25 | 0.10 | 0.15 | default |

### 4.8 Selection
1. Rank by `final`; **forced-in** segments rank first, **forced-out** segments are skipped.
2. Take segments until the target length is filled; extend any shorter than 3 s to 3 s.
3. Pad 0.5 s before / 0.7 s after; cap at 25 s; clamp to the video.
4. Sort chronologically; merge clips overlapping or within 1.5 s.

Greedy selection is *O(n log n)* and near-optimal here because clip values are independent; a knapsack DP would gain little since merging (step 4) changes the items anyway.

### 4.9 Boundary refinement
Each clip's start is moved to the first word's onset minus lead-in and its end to the last word's offset plus lead-out; both are then snapped to the nearest silence edge (librosa non-silent intervals, ±0.6/0.8 s) with a 0.15 s margin. Cuts therefore land on pauses, not mid-word. Clips that would collapse below 1 s fall back to their original bounds.

### 4.10 Rendering and outputs
Frame-accurate re-encoded cuts (libx264, CRF 20) concatenated losslessly; optional caption burn-in; audio-only sources produce an audio reel. Outputs: reel, `.srt` on the reel's own timeline, YouTube-style `chapters.txt` (LLM-generated from a thinned transcript), LLM title and summary, `manifest.json` (all four signals per segment, clips, chapters, run metadata), `manifest.txt`, and `labels.json` (the clip set as a truth file).

### 4.11 Learned fusion
Every saved analysis with a matching truth file contributes one row per segment: features = the four signals, target = fraction of the segment inside a labelled highlight. Non-negative least squares (projected gradient descent, numpy) fits per-domain weights, which are normalised and written to `data/learned_weights.json`; `config.py` uses them automatically. The UI's *Save picks as labels* button makes labelling part of normal use.

### 4.12 Web application
FastAPI backend; jobs run in background threads and persist to disk, so finished reels survive restarts. Endpoints: create job (upload or URL via yt-dlp), progress, re-cut (new weights, length, forced picks), files, history, delete. Uploads are validated by extension and size (1.5 GB cap). The single-file front end (no build step) provides:
- source tabs (upload / link), domain preset with a live weight bar, length, ASR backend, vision toggle, caption burn-in;
- seven-stage progress;
- the **score timeline** — one bar per segment, height = fused score, amber = in reel, chapter markers overlaid; hover shows text, all five scores, the LLM reason and the VLM caption; click cycles a segment through *force in* (green) → *force out* (red) → neutral;
- four weight sliders and a length slider with **Re-cut reel** (fusion + render only, typically seconds);
- player, downloads, chapters, clickable clip list, and a persistent history panel.

Accessibility: keyboard-operable drop zone and controls, visible focus rings, reduced-motion respected.

---

## 5. Technologies

| Layer | Tool | Cost |
|---|---|---|
| Language | Python 3.10+ | free |
| ASR | Whisper large-v3-turbo (Groq) / faster-whisper | free tier / open source |
| LLM | Llama 3.3 70B (Groq, JSON mode) | free tier |
| VLM | Llama 4 Scout (Groq, image input) | free tier |
| Audio | librosa, soundfile, numpy | open source |
| Vision | PySceneDetect, OpenCV | open source |
| Video | FFmpeg / ffprobe, yt-dlp | open source |
| Backend | FastAPI, uvicorn, pydantic | open source |
| Front end | HTML / CSS / vanilla JS, IBM Plex fonts | open source |

Total running cost: ₹0. Hardware: any laptop; a GPU only accelerates the optional local ASR.

---

## 6. Mapping to the Course Syllabus

| Unit | Where it appears |
|---|---|
| Characteristics of AI problems; building AI systems | §2 (subjectivity, multimodality, evaluation); the analyse/render architecture |
| Search techniques and complexity | Budgeted greedy selection with cost analysis and the knapsack alternative (§4.8) |
| Knowledge representation | The manifest as a structured representation of "what matters and why"; domain presets as production rules (IF domain = sports THEN weight audio 0.40); chapters as a hierarchical representation of the video |
| Neural networks | Whisper, Llama 3.3 and Llama 4 Scout are the learned components; the report contrasts them with the hand-designed signals |
| Intelligent agents | The orchestrator is a goal-based agent: percepts (transcript, audio, frames) → utility (fused score) → actions (select, cut, render); the re-cut loop adds a human in the loop |
| Fuzzy logic | Acoustic excitement and the speech gate are graded memberships in [0, 1] built from robust scaling and logistic squashing, not crisp thresholds |
| Expert systems | Every decision carries a justification (LLM reason, VLM caption, per-signal scores) — an explanation facility |

---

## 7. Evaluation Protocol

### 7.1 Quantitative
For each test video a team member labels highlight intervals (`data/truth/<name>.json`; the UI can export these). `src/evaluate.py` reports time-overlap **precision**, **recall**, **F1** and **highlight hit-rate** (fraction of labelled highlights touched by any clip). Time-overlap is used so partial hits count proportionally and the metric does not depend on Whisper's segmentation. With two annotators, **Jaccard agreement** between their label sets is reported as the ceiling any system could reach.

### 7.2 Ablations
`src/ablate.py` re-fuses a saved analysis as *full*, *no_<signal>* and *only_<signal>* for each of the four signals and evaluates each variant — no transcription or API calls, so the full table takes seconds per video.

### 7.3 Learned vs. preset weights
After labelling, `src/learn_weights.py` fits weights per domain; the same videos are re-evaluated with learned weights (in-sample) and, where enough videos exist, with leave-one-out.

### 7.4 Qualitative
Three to five viewers rate each reel on 1–5 scales for coherence, informativeness and usefulness, and judge whether the stated reasons match what they saw.

### 7.5 Test set
Self-curated short (5–10 min) public videos, one to two per domain. `src/batch.py` runs the whole folder and writes `results.csv` with timing per source minute.

---

## 8. Verification Performed

The following was executed in the development environment (which has no Groq key and could not install PyTorch):

- **Unit tests** on synthetic data: preset and learned-weight loading and normalisation; four-signal fusion; greedy selection with forced-in/forced-out overrides; chronological merge; analysis save/load; ablation variant generation; non-negative weight fitting; evaluation metrics and agreement; SRT and chapter writers.
- **Integration test on a real synthetic video** (60 s, four colour scenes, two loud noise bursts, mocked transcript with word timestamps and mocked LLM scores): audio decoding via FFmpeg, acoustic scoring (both bursts scored 1.0), scene detection (cuts found at exactly 15/30/45 s), fusion, selection, boundary refinement, FFmpeg cutting and concatenation (17 s reel, three clips), captions, chapters, manifests and labels.
- **Server smoke test**: presets, job listing, static UI, extension and duration validation, a job that fails cleanly without an API key, the re-cut guard, deletion; front-end JavaScript syntax-checked with Node.

One integration bug was found and fixed during this: librosa 1.0 cannot open video containers directly, so audio is now decoded through FFmpeg to a cached WAV.

**Not yet verified**: the three Groq-dependent calls (Whisper transcription with word timestamps, LLM scoring/summary/chapters, VLM frame scoring) and yt-dlp download. These need a key and network; they are the first things to run on a real machine.

---

## 9. Results

*(to be completed after real runs — the table structure is fixed so results drop straight in)*

| Video | Domain | Source | Reel | Clips | P | R | F1 | Hit | Agreement | Full F1 | no_meaning | no_audio | no_scene | no_visual | learned F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | lecture | | | | | | | | | | | | | | |
| | sports | | | | | | | | | | | | | | |
| | vlog | | | | | | | | | | | | | | |

Record alongside: seconds of processing per source minute (Groq vs local ASR, vision on/off); Likert means; a gallery of failure cases with the manifest's reasons.

---

## 10. Limitations

1. **Speech-dependent.** Little or no speech leaves only audio, scene and visual signals.
2. **Visual scoring is sampled.** One frame per candidate segment can miss action that happens between samples; only top-*k* candidates are scored at all.
3. **Acoustic proxies are heuristics.** The speech gate suppresses steady music but not, for example, sound effects; a trained audio-event tagger would be stronger.
4. **LLM and VLM scores are not bit-reproducible** (temperature 0.2); clip-level results are stable in practice.
5. **Free-tier limits.** Very long videos with vision enabled can hit rate limits; batching, backoff and `visual_top_k` mitigate.
6. **Ground truth is one person's taste** unless agreement is measured; the protocol requires it but the data must still be collected.
7. **Learned weights are linear.** A non-linear model (gradient boosting) might capture interactions between signals; linear weights were chosen for interpretability.

---

## 11. Future Work

- Dense visual scoring with a lightweight local model (CLIP-style embeddings) so every segment gets a visual score.
- Pretrained audio-event tagging (applause, laughter, cheering) replacing prosodic heuristics.
- Non-linear fusion and personalisation from a user's history of picks.
- Speaker diarisation to weight "who said it" and to build speaker-specific reels.
- Multilingual output: translated captions and dubbed reels from the same transcript.
- Streaming/near-real-time operation for live events.

---

## 12. Conclusion

The project shows that a useful, explainable, interactive highlight generator can be built by combining LLM and VLM judgement with classical audio and vision cues under a transparent fusion rule — without training a model, without paid services, and with a research-grade evaluation loop (labels, agreement, ablations, learned weights) built into ordinary use. Its architecture separates the expensive analysis from the cheap decision, which is what turns a batch tool into an interactive one.

---

## 13. References

1. Unsupervised Video Highlight Detection by Learning from Audio and Visual Recurrence. arXiv:2407.13933, 2024.
2. Video Summarization with Large Language Models (LLMVS). arXiv:2504.11199, 2025.
3. Unsupervised Transcript-assisted Video Summarization and Highlight Detection. arXiv:2505.23268, 2025.
4. Adaptive Video Highlight Detection by Learning from User History. ECCV 2020.
5. PHD-GIFs: Personalized Highlight Detection for Automatic GIF Creation. Google Research / NTU.
6. Mr. HiSum: A Large-scale Dataset for Video Highlight Detection. NeurIPS 2023.
7. Scribby: A Multi-Level LLM Framework for Semantic Video Analysis. arXiv:2606.14762, 2026 (VT-SSum, YTSEG).
8. Radford et al. Robust Speech Recognition via Large-Scale Weak Supervision (Whisper). 2022.
9. de Cheveigné & Kawahara. YIN, a fundamental frequency estimator for speech and music. JASA 2002.
10. librosa, PySceneDetect, FFmpeg, yt-dlp, FastAPI — project documentation.

## Appendix — Running the system
See `README.md`: web app `uvicorn app.server:app --port 8000`; CLI `python -m src.pipeline`; tools `src.evaluate`, `src.ablate`, `src.learn_weights`, `src.batch`.
