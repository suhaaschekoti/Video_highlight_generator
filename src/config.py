"""
Central configuration. Everything tunable lives here.
Learned weights (from src/learn_weights.py) override presets when present.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DOMAIN_PRESETS: dict[str, dict[str, float]] = {
    "lecture": {"semantic": 0.65, "audio": 0.10, "scene": 0.10, "visual": 0.15},
    "sports":  {"semantic": 0.30, "audio": 0.40, "scene": 0.10, "visual": 0.20},
    "vlog":    {"semantic": 0.45, "audio": 0.25, "scene": 0.10, "visual": 0.20},
    "general": {"semantic": 0.50, "audio": 0.25, "scene": 0.10, "visual": 0.15},
}
SIGNALS = ("semantic", "audio", "scene", "visual")
LEARNED_WEIGHTS_PATH = Path("data/learned_weights.json")


def load_weights(domain: str) -> dict[str, float]:
    """Preset weights, replaced by learned weights for that domain if a fit exists."""
    w = dict(DOMAIN_PRESETS.get(domain, DOMAIN_PRESETS["general"]))
    if LEARNED_WEIGHTS_PATH.exists():
        try:
            learned = json.loads(LEARNED_WEIGHTS_PATH.read_text())
            if domain in learned:
                w.update(learned[domain])
        except (json.JSONDecodeError, OSError):
            pass
    return w


@dataclass
class PipelineConfig:
    domain: str = "general"

    # transcription
    asr_backend: str = "groq"                    # "groq" | "local"
    groq_asr_model: str = "whisper-large-v3-turbo"
    local_asr_model: str = "small"
    asr_language: str | None = None
    asr_chunk_seconds: int = 600

    # LLM
    llm_model: str = "openai/gpt-oss-120b"  
    llm_batch_size: int = 20
    llm_max_retries: int = 5

    # vision (Groq-hosted VLM; only the top candidates get a frame scored)
    use_visual: bool = True
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    visual_top_k: int = 40

    # audio
    use_speech_gate: bool = True                 # damp audio score where the sound is music, not speech/reaction

    # scene
    scene_detector_threshold: float = 27.0

    # fusion weights (filled from presets/learned in __post_init__)
    weights: dict[str, float] = field(default_factory=dict)

    # selection
    target_highlight_duration_s: int = 120
    min_clip_duration_s: float = 3.0
    max_clip_duration_s: float = 25.0
    lead_in_padding_s: float = 0.5
    lead_out_padding_s: float = 0.7
    merge_gap_s: float = 1.5
    refine_boundaries: bool = True               # snap cuts to word edges / silences

    # outputs
    burn_subtitles: bool = False
    generate_summary: bool = True
    generate_chapters: bool = True

    output_dir: str = "data/output"
    transcript_cache_dir: str = "data/transcripts"
    jobs_dir: str = "data/jobs"

    def __post_init__(self):
        base = load_weights(self.domain)
        base.update(self.weights or {})
        if not self.use_visual:
            base["visual"] = 0.0
        self.weights = normalise(base)

    def validate(self):
        assert self.asr_backend in ("groq", "local")
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, self.weights


def normalise(w: dict[str, float]) -> dict[str, float]:
    w = {k: max(0.0, float(w.get(k, 0.0))) for k in SIGNALS}
    s = sum(w.values()) or 1.0
    return {k: v / s for k, v in w.items()}


def get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError("GROQ_API_KEY not set. Free key: https://console.groq.com/keys — put it in .env")
    return key
