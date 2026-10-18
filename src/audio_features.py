"""
Stage 2a: acoustic excitement + speech/music gate.

Excitement = 0.5·loudness + 0.25·pitch variability + 0.25·spectral flux
(each robust-z-scored across the video, logistic-squashed to (0,1)).

Speech gate: speech has strong amplitude modulation at the syllable rate
(~2–8 Hz); sustained music does not, and music is spectrally flatter over
time. A gate in [0.3, 1] multiplies the excitement so background music
can't masquerade as a cheering crowd. Cheap, dependency-free, tunable.
"""

import numpy as np
import librosa

from src.transcribe import Segment


def load_audio(path: str, sr: int = 16000, cache_dir: str = "data/transcripts"):
    """Decode any container to 16 kHz mono via ffmpeg (soundfile can't open mp4/mkv),
    caching the wav so render/re-cut don't re-decode."""
    import os, subprocess
    from pathlib import Path
    os.makedirs(cache_dir, exist_ok=True)
    wav = Path(cache_dir) / f"{Path(path).stem}.{sr}.wav"
    if not wav.exists() or wav.stat().st_mtime < Path(path).stat().st_mtime:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr),
                        "-map", "0:a:0", "-f", "wav", str(wav)], check=True, capture_output=True)
    y, sample_rate = librosa.load(str(wav), sr=sr, mono=True)
    return y, sample_rate


def _features(clip: np.ndarray, sr: int) -> tuple[float, float, float, float]:
    hop = 256
    rms_track = librosa.feature.rms(y=clip, frame_length=1024, hop_length=hop)[0]
    rms = float(np.mean(rms_track))

    try:
        f0 = librosa.yin(clip, fmin=65, fmax=1000, sr=sr)
        f0 = f0[np.isfinite(f0)]
        pitch_var = float(np.std(f0)) if len(f0) > 2 else 0.0
    except Exception:
        pitch_var = 0.0

    S = np.abs(librosa.stft(clip, n_fft=1024, hop_length=512))
    flux = float(np.mean(np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0)))) if S.shape[1] > 1 else 0.0

    # syllable-rate modulation: fraction of envelope spectrum energy in 2–8 Hz
    env = rms_track - np.mean(rms_track)
    speech_mod = 0.0
    if len(env) > 16:
        spec = np.abs(np.fft.rfft(env))
        freqs = np.fft.rfftfreq(len(env), d=hop / sr)
        band = (freqs >= 2) & (freqs <= 8)
        total = np.sum(spec[1:]) + 1e-9
        speech_mod = float(np.sum(spec[band]) / total)
    return rms, pitch_var, flux, speech_mod


def _robust(v: np.ndarray) -> np.ndarray:
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-9
    z = (v - med) / (1.4826 * mad)
    return 1.0 / (1.0 + np.exp(-z))


def score_segments_by_audio_energy(y, sr, segments: list[Segment], use_speech_gate=True,
                                   w_rms=0.5, w_pitch=0.25, w_flux=0.25) -> dict[int, float]:
    feats = []
    for seg in segments:
        clip = y[int(seg.start * sr):int(seg.end * sr)]
        feats.append(_features(clip, sr) if len(clip) >= sr * 0.2 else (0.0, 0.0, 0.0, 0.0))
    if not feats:
        return {}
    a = np.array(feats)
    excite = w_rms * _robust(a[:, 0]) + w_pitch * _robust(a[:, 1]) + w_flux * _robust(a[:, 2])

    if use_speech_gate:
        mod = a[:, 3]
        # map modulation ratio to a gate in [0.3, 1]; ~0.35+ is clearly speech-like
        gate = 0.3 + 0.7 * np.clip((mod - 0.15) / 0.25, 0.0, 1.0)
        excite = excite * gate

    lo, hi = excite.min(), excite.max()
    norm = (excite - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(excite)
    return {seg.id: float(v) for seg, v in zip(segments, norm)}
