"""
Stage 2b: Visual scene-change detection (supporting signal).

Tells us *where* visual cuts happen (camera change, slide transition, replay
cut). A segment starting near a cut gets a small boost, on the reasoning that
editors and directors place cuts at meaningful boundaries. Fails soft: if
detection errors out (odd codec, no video stream) every segment scores 0.
"""

from src.transcribe import Segment


def detect_scene_boundaries(video_path: str, threshold: float = 27.0) -> list[float]:
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path)
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold))
        manager.detect_scenes(video)
        return sorted({scene[0].get_seconds() for scene in manager.get_scene_list()})
    except Exception:
        return []


def score_segments_by_scene_proximity(segments: list[Segment], boundaries: list[float],
                                      window_s: float = 2.0) -> dict[int, float]:
    if not boundaries:
        return {seg.id: 0.0 for seg in segments}
    scores = {}
    for seg in segments:
        d = min(abs(seg.start - b) for b in boundaries)
        scores[seg.id] = max(0.0, 1.0 - d / window_s)
    return scores
