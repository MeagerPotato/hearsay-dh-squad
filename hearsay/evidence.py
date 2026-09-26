"""Shared contract between forensic experts, the orchestrator, fusion and the report writer.

Every expert is a function `analyze(ctx) -> Evidence`, registered with @expert(...). The
orchestrator decides which experts run for a file (cheap-first cascade); fusion consumes
`Evidence.features`; the report renders `Evidence.findings`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

CATEGORIES = (
    "container",      # file/container forensics, metadata, MAC times, codec chain
    "spectral",       # spectrogram / frequency-domain artefacts, band-limiting, vocoder traces
    "prosody",        # pitch contour, jitter/shimmer, pauses, breaths, speaking rate
    "environment",    # ENF mains hum, background/noise-floor consistency, reverb, replay cues
    "compression",    # double encoding, transcoding, codec cut-offs
    "speaker",        # speaker-embedding consistency / drift within the clip
    "deep",           # deep-learning anti-spoofing detectors
    "splice",         # discontinuities: phase breaks, DC jumps, digital-silence gaps, seams
    "semantic",       # transcript, ASR confidence, repeated-script detection
)


@dataclass
class Finding:
    """One human-readable piece of evidence. direction: +1 suggests synthetic, -1 suggests real, 0 neutral."""
    text: str
    direction: int = 0
    strength: float = 0.0  # 0..1, how strongly this finding should sway an analyst


@dataclass
class Evidence:
    expert: str
    category: str
    features: Dict[str, float] = field(default_factory=dict)   # numeric features for fusion
    findings: List[Finding] = field(default_factory=list)      # explanations for the report
    score: Optional[float] = None       # optional spoof-leaning score (higher = more synthetic)
    applicable: bool = True             # False when the expert has nothing to say for this file
    runtime_ms: float = 0.0
    artifacts: Dict[str, Any] = field(default_factory=dict)    # arrays for plots (not serialised to TSV)
    # Whole-expert failure (set by @expert on an exception, with applicable=False); such evidence is excluded
    # from scoring. A failed sub-analysis whose other outputs are valid belongs in artifacts["partial_errors"]
    # (the orchestrator moves an `error` set with applicable=True there).
    error: Optional[str] = None


@dataclass
class Context:
    """Everything experts may need for one file. Heavy intermediates are cached in `cache`."""
    path: str
    audio: np.ndarray                    # mono float32 at `sr` (16 kHz analysis rate)
    sr: int
    native: Dict[str, Any] = field(default_factory=dict)   # ffprobe/container info, native sample rate, etc.
    cache: Dict[str, Any] = field(default_factory=dict)    # shared intermediates (stft, f0, transcript...)
    corpus: Dict[str, Any] = field(default_factory=dict)   # batch-level context (e.g. transcript counts)


@dataclass
class ExpertSpec:
    name: str
    category: str
    fn: Callable[[Context], Evidence]
    cost: str = "cheap"           # "cheap" | "medium" | "expensive" — used by the router
    description: str = ""


REGISTRY: Dict[str, ExpertSpec] = {}


def expert(name: str, category: str, cost: str = "cheap", description: str = ""):
    assert category in CATEGORIES, category

    def deco(fn: Callable[[Context], Evidence]):
        def wrapped(ctx: Context) -> Evidence:
            t = time.perf_counter()
            try:
                ev = fn(ctx)
            except Exception as e:  # an expert must never crash the pipeline
                ev = Evidence(expert=name, category=category, applicable=False, error=f"{type(e).__name__}: {e}")
            ev.runtime_ms = (time.perf_counter() - t) * 1000
            return ev

        REGISTRY[name] = ExpertSpec(name, category, wrapped, cost, description or (fn.__doc__ or "").strip())
        return wrapped

    return deco
