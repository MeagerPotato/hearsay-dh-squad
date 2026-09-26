"""Speaker-embedding forensics: is the same voice speaking from the first second to the last?

Technique
---------
An ECAPA-TDNN speaker-verification network (SpeechBrain, trained on VoxCeleb) turns any stretch of
speech into a 192-dimensional "voiceprint". Two within-clip tests:

* sliding windows - 1.5 s windows (hop 0.5 s) are compared with the clip's average voice (cosine
  similarity to the centroid); a partial voice conversion or an unstable vocoder makes some windows
  stray from the rest;
* split scan - pauses and room-noise lead-ins/tails are removed (webrtcvad + energy gate) and, for every
  candidate cut point, the voiceprint of the speech before the cut is compared with the speech after it.
  A splice of two talkers gives a cut where the two sides disagree strongly, wherever the splice is.

The whole-clip voiceprint is cached in ``ctx.cache["spk_emb"]`` so a batch-level step can group files
by voice with :func:`cluster_speakers` and score how *typical* each clip is of its speaker cluster.

What we measured (see the module report): the split scan separates synthetic two-talker splices from
genuine clips almost perfectly (AUC ~0.98) and flags none of the low-AASIST test clips or LJSpeech clips
at the chosen threshold. Within-clip drift does NOT separate whole-utterance TTS from real speech on the
HEARSAY test set; out of domain, TTS voices are, if anything, *more* uniform than real speakers, so a stable
identity is reported as neutral, not as evidence of authenticity. The raw embedding magnitude
(``speaker_consistency_emb_norm``) is lower for AASIST-flagged test clips (d ~ -0.9) but not out of domain.

Caveats: 1.5 s windows give noisy voiceprints (within-clip similarities of 0.75-0.9 are normal for
genuine speech), and minima get lower as clips get longer, so thresholds are deliberately
conservative -- false alarms on real audio are expensive.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..evidence import Context, Evidence, Finding, expert

NAME = "speaker_consistency"
CATEGORY = "speaker"

ROOT = Path(__file__).resolve().parents[2]
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = Path(os.environ.get("HEARSAY_ECAPA_DIR", ROOT / "work" / "models" / "spkrec-ecapa-voxceleb"))
MODEL_FILES = ("hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt",
               "classifier.ckpt", "label_encoder.txt")
MODEL_SR = 16000

# Windowing
WIN_S = 1.5            # voiceprint window length
HOP_S = 0.5            # window hop (scaled down proportionally for short clips)
MIN_WIN_S = 0.8        # shortest window ECAPA is still meaningful on
MAX_WINDOWS = 60       # long user files: hop grows so we never embed more than this many windows
MAX_DIRECT_S = 30.0    # whole-clip voiceprint computed in one pass up to this length
MIN_SPEECH_FRAC = 0.5   # windows with less active speech than this are shown but not scored
BATCH = 4              # segments per forward pass. Larger CPU batches hit a pathological slow path
                       # in torch conv1d (16 x 3.5 s took 150 s and 1.5 GB vs 1 s at batch 4)

# Decision thresholds on cosine similarity, calibrated on 110 HEARSAY test clips (incl. the 30 longest),
# 20 LJSpeech + 20 DiffSSD clips (organizer pipeline) and 20 synthetic cross-speaker splices: at these values
# 1/40 low-AASIST test clips (borderline, 0.14), 0/20 LJSpeech and 3/30 long test clips are flagged, while
# 16/20 splices are. Borderline values give weak findings (false alarms on real audio cost 4x).
STABLE_MIN_SIM = 0.80   # every window this close to the centroid -> identity looks stable
DRIFT_MIN_SIM = 0.45    # some window this far from the centroid -> identity drift
SPLIT_MIN_SIM = 0.15    # speech before vs after some cut point this dissimilar -> two different voices

# Split scan ("before t" vs "after t")
MIN_SIDE_S = 0.75       # each side of a candidate cut needs at least this much active speech
MAX_SIDE_S = 3.0        # ... and uses at most this much (local comparison; bounds cost on long files)
SPLIT_STEP_S = 0.3      # candidate cut spacing
MAX_SPLITS = 16         # cap on candidate cuts for long files

# Per-clip feature columns, always present (NaN when not measurable) so fusion sees a fixed set.
# speaker_consistency_typicality / _cluster_size are added only by the batch-level cluster_speakers() pass.
CLIP_FEATURES = ("emb_norm", "min_sim", "mean_sim", "sim_std", "adjacent_min_sim", "max_adjacent_drop",
                 "split_min_sim")

_MODEL = None
_MODEL_ERR: Optional[str] = None
_LOCK = threading.Lock()


# ----------------------------------------------------------------------------------------------
# Model (lazy module-level singleton)
# ----------------------------------------------------------------------------------------------
def _get_model():
    """Load the ECAPA encoder once per process (CPU, one thread). Downloads ~90 MB on first use only."""
    global _MODEL, _MODEL_ERR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_ERR is not None:  # do not retry a failed load (e.g. offline) for every file
        raise RuntimeError(_MODEL_ERR)
    with _LOCK:
        if _MODEL is None:
            try:
                import torch
                torch.set_num_threads(1)
                from speechbrain.inference.speaker import EncoderClassifier
                from speechbrain.utils.fetching import LocalStrategy

                if not all((MODEL_DIR / f).exists() for f in MODEL_FILES):
                    from huggingface_hub import snapshot_download
                    snapshot_download(MODEL_SOURCE, local_dir=str(MODEL_DIR), allow_patterns=list(MODEL_FILES))
                model = EncoderClassifier.from_hparams(
                    source=str(MODEL_DIR), savedir=str(MODEL_DIR), run_opts={"device": "cpu"},
                    overrides={"pretrained_path": str(MODEL_DIR)},  # checkpoints from MODEL_DIR, not the Hub
                    local_strategy=LocalStrategy.NO_LINK)
                model.eval()
                _MODEL = model
            except Exception as e:
                _MODEL_ERR = f"ECAPA speaker model unavailable ({type(e).__name__}: {e})"
                raise RuntimeError(_MODEL_ERR) from e
    return _MODEL


def embed_segments(segments: List[np.ndarray]) -> np.ndarray:
    """Raw (un-normalised) ECAPA voiceprints for 16 kHz float32 segments, shape (n, 192).

    Segments of unequal length are zero-padded and passed with relative lengths, so the padding is
    ignored by the model's statistics pooling.
    """
    import torch

    model = _get_model()
    order = np.argsort([len(s) for s in segments], kind="stable")  # batch similar lengths -> little padding
    out = np.zeros((len(segments), 0), np.float32)
    with torch.inference_mode():
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            chunk = [segments[k] for k in idx]
            n = max(len(s) for s in chunk)
            x = np.zeros((len(chunk), n), np.float32)
            for j, s in enumerate(chunk):
                x[j, :len(s)] = s
            lens = torch.tensor([len(s) / n for s in chunk], dtype=torch.float32)
            e = model.encode_batch(torch.from_numpy(x), wav_lens=lens).reshape(len(chunk), -1).cpu().numpy()
            if out.shape[1] == 0:
                out = np.zeros((len(segments), e.shape[1]), np.float32)
            out[idx] = e
    return out


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _analysis_audio(ctx: Context) -> np.ndarray:
    """Mono float32 at 16 kHz with NaN/inf removed, whatever the context carries."""
    x = np.nan_to_num(np.asarray(ctx.audio, dtype=np.float32).reshape(-1))
    if ctx.sr != MODEL_SR and len(x):
        import librosa
        x = librosa.resample(x, orig_sr=ctx.sr, target_sr=MODEL_SR).astype(np.float32)
    return x


def _speech_frames(x: np.ndarray, sr: int, frame_s: float = 0.025, hop_s: float = 0.010) -> Tuple[np.ndarray, int, int]:
    """Speech mask on a 10 ms grid: webrtcvad (aggressiveness 3) AND an energy gate.

    The energy gate (within 30 dB of the clip's loud level and 10 dB above its noise floor) removes the
    long low-level room-noise lead-ins/tails of VCTK-style recordings, which webrtcvad alone often calls
    speech and which otherwise look like "a different voice". Energy uses a cumulative sum of squares so
    memory stays O(n) for long files. Falls back to the energy gate alone if webrtcvad is unavailable.
    """
    fl, hl = int(frame_s * sr), int(hop_s * sr)
    if len(x) < fl:
        return np.zeros(0, bool), fl, hl
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(0, len(x) - fl + 1, hl)
    db = 10 * np.log10((c[starts + fl] - c[starts]) / fl + 1e-12)
    loud, floor = np.percentile(db, 95), np.percentile(db, 10)
    mask = db > max(loud - 30.0, floor + 10.0, -70.0)
    if sr == 16000:
        try:
            import webrtcvad
            vad = webrtcvad.Vad(3)
            pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
            n = min(len(starts), len(pcm) // hl)
            v = np.fromiter((vad.is_speech(pcm[i * hl:(i + 1) * hl].tobytes(), sr) for i in range(n)), bool, n)
            both = mask.copy()
            both[:n] &= v
            both[n:] = False
            # webrtcvad rejects some heavily clipped/odd-but-loud speech; keep the energy gate alone then
            # (stationary noise never passes the energy gate, so this does not let noise through)
            if both.sum() * hl >= 0.5 * sr or mask.sum() * hl < 1.0 * sr:
                mask = both
        except Exception:
            pass
    return _clean_runs(mask), fl, hl


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[start, end) index pairs of the True runs of a boolean array."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _clean_runs(mask: np.ndarray, bridge: int = 10, min_run: int = 20) -> np.ndarray:
    """Bridge gaps <= 100 ms inside words, then drop 'speech' runs < 200 ms (clicks, knocks, breaths).

    Isolated noise bursts in long pauses otherwise survive the VAD and get compared as if they were a voice.
    """
    m = mask.copy()
    for a, b in _runs(~m):
        if a > 0 and b < len(m) and b - a <= bridge:
            m[a:b] = True
    for a, b in _runs(m):
        if b - a < min_run:
            m[a:b] = False
    return m


def _window_plan(n: int, sr: int) -> Tuple[int, List[int]]:
    """Window length (samples) and start offsets; empty when the clip is too short for >= 2 windows."""
    win_s = min(WIN_S, n / sr / 2)
    if win_s < MIN_WIN_S:
        return 0, []
    w = int(round(win_s * sr))
    hop = max(int(round(HOP_S * sr * win_s / WIN_S)), int(np.ceil((n - w) / (MAX_WINDOWS - 1))))
    starts = list(range(0, n - w + 1, hop))
    if n - (starts[-1] + w) > hop // 2:  # make sure the tail of the clip is covered too
        starts.append(n - w)
    return w, starts


def _speech_fraction(mask: np.ndarray, fl: int, hl: int, a: int, b: int) -> float:
    """Fraction of speech frames whose analysis frame lies inside samples [a, b)."""
    if mask.size == 0:
        return 0.0
    i0, i1 = int(np.ceil(a / hl)), int((b - fl) // hl) + 1
    seg = mask[max(i0, 0):min(i1, mask.size)]
    return float(seg.mean()) if seg.size else 0.0


def _speech_only(x: np.ndarray, mask: np.ndarray, fl: int, hl: int, hang: int = 3) -> np.ndarray:
    """Sample indices of active speech (frame mask dilated by `hang` frames = 30 ms on each side)."""
    if mask.size == 0:
        return np.arange(len(x))
    m = np.convolve(mask.astype(np.float32), np.ones(2 * hang + 1, np.float32), mode="same") > 0
    keep = np.zeros(len(x), bool)
    for i in np.flatnonzero(m):
        keep[i * hl:i * hl + fl] = True
    return np.flatnonzero(keep)


def _split_scan(x: np.ndarray, sr: int, mask: np.ndarray, fl: int, hl: int):
    """Compare the voiceprint of the speech before each candidate cut point with the speech after it.

    A splice of two talkers shows up as a cut point where the two sides disagree strongly, wherever
    the splice is. Pauses and trailing breath/noise are removed first (speech-only signal), otherwise a
    side that is mostly silence looks like "a different voice". Each side uses at most MAX_SIDE_S of speech
    next to the cut. Returns (cut time in the original clip, min similarity, [(t, sim, shorter side s), ...])
    or (None, None, []) when there is not enough speech for two sides of MIN_SIDE_S seconds each.
    """
    keep = _speech_only(x, mask, fl, hl)
    xs = x[keep]
    side, ctx_len = int(MIN_SIDE_S * sr), int(MAX_SIDE_S * sr)
    if len(xs) < 2 * side:
        return None, None, []
    step = max(int(SPLIT_STEP_S * sr), int(np.ceil((len(xs) - 2 * side) / (MAX_SPLITS - 1))))
    cuts = list(range(side, len(xs) - side + 1, step))
    segs = [s for c in cuts for s in (xs[max(0, c - ctx_len):c], xs[c:c + ctx_len])]
    E = _unit(embed_segments(segs))
    sims = np.sum(E[0::2] * E[1::2], axis=1)
    times = [float(keep[c]) / sr for c in cuts]  # map speech-only offsets back to clip time
    shorter = [min(c, len(xs) - c, ctx_len) / sr for c in cuts]  # speech (s) on the thinner side of each cut
    i = int(np.argmin(sims))
    curve = [(round(t, 3), round(float(v), 4), round(m, 2)) for t, v, m in zip(times, sims, shorter)]
    return times[i], float(sims[i]), curve


# ----------------------------------------------------------------------------------------------
# Expert
# ----------------------------------------------------------------------------------------------
@expert(NAME, CATEGORY, cost="medium")
def speaker_consistency(ctx: Context) -> Evidence:
    """Speaker-embedding consistency: ECAPA-TDNN voiceprints on sliding 1.5 s windows plus a before/after
    split scan of the speech; flags a voice identity that changes mid-clip (splice of two talkers, partial
    voice conversion) and caches the clip voiceprint for batch-level speaker clustering."""
    ev = Evidence(expert=NAME, category=CATEGORY, features={f"{NAME}_{k}": float("nan") for k in CLIP_FEATURES})
    sr = MODEL_SR
    x = _analysis_audio(ctx)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    mask, fl, hl = _speech_frames(x, sr)
    speech_s = float(mask.sum()) * hl / sr if mask.size else 0.0
    if len(x) < MIN_WIN_S * sr or peak < 1e-4 or speech_s < 0.5:
        ev.applicable = False
        ev.findings.append(Finding(f"Too little speech ({speech_s:.1f} s) to extract a reliable voiceprint; "
                                   "speaker-consistency analysis skipped.", 0, 0.0))
        return ev
    x = x / peak  # level does not matter to ECAPA (sentence mean normalisation), but keep inputs sane

    # --- whole-clip voiceprint (cached for batch clustering and other experts) ---------------
    w, starts = _window_plan(len(x), sr)
    win_emb = None
    if w:
        win_emb = embed_segments([x[s:s + w] for s in starts])
    if "spk_emb" in ctx.cache and "spk_emb_norm" in ctx.cache:
        clip_emb, emb_norm = np.asarray(ctx.cache["spk_emb"], np.float32), float(ctx.cache["spk_emb_norm"])
    else:
        if len(x) <= MAX_DIRECT_S * sr or win_emb is None:
            raw = embed_segments([x[:int(MAX_DIRECT_S * sr)]])[0]
        else:  # very long file: average of window voiceprints
            raw = win_emb.mean(axis=0)
        emb_norm = float(np.linalg.norm(raw))
        clip_emb = _unit(raw).astype(np.float32)
        ctx.cache["spk_emb"], ctx.cache["spk_emb_norm"] = clip_emb, emb_norm
    ev.artifacts["spk_emb"] = clip_emb
    ev.features[f"{NAME}_emb_norm"] = emb_norm  # raw embedding magnitude ~ how "speaker-like" ECAPA finds it

    if win_emb is None or len(starts) < 2:
        ev.findings.append(Finding(f"Clip too short ({len(x) / sr:.1f} s) for a within-clip drift test; "
                                   "only the whole-clip voiceprint was extracted.", 0, 0.0))
        _add_typicality(ctx, ev, clip_emb)
        return ev

    # --- sliding-window drift ---------------------------------------------------------------
    E = _unit(win_emb)
    frac = np.array([_speech_fraction(mask, fl, hl, s, s + w) for s in starts])
    ok = frac >= MIN_SPEECH_FRAC
    if ok.sum() < 2:  # mostly pauses: score every window rather than none
        ok[:] = True
    centroid = _unit(E[ok].mean(axis=0))
    sim = E @ centroid
    t = (np.asarray(starts) + w / 2) / sr
    adj = np.sum(E[:-1] * E[1:], axis=1)            # voiceprint agreement of neighbouring windows
    both = ok[:-1] & ok[1:]
    jumps = np.abs(np.diff(sim))[both] if both.any() else np.abs(np.diff(sim))

    s_ok = sim[ok]
    i_min = int(np.flatnonzero(ok)[np.argmin(s_ok)])
    min_sim, mean_sim = float(s_ok.min()), float(s_ok.mean())

    # --- split scan: "before t" vs "after t" voiceprints (non-overlapping) ---------------------
    split_t, split_sim, split_curve = _split_scan(x, sr, mask, fl, hl)

    f = ev.features
    f[f"{NAME}_min_sim"] = min_sim
    f[f"{NAME}_mean_sim"] = mean_sim
    f[f"{NAME}_sim_std"] = float(s_ok.std())
    f[f"{NAME}_adjacent_min_sim"] = float(adj[both].min() if both.any() else adj.min())
    f[f"{NAME}_max_adjacent_drop"] = float(jumps.max()) if jumps.size else 0.0
    if split_sim is not None:
        f[f"{NAME}_split_min_sim"] = split_sim

    ev.artifacts["speaker_drift"] = {
        "t": t.round(3).tolist(), "sim": sim.round(4).tolist(), "adjacent_sim": adj.round(4).tolist(),
        "speech_frac": frac.round(3).tolist(), "scored": ok.tolist(), "window_s": w / sr,
        "split_t": [c[0] for c in split_curve], "split_sim": [c[1] for c in split_curve],
        "split_side_s": [c[2] for c in split_curve],
        "drift_threshold": DRIFT_MIN_SIM, "split_threshold": SPLIT_MIN_SIM,
    }

    # --- findings ---------------------------------------------------------------------------
    t_min = float(t[i_min])
    drift_win = min_sim < DRIFT_MIN_SIM
    drift_split = split_sim is not None and split_sim < SPLIT_MIN_SIM
    if drift_split or drift_win:
        # 0.15 just below a threshold, ~0.75 for clear two-voice splices (similarity <= -0.05)
        severity = max((SPLIT_MIN_SIM - split_sim) / 0.2 if drift_split else 0.0,
                       (DRIFT_MIN_SIM - min_sim) / 0.2 if drift_win else 0.0)
        parts = []
        if drift_split:
            side = split_curve[int(np.argmin([c[1] for c in split_curve]))][2]
            parts.append(f"the voice before and after {split_t:.1f} s matches with only {split_sim:.2f} cosine "
                         f"similarity (shorter side {side:.1f} s of speech), where genuine single-talker clips "
                         f"score above {SPLIT_MIN_SIM:.2f}")
        if drift_win:
            parts.append(f"window similarity to the clip's average voice drops to {min_sim:.2f} at {t_min:.1f} s")
        ev.findings.append(Finding(
            "Voice identity drifts mid-clip (" + "; ".join(parts) + ") - possible splice of two recordings, "
            "partial voice conversion or an unstable synthesiser. A genuine second speaker (interview, call), "
            "coughs, laughs or noise bursts cause this too, especially when one side is short.",
            +1, float(np.clip(0.15 + 0.6 * severity, 0.15, 0.75))))
    elif min_sim >= STABLE_MIN_SIM:
        ev.findings.append(Finding(
            f"Voice identity stable across the clip (min window similarity {min_sim:.2f}"
            + (f", before/after split {split_sim:.2f}" if split_sim is not None else "")
            + "). This rules out a crude splice of two talkers, but it is not evidence of authenticity: "
              "text-to-speech voices are, if anything, more uniform than real speakers.", 0, 0.0))
    else:
        ev.findings.append(Finding(
            f"Voice identity broadly consistent (min window similarity {min_sim:.2f} at {t_min:.1f} s, mean "
            f"{mean_sim:.2f}); the variation is within what short windows of natural speech show.", 0, 0.0))

    _add_typicality(ctx, ev, clip_emb)
    return ev


def _add_typicality(ctx: Context, ev: Evidence, emb: np.ndarray) -> None:
    """If a batch step supplied speaker clusters (see cluster_speakers), report how typical this voice is."""
    base = os.path.basename(ctx.path)
    typ = (ctx.corpus.get("speaker_typicality") or {}).get(base)
    size = (ctx.corpus.get("speaker_cluster_size") or {}).get(base)
    if typ is None and ctx.corpus.get("speaker_centroids") is not None:
        C = _unit(np.asarray(ctx.corpus["speaker_centroids"], np.float32))
        if C.ndim == 2 and C.shape[1] == emb.shape[0] and len(C):
            typ = float(np.max(C @ emb))
    if typ is None or not np.isfinite(typ):
        return
    ev.features[f"{NAME}_typicality"] = float(typ)
    if size is not None:
        ev.features[f"{NAME}_cluster_size"] = float(size)
        if int(size) > 0:
            text = (f"The same voice occurs in {int(size)} other clip(s) of this batch (cosine {typ:.2f} to their "
                    "average voiceprint). Clones of a target speaker often sit slightly off the genuine cluster, "
                    "so a low value here is worth a closer look.")
        else:
            text = (f"No other clip in this batch shares this voice (nearest other voice: cosine {typ:.2f}).")
        ev.findings.append(Finding(text, 0, 0.0))


# ----------------------------------------------------------------------------------------------
# Batch-level helper
# ----------------------------------------------------------------------------------------------
def cluster_speakers(embeddings: Dict[str, np.ndarray], threshold: float = 0.45) -> Dict[str, dict]:
    """Group whole-clip voiceprints into speakers and score each clip's speaker typicality.

    embeddings: {basename: 192-d voiceprint (ctx.cache["spk_emb"])}.
    threshold:  minimum average cosine similarity for two groups to be merged (average-linkage
                agglomerative clustering). ECAPA same-speaker pairs on 3-4 s clips are typically
                0.5-0.8, different speakers 0.0-0.3.

    Returns a dict ready to merge into ctx.corpus:
      speaker_cluster       {basename: cluster id}
      speaker_cluster_size  {basename: number of OTHER clips in the same cluster}
      speaker_typicality    {basename: cosine to the nearest cluster centroid, computed leave-one-out so a
                             clip never matches itself; a singleton is scored against the other clusters}
      speaker_centroids     (k, 192) array of unit centroids (for scoring new files)
    """
    names = [k for k, v in embeddings.items() if v is not None and np.all(np.isfinite(v))]
    if not names:
        return {"speaker_cluster": {}, "speaker_cluster_size": {}, "speaker_typicality": {},
                "speaker_centroids": np.zeros((0, 192), np.float32)}
    X = _unit(np.stack([np.asarray(embeddings[k], np.float32).reshape(-1) for k in names]))
    if len(names) == 1:
        labels = np.zeros(1, int)
    else:
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                         distance_threshold=1.0 - threshold).fit_predict(X)
    k = int(labels.max()) + 1
    sums = np.stack([X[labels == c].sum(axis=0) for c in range(k)])
    counts = np.bincount(labels, minlength=k)
    centroids = _unit(sums)

    typ, size = {}, {}
    for i, name in enumerate(names):
        c = labels[i]
        loo = sums.copy()
        loo[c] -= X[i]                      # remove this clip from its own cluster
        valid = counts - (np.arange(k) == c) > 0
        typ[name] = float(np.max(_unit(loo[valid]) @ X[i])) if valid.any() else float("nan")
        size[name] = int(counts[c] - 1)
    return {"speaker_cluster": {n: int(l) for n, l in zip(names, labels)}, "speaker_cluster_size": size,
            "speaker_typicality": typ, "speaker_centroids": centroids.astype(np.float32)}
