"""Semantic forensics: what is said, how it reads, and whether the same script recurs across a batch.

Two experts:

``transcript`` (expensive)
    Speech-to-text with Whisper small.en (cached transcripts are used when available). From the text we
    measure speaking rate, spontaneous-speech markers (fillers such as "um"/"uh", back-to-back word
    repeats), mean word length, and whether the recogniser got stuck in a loop ("S-S-S-S-...") - which
    happens when the audio contains garbled phonemes or vocoder buzz rather than clean speech. When the
    model actually runs we also report its confidence (mean token log-probability).

``repeated_script`` (cheap, batch-level)
    Uses statistics over the transcripts of the whole batch (see :func:`build_corpus_stats`): how many
    OTHER clips speak the same sentence (fuzzy match, tolerant to recognition errors), how far this
    clip's transcript deviates from the other readings of that sentence (mis-heard words = degraded
    articulation), and whether the text looks like a complete read sentence or a fragment. Synthetic
    campaigns and anti-spoofing corpora re-use one script across many voices/systems, so a sentence
    that recurs many times in a batch is a (weak, batch-dependent) hallmark of mass production.
"""
from __future__ import annotations

import json
import os
import re
import threading
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..evidence import Context, Evidence, Finding, expert

CATEGORY = "semantic"
ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS_JSON = Path(os.environ.get("HEARSAY_TRANSCRIPTS", ROOT / "work" / "test_transcripts.json"))
WHISPER_ID = "openai/whisper-small.en"
ASR_SR = 16000
ASR_CHUNK_S = 30.0       # Whisper's context window
ASR_MAX_S = 300.0        # never transcribe more than 5 minutes of a user file

FILLERS = {"uh", "um", "umm", "uhm", "er", "erm", "ah", "eh", "hmm", "hm", "mm", "mhm"}
CONVERSATIONAL = ("you know", "i mean", "kind of like", "yeah", "okay so", "gonna", "wanna")
# Short fragments of the standard VCTK elicitation passages (the "Please call Stella" accent paragraph and
# the Rainbow Passage), used only to recognise scripted read speech.
ELICITATION_CUES = ("please call stella", "things with her from the store", "snow peas", "blue cheese",
                    "three red bags", "when the sunlight strikes", "division of white light", "pot of gold",
                    "boiling pot of gold", "ideas about the rainbow")

REPEAT_SIM = 0.72        # char 3-4-gram TF-IDF cosine above which two transcripts are "the same script"
REPEAT_MIN_WORDS = 4     # very short phrases ("thank you") are not scripts
LOOP_COMPRESSION = 2.4   # Whisper's own hallucination threshold on gzip compression ratio
LOOP_RUN = 6             # or this many identical tokens in a row
DEVIATION_FLAG = 0.15    # repeat_deviation above which a reading counts as mis-heard (max possible ~0.28)


# ----------------------------------------------------------------------------------------------
# Text utilities
# ----------------------------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Lower-case, strip punctuation (keep apostrophes), collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower()).split())


def compression_ratio(text: str) -> float:
    """gzip compression ratio of the transcript; > 2.4 flags Whisper repetition loops."""
    b = (text or "").encode("utf-8")
    return len(b) / len(zlib.compress(b)) if b else 0.0


def longest_run(tokens: List[str]) -> int:
    """Length of the longest run of identical consecutive tokens."""
    best = run = 1 if tokens else 0
    for a, b in zip(tokens, tokens[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def is_asr_loop(text: str) -> bool:
    """Whisper got stuck repeating itself ("S-S-S-S...", "a-l-l-l-l...", "the the the the ...")."""
    toks = re.findall(r"[a-z0-9']+", (text or "").lower())
    return compression_ratio(text) > LOOP_COMPRESSION or longest_run(toks) >= LOOP_RUN


def text_stats(text: str) -> Dict[str, float]:
    """Per-transcript lexical statistics (no audio needed)."""
    words = normalize_text(text).split()
    fillers = sum(w in FILLERS for w in words)
    repeats = sum(a == b and a not in FILLERS for a, b in zip(words, words[1:]))
    content = [w for w in words if w not in FILLERS]
    return {
        "words": float(len(words)),
        "filler_count": float(fillers),
        "repetition_count": float(repeats),
        "mean_word_len": float(np.mean([len(w) for w in content])) if content else 0.0,
        "compression_ratio": float(compression_ratio(text)),
    }


def read_style(text: str) -> Tuple[float, List[str]]:
    """How much the transcript looks like one complete, scripted read sentence (VCTK / newspaper style).

    Returns (score 0..1, list of criteria that failed). Criteria: starts with a capital letter (Whisper
    lower-cases a first word it hears as mid-sentence), ends with terminal punctuation, 4-40 words, no
    fillers or conversational markers, no recogniser loop. Known elicitation passages count as read speech.
    """
    raw = (text or "").strip()
    norm = normalize_text(raw)
    words = norm.split()
    if not words:
        return 0.0, ["no words recognised"]
    if any(c in norm for c in ELICITATION_CUES) and not is_asr_loop(raw):
        return 1.0, []
    failed = []
    core = raw.strip("\"'“”‘’()[] ")  # ignore quotes/brackets around the sentence
    if core[:1].isalpha() and core[:1].islower():
        failed.append("starts mid-sentence")
    if core[-1:] not in (".", "!", "?"):
        failed.append("no sentence ending")
    if not 4 <= len(words) <= 40:
        failed.append(f"{len(words)} words")
    padded = f" {norm} "
    if any(w in FILLERS for w in words) or any(f" {c} " in padded for c in CONVERSATIONAL):
        failed.append("spontaneous-speech markers")
    if is_asr_loop(raw):
        failed.append("recogniser loop")
    return 1.0 - len(failed) / 5.0, failed


# ----------------------------------------------------------------------------------------------
# Cached transcripts and Whisper (lazy module-level singletons)
# ----------------------------------------------------------------------------------------------
_CACHED: Optional[Dict[str, str]] = None
_CACHED_CORPUS: Optional[dict] = None
_ASR = None
_ASR_ERR: Optional[str] = None
_LOCK = threading.Lock()


def load_cached_transcripts() -> Dict[str, str]:
    """{basename: transcript} precomputed with Whisper small.en for the HEARSAY test set ({} if absent)."""
    global _CACHED
    if _CACHED is None:
        try:
            with open(TRANSCRIPTS_JSON) as fh:
                _CACHED = {str(k): str(v) for k, v in json.load(fh).items()}
        except Exception:
            _CACHED = {}
    return _CACHED


def _get_asr():
    """Whisper small.en processor + model, loaded once (MPS when available, else CPU)."""
    global _ASR, _ASR_ERR
    if _ASR is not None:
        return _ASR
    if _ASR_ERR is not None:
        raise RuntimeError(_ASR_ERR)
    with _LOCK:
        if _ASR is None:
            try:
                import torch
                from transformers import WhisperForConditionalGeneration, WhisperProcessor
                try:  # prefer the local HF cache, no network round-trip
                    from huggingface_hub import snapshot_download
                    path = snapshot_download(WHISPER_ID, local_files_only=True)
                except Exception:
                    path = WHISPER_ID
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                if device == "cpu":
                    torch.set_num_threads(1)
                proc = WhisperProcessor.from_pretrained(path)
                model = WhisperForConditionalGeneration.from_pretrained(path).to(device).eval()
                _ASR = (proc, model, device)
            except Exception as e:
                _ASR_ERR = f"Whisper unavailable ({type(e).__name__}: {e})"
                raise RuntimeError(_ASR_ERR) from e
    return _ASR


def _decoder_prompt(proc, model) -> List[int]:
    """Whisper's forced decoder prefix for English-only models: <|startoftranscript|><|notimestamps|>."""
    sot = model.generation_config.decoder_start_token_id
    nots = proc.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    return [int(sot)] + ([int(nots)] if isinstance(nots, int) and nots >= 0 else [])


def transcribe(audio: np.ndarray, sr: int) -> Tuple[str, Optional[float]]:
    """Transcribe mono audio with Whisper small.en in 30 s chunks.

    Returns (text, mean log-probability of the emitted text tokens, or None). The confidence comes from a
    teacher-forced re-scoring pass of the decoded tokens, which is stable across transformers versions
    (generate(..., output_scores=True) breaks on some of them).
    """
    import torch

    proc, model, device = _get_asr()
    x = np.nan_to_num(np.asarray(audio, np.float32).reshape(-1))
    if sr != ASR_SR and len(x):
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=ASR_SR).astype(np.float32)
    x = x[:int(ASR_MAX_S * ASR_SR)]
    special = set(proc.tokenizer.all_special_ids)
    prompt = _decoder_prompt(proc, model)
    texts, logps = [], []
    step = int(ASR_CHUNK_S * ASR_SR)
    for i in range(0, max(len(x), 1), step):
        chunk = x[i:i + step]
        if len(chunk) < 0.2 * ASR_SR or np.max(np.abs(chunk)) < 1e-4:
            continue
        feats = proc(chunk, sampling_rate=ASR_SR, return_tensors="pt").input_features.to(device)
        with torch.inference_mode():
            seq = model.generate(feats, max_new_tokens=220)
            seq = seq if torch.is_tensor(seq) else seq.sequences
            toks = [t for t in seq[0].tolist() if t not in special]   # text tokens only
            texts.append(proc.tokenizer.decode(toks, skip_special_tokens=True).strip())
            if toks:
                dec = torch.tensor([prompt + toks], device=device)
                logits = model(input_features=feats, decoder_input_ids=dec).logits[0].float()
                lp = torch.log_softmax(logits[len(prompt) - 1:-1], dim=-1)
                logps.extend(lp.gather(-1, dec[0, len(prompt):, None])[:, 0].cpu().tolist())
    text = " ".join(t for t in texts if t)
    return text, (float(np.mean(logps)) if logps else None)


def _speech_span_s(x: np.ndarray, sr: int) -> float:
    """Seconds from the first to the last energetic 10 ms frame (leading/trailing silence excluded)."""
    fl, hl = int(0.025 * sr), int(0.010 * sr)
    if len(x) < fl:
        return 0.0
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, np.float64) ** 2)])
    starts = np.arange(0, len(x) - fl + 1, hl)
    db = 10 * np.log10((c[starts + fl] - c[starts]) / fl + 1e-12)
    idx = np.flatnonzero(db > max(np.percentile(db, 95) - 35.0, np.percentile(db, 10) + 6.0, -70.0))
    return float((idx[-1] - idx[0]) * hl + fl) / sr if idx.size else 0.0


# ----------------------------------------------------------------------------------------------
# Expert 1: transcript
# ----------------------------------------------------------------------------------------------
@expert("transcript", CATEGORY, cost="expensive")
def transcript(ctx: Context) -> Evidence:
    """Whisper small.en transcript (cached for known files): speaking rate, fillers/disfluencies, word
    length, recogniser loops and ASR confidence - read vs spontaneous speech and garbled articulation."""
    ev = Evidence(expert="transcript", category=CATEGORY)
    base = os.path.basename(ctx.path)
    logprob = None
    if isinstance(ctx.cache.get("transcript"), str):
        text, source = ctx.cache["transcript"], ctx.cache.get("transcript_source", "context")
        logprob = ctx.cache.get("asr_logprob")
    elif base in load_cached_transcripts():
        text, source = load_cached_transcripts()[base], "cached Whisper small.en"
    else:
        text, logprob = transcribe(ctx.audio, ctx.sr)
        source = "Whisper small.en"
    ctx.cache["transcript"], ctx.cache["transcript_source"] = text, source
    if logprob is not None:
        ctx.cache["asr_logprob"] = logprob

    st = text_stats(text)
    span = _speech_span_s(np.nan_to_num(np.asarray(ctx.audio, np.float32)), ctx.sr)
    f = ev.features       # fixed column set; NaN = not measurable (no speech span / Whisper did not run)
    f["semantic_words"] = st["words"]
    f["semantic_words_per_second"] = st["words"] / span if span >= 0.3 else float("nan")
    f["semantic_filler_count"] = st["filler_count"]
    f["semantic_repetition_count"] = st["repetition_count"]
    f["semantic_mean_word_len"] = st["mean_word_len"]
    f["semantic_compression_ratio"] = st["compression_ratio"]
    f["semantic_asr_logprob"] = float(logprob) if logprob is not None else float("nan")
    ev.artifacts.update({"transcript": text, "transcript_source": source, "speech_span_s": span})

    # --- findings ---------------------------------------------------------------------------
    shown = text if len(text) <= 160 else text[:157] + "..."
    if st["words"] == 0:
        ev.findings.append(Finding(f"No intelligible English words recognised ({source}). Either not speech, "
                                   "not English, or too degraded to understand.", 0, 0.0))
        return ev
    rate = f["semantic_words_per_second"] if np.isfinite(f["semantic_words_per_second"]) else None
    ev.findings.append(Finding(
        f'Transcript ({source}): "{shown}" - {int(st["words"])} words'
        + (f" in {span:.1f} s of speech ({rate:.1f} words/s; read speech is typically 2-3.5)" if rate else "")
        + ".", 0, 0.0))
    if is_asr_loop(text):
        ev.findings.append(Finding(
            f"The speech recogniser got stuck in a loop (compression ratio {st['compression_ratio']:.1f}, "
            "repeating the same fragment). This happens when the audio contains garbled phonemes or buzzy "
            "vocoder artefacts instead of clean speech; on the HEARSAY test set both clips where this happened "
            "(2 of 1671) were scored strongly synthetic by the AASIST detector.", +1, 0.5))
    if st["filler_count"] > 0:
        ev.findings.append(Finding(
            f"Contains {int(st['filler_count'])} spontaneous-speech filler(s) (um/uh/er). Unscripted human speech "
            "is full of these while most text-to-speech reads a clean script, so this weakly favours a real "
            "speaker (some modern TTS inserts fillers deliberately, and Whisper tends to drop them).", -1, 0.15))
    if logprob is not None and logprob < -0.8:
        ev.findings.append(Finding(
            f"Speech recogniser is unsure of the words (mean token log-probability {logprob:.2f}; clear read "
            "speech is usually above -0.5). Mumbled or garbled articulation is a mild synthesis cue, but noise, "
            "accents and non-English speech cause it too.", +1, 0.2))
    return ev


# ----------------------------------------------------------------------------------------------
# Batch-level corpus statistics
# ----------------------------------------------------------------------------------------------
def build_corpus_stats(transcripts: Dict[str, str], threshold: float = REPEAT_SIM) -> dict:
    """Batch statistics over {basename: transcript}; merge the result into ctx.corpus.

    For each file:
      repeat_count[bn]      number of OTHER files whose transcript is the same sentence (character 3-4-gram
                            TF-IDF cosine >= threshold, so recognition slips like "analyst"/"analysts" or
                            "golf"/"goth" still match; transcripts under 4 words never match)
      repeat_deviation[bn]  1 - mean similarity to those matches (only when repeat_count >= 1): how far
                            this reading was mis-heard relative to the other readings of the script
      repeat_group[bn]      id of the connected group of matching transcripts (-1 when unique)
      repeat_group_text[g]  the group's most central transcript (what the script "should" say)
      read_style[bn]        0..1 match to complete, scripted read-sentence style (see read_style())
      read_style_flags[bn]  which read-style criteria failed
    plus corpus_size (number of transcripts).
    """
    names = list(transcripts)
    texts = [transcripts[n] or "" for n in names]
    norm = [normalize_text(t) for t in texts]
    n = len(names)
    out = {"corpus_size": n, "repeat_count": {}, "repeat_deviation": {}, "repeat_group": {},
           "repeat_group_text": {}, "read_style": {}, "read_style_flags": {}}
    for name, t in zip(names, texts):
        score, failed = read_style(t)
        out["read_style"][name], out["read_style_flags"][name] = score, failed

    eligible = [i for i in range(n) if len(norm[i].split()) >= REPEAT_MIN_WORDS]
    nbrs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if len(eligible) >= 2:
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True).fit_transform(
            [norm[i] for i in eligible])  # rows are L2-normalised -> dot product = cosine
        for r0 in range(0, len(eligible), 512):  # chunked so memory stays O(512 * n)
            S = (X[r0:r0 + 512] @ X.T).toarray()
            for r in range(S.shape[0]):
                S[r, r0 + r] = 0.0
                j = np.flatnonzero(S[r] >= threshold)
                nbrs[r0 + r] = (j, S[r, j])

    # connected groups of matching transcripts
    group = {k: -1 for k in range(len(eligible))}
    gid = 0
    for k in range(len(eligible)):
        if group[k] != -1 or not len(nbrs.get(k, ([],))[0]):
            continue
        stack, group[k] = [k], gid
        members = []
        while stack:
            u = stack.pop()
            members.append(u)
            for v in nbrs[u][0]:
                if group[v] == -1:
                    group[v] = gid
                    stack.append(v)
        # most central member = the one with the highest summed similarity to its matches
        central = max(members, key=lambda m: float(nbrs[m][1].sum()))
        out["repeat_group_text"][gid] = texts[eligible[central]]
        gid += 1

    for i, name in enumerate(names):
        out["repeat_count"][name], out["repeat_group"][name] = 0, -1
    for k, i in enumerate(eligible):
        j, s = nbrs.get(k, (np.zeros(0, int), np.zeros(0)))
        out["repeat_count"][names[i]] = int(len(j))
        out["repeat_group"][names[i]] = int(group[k])
        if len(j):
            out["repeat_deviation"][names[i]] = float(1.0 - s.mean())
    return out


def _cached_corpus() -> dict:
    """Corpus statistics of the cached HEARSAY test transcripts (built once, ~0.1 s)."""
    global _CACHED_CORPUS
    if _CACHED_CORPUS is None:
        _CACHED_CORPUS = build_corpus_stats(load_cached_transcripts())
    return _CACHED_CORPUS


# ----------------------------------------------------------------------------------------------
# Expert 2: repeated_script
# ----------------------------------------------------------------------------------------------
@expert("repeated_script", CATEGORY, cost="cheap")
def repeated_script(ctx: Context) -> Evidence:
    """Batch script analysis from transcripts: how many other clips speak the same sentence (mass-produced
    scripts), how badly this reading deviates from the others, and whether the text is a complete read
    sentence or a fragment/loop. Uses ctx.corpus (build_corpus_stats) or the cached test-set transcripts."""
    # fixed column set; NaN = unknown (no batch transcripts, or no other reading of this sentence)
    ev = Evidence(expert="repeated_script", category=CATEGORY,
                  features={k: float("nan") for k in ("semantic_repeat_count", "semantic_repeat_deviation",
                                                      "semantic_read_style")})
    base = os.path.basename(ctx.path)
    corpus, source = ctx.corpus, "batch"
    if base not in (corpus.get("repeat_count") or {}):
        if base in load_cached_transcripts():
            corpus, source = _cached_corpus(), "cached HEARSAY test-set transcripts"
        else:
            ev.applicable = False
            ev.findings.append(Finding("No batch transcripts available, so repeated-script analysis was skipped "
                                       "(it needs the transcripts of the other files in the batch).", 0, 0.0))
            return ev

    count = int(corpus["repeat_count"][base])
    dev = (corpus.get("repeat_deviation") or {}).get(base)
    style = (corpus.get("read_style") or {}).get(base)
    flags = (corpus.get("read_style_flags") or {}).get(base, [])
    gid = (corpus.get("repeat_group") or {}).get(base, -1)
    script = (corpus.get("repeat_group_text") or {}).get(gid, "")
    own = load_cached_transcripts().get(base) if source != "batch" else None
    if own is None:
        own = ctx.cache.get("transcript") if isinstance(ctx.cache.get("transcript"), str) else None
    n = int(corpus.get("corpus_size", 0))

    ev.features["semantic_repeat_count"] = float(count)
    if dev is not None:
        ev.features["semantic_repeat_deviation"] = float(dev)
    if style is not None:
        ev.features["semantic_read_style"] = float(style)
    ev.artifacts.update({"corpus_source": source, "corpus_size": n, "repeat_group": gid, "group_script": script})

    # --- findings ---------------------------------------------------------------------------
    quoted = f' ("{script[:100]}")' if script else ""
    if count >= 5:
        ev.findings.append(Finding(
            f"The same sentence{quoted} appears in {count} other clips of the batch - mass-produced scripts "
            "are a hallmark of synthetic campaigns (on the HEARSAY test set, 44% of clips whose sentence recurs "
            "5+ times are flagged by the AASIST detector, versus 15% of clips with a unique sentence).",
            +1, float(min(0.5, 0.3 + 0.01 * (count - 5)))))
    elif count >= 1:
        ev.findings.append(Finding(
            f"The sentence{quoted} is shared with {count} other clip(s). A few repeats are normal in read-speech "
            "collections, so this is not suspicious on its own.", 0, 0.0))
    elif n > 1:
        ev.findings.append(Finding(f"The sentence is unique within this batch of {n} transcribed clips.", 0, 0.0))
    if count >= 2 and dev is not None and dev >= DEVIATION_FLAG and own:
        ev.findings.append(Finding(
            f'The recogniser heard "{own[:100]}" where the other readings of this script say "{script[:100]}" '
            f"(deviation {dev:.2f}). Words mis-heard only in this copy suggest slurred or garbled articulation, "
            "which is more common in synthetic speech than in careful human reading.", +1, 0.25))
    if "recogniser loop" in flags:
        ev.findings.append(Finding(
            "The transcript degenerates into a repetition loop, i.e. the recogniser could not map part of the "
            "audio onto words - typical of vocoder buzz or garbled synthetic phonemes.", +1, 0.5))
    elif flags and style is not None and style < 1.0:
        fragment = [x for x in flags if x in ("starts mid-sentence", "no sentence ending")]
        if fragment:
            ev.findings.append(Finding(
                f"The transcript reads like a fragment ({', '.join(fragment)}) rather than a complete scripted "
                "sentence: the recogniser did not hear a natural sentence onset/ending, which can mean the audio "
                "was cut from a longer synthetic utterance or its first/last words are smeared.",
                +1, 0.2 if "no sentence ending" in fragment else 0.15))
    return ev
