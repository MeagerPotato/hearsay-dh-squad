"""Container, metadata and chain-of-custody forensics.

This expert never looks at the audio samples. It reads the file's bytes and metadata the way a
digital-evidence examiner would:

  * what the file really is (magic bytes) versus what its extension claims,
  * the codec / sample rate / bit depth / channel layout / bitrate reported by ffprobe,
  * the internal structure: a RIFF chunk walk for WAV (chunk list, INFO/bext/iXML provenance,
    size-field consistency), ID3 / Xing / LAME / MP4 atoms / Vorbis comments via mutagen, and
    exiftool for anything else,
  * which software last wrote the file (encoder tags such as ``Lavf58.29.100`` or ``LAME3.100``) and
    whether that matches known TTS-service output profiles,
  * embedded creation timestamps versus the filesystem MAC times,
  * a SHA-256 so the report can be tied to an exact byte stream.

Honesty rule: when a file has been rewritten by a transcoder (e.g. every HEARSAY test clip carries
the same FFmpeg header) the original provenance is gone and we say so ("container evidence
uninformative") instead of inventing a signal.

A compact summary is left in ``ctx.cache["container"]`` so the orchestrator and the compression
experts can adapt (e.g. analyse at the native sample rate, or know the declared MP3 bitrate).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from ..audio import probe
from ..evidence import Context, Evidence, Finding, expert

EXIFTOOL = shutil.which("exiftool") or ("/opt/homebrew/bin/exiftool" if os.path.exists("/opt/homebrew/bin/exiftool") else None)

# --------------------------------------------------------------------------------------------------
# Reference tables
# --------------------------------------------------------------------------------------------------

# Extension -> container family we would expect to find inside.
EXT_FAMILY = {
    ".wav": "wav", ".wave": "wav", ".bwf": "wav", ".rf64": "wav",
    ".mp3": "mp3", ".mp2": "mp3", ".mpga": "mp3",
    ".m4a": "mp4", ".mp4": "mp4", ".m4b": "mp4", ".mov": "mp4", ".3gp": "mp4", ".3g2": "mp4", ".m4v": "mp4",
    ".aac": "aac", ".flac": "flac", ".ogg": "ogg", ".oga": "ogg", ".opus": "ogg", ".spx": "ogg",
    ".aif": "aiff", ".aiff": "aiff", ".aifc": "aiff", ".amr": "amr", ".awb": "amr",
    ".webm": "matroska", ".mka": "matroska", ".mkv": "matroska", ".caf": "caf", ".au": "au", ".snd": "au",
    ".wma": "asf", ".asf": "asf", ".wv": "wavpack", ".ape": "ape",
}

LOSSY_CODECS = {"mp3", "mp2", "mp1", "aac", "opus", "vorbis", "amr_nb", "amr_wb", "gsm", "gsm_ms", "wmav1", "wmav2",
                "ac3", "eac3", "speex", "adpcm_ima_wav", "adpcm_ms", "g722", "g723_1", "g729", "pcm_alaw", "pcm_mulaw",
                "qdm2", "atrac3", "ilbc", "nellymoser", "adpcm_g726"}

WAV_FORMAT_TAGS = {1: "PCM", 2: "MS-ADPCM", 3: "IEEE float", 6: "A-law", 7: "mu-law", 0x11: "IMA-ADPCM",
                   0x31: "GSM 6.10", 0x50: "MPEG", 0x55: "MPEG Layer 3", 0x161: "WMA", 0xFFFE: "Extensible"}

# Chunks that any ordinary WAV writer may emit; everything else is "non-standard" and worth listing.
STANDARD_WAV_CHUNKS = {"fmt ", "data", "fact", "LIST", "PEAK", "JUNK", "junk", "pad ", "PAD ", "ds64", "id3 ", "ID3 "}
# Chunks written by professional recorders / DAWs: evidence of a genuine capture chain.
PROVENANCE_CHUNKS = {"bext": "Broadcast-WAV (bext) origination record", "iXML": "iXML field-recorder metadata",
                     "axml": "aXML metadata", "cart": "CartChunk broadcast metadata", "FLLR": "Apple filler chunk",
                     "minf": "Pro Tools session chunk", "elm1": "Pro Tools chunk", "regn": "Pro Tools region chunk",
                     "umid": "Pro Tools UMID", "LGWV": "Logic Pro chunk", "ResU": "Logic Pro chunk",
                     "cue ": "cue points", "smpl": "sampler chunk", "inst": "instrument chunk", "acid": "ACID loop chunk"}

# Encoder / software strings -> (label, kind). kind drives the direction of the finding.
#   "tts"       : text-to-speech / voice-cloning service or toolkit  -> suggests synthetic
#   "device"    : recording hardware / phone recorder app             -> suggests a real capture chain
#   "transcoder": generic conversion tool; provenance stripped        -> neutral
#   "editor"    : audio editor / DAW                                  -> neutral (editing != synthesis)
#   "library"   : programming library (Python/C) wrote the file      -> neutral, weakly TTS-typical
#   "platform"  : OS media framework (used by recorders and converters alike) -> neutral
ENCODER_SIGNATURES: List[Tuple[str, str, str]] = [
    (r"eleven\s*labs|elevenlabs", "ElevenLabs", "tts"),
    (r"play\.?ht", "PlayHT", "tts"),
    (r"resemble", "Resemble AI", "tts"),
    (r"murf|wellsaid|speechify|lovo|listnr|descript|overdub|respeecher|voice\.?ai|uberduck|fakeyou",
     "commercial voice-cloning service", "tts"),
    (r"coqui|xtts|tortoise|\bbark\b|openvoice|styletts|vits|piper|espnet|tacotron|fastspeech|hifi-?gan|rvc|so-vits",
     "open-source TTS / voice-conversion toolkit", "tts"),
    (r"openai|tts-1|polly|wavenet|azure.*(speech|tts)|cognitive services", "cloud TTS API", "tts"),
    (r"^lavf|^lavc|ffmpeg|libav", "FFmpeg (libavformat/libavcodec)", "transcoder"),
    (r"^lame|lame3|lame\s*\d", "LAME MP3 encoder", "transcoder"),
    (r"sox\b", "SoX", "transcoder"),
    (r"libsndfile", "libsndfile (used by Python soundfile / torchaudio)", "library"),
    (r"libopus|libvorbis|xiph\.org|reference libflac|libflac", "Xiph reference codec library", "transcoder"),
    (r"audacity", "Audacity", "editor"),
    (r"adobe|audition|premiere", "Adobe audio software", "editor"),
    (r"logic pro|garageband|pro tools|reaper|cubase|nuendo|fl studio|ableton|wavelab|sound forge|izotope",
     "digital audio workstation", "editor"),
    (r"voice ?memos|com\.apple\.voicememos|recorder|dictaphone|olympus|tascam|zoom h\d|zoom f\d|marantz|roland|"
     r"sony ic|philips|samsung voice|pixel recorder|lg voice|huawei|xiaomi|whatsapp|telegram",
     "recording device or phone recorder app", "device"),
    (r"itunes|core ?media|coreaudio|apple", "Apple media framework", "platform"),  # recorders AND converters use it
]

# LAME's default low-pass (Hz) by total CBR bitrate (kbps). Used by the compression expert.
LAME_LOWPASS = [(8, 2000), (16, 3700), (24, 3900), (32, 5500), (40, 7000), (48, 7500), (56, 10000), (64, 11000),
                (80, 13500), (96, 15100), (112, 15600), (128, 17000), (160, 17500), (192, 18600), (224, 19400),
                (256, 19700), (320, 20500)]

# --------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------


def _utc(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _num(v: Any, cast=float) -> float:
    """ffprobe reports numbers as strings and sometimes as 'N/A'; never let that crash the expert."""
    try:
        return cast(float(v))
    except (TypeError, ValueError):
        return cast(0)


def _clean(s: Any, limit: int = 120) -> str:
    """Printable, single-line, bounded version of a metadata value."""
    s = s.decode("latin-1", "replace") if isinstance(s, bytes) else str(s)
    s = s.replace("\x00", "").strip()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = "".join(ch if ch.isprintable() else "?" for ch in s)
    return s[:limit]


def get_probe(ctx: Context) -> Dict[str, Any]:
    """ffprobe JSON for ctx.path, shared through ctx.cache / ctx.native so it runs once per file."""
    if "ffprobe" in ctx.cache:
        return ctx.cache["ffprobe"]
    p = None
    if isinstance(ctx.native.get("ffprobe"), dict):
        p = ctx.native["ffprobe"]
    elif "streams" in ctx.native or "format" in ctx.native:
        p = ctx.native
    if p is None:
        p = probe(ctx.path) if ctx.path and os.path.exists(ctx.path) else {}
    ctx.cache["ffprobe"] = p or {}
    return ctx.cache["ffprobe"]


def audio_stream(pr: Dict[str, Any]) -> Dict[str, Any]:
    return next((s for s in pr.get("streams", []) if s.get("codec_type") == "audio"), {})


def lame_expected_lowpass(kbps: float) -> Optional[float]:
    """LAME's default low-pass for a CBR bitrate (linear interpolation of its tuning table)."""
    if not kbps or kbps <= 0:
        return None
    pts = LAME_LOWPASS
    if kbps <= pts[0][0]:
        return float(pts[0][1])
    for (b0, f0), (b1, f1) in zip(pts, pts[1:]):
        if kbps <= b1:
            return float(f0 + (f1 - f0) * (kbps - b0) / (b1 - b0))
    return float(pts[-1][1])


# --------------------------------------------------------------------------------------------------
# Format sniffing
# --------------------------------------------------------------------------------------------------


def _id3v2_size(head: bytes) -> int:
    """Total size of a leading ID3v2 tag (0 if none)."""
    if len(head) >= 10 and head[:3] == b"ID3":
        sz = (head[6] & 0x7F) << 21 | (head[7] & 0x7F) << 14 | (head[8] & 0x7F) << 7 | (head[9] & 0x7F)
        return 10 + sz + (10 if head[5] & 0x10 else 0)
    return 0


def _is_mpeg_audio_sync(b: bytes) -> bool:
    """MPEG audio frame header (layer I/II/III), excluding the ADTS-AAC pattern (layer bits 00)."""
    return len(b) >= 2 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0 and (b[1] & 0x06) != 0


def sniff_format(path: str) -> Tuple[str, str]:
    """Identify the real container from magic bytes. Returns (family, human description)."""
    with open(path, "rb") as f:
        head = f.read(64)
        if head[:3] == b"ID3":
            f.seek(_id3v2_size(head))
            after = f.read(4)
            if _is_mpeg_audio_sync(after):
                return "mp3", "MPEG audio (MP3) with ID3v2 tag"
            if len(after) >= 2 and after[0] == 0xFF and (after[1] & 0xF6) == 0xF0:
                return "aac", "ADTS AAC with ID3v2 tag"
            return "mp3", "ID3v2-tagged stream (MPEG audio assumed)"
    if head[:4] in (b"RIFF", b"RF64", b"BW64", b"RIFX") and head[8:12] == b"WAVE":
        return "wav", {b"RIFF": "RIFF/WAVE", b"RF64": "RF64/WAVE", b"BW64": "BW64/WAVE", b"RIFX": "RIFX/WAVE (big-endian)"}[head[:4]]
    if head[:4] == b"fLaC":
        return "flac", "FLAC"
    if head[:4] == b"OggS":
        kind = "Opus" if b"OpusHead" in head else "Vorbis" if b"vorbis" in head else "Speex" if b"Speex" in head else "FLAC" if b"FLAC" in head else "unknown codec"
        return "ogg", f"Ogg ({kind})"
    if head[4:8] == b"ftyp":
        return "mp4", f"ISO-BMFF/MP4 (brand '{_clean(head[8:12])}')"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff", _clean(head[8:12])
    if head[:5] == b"#!AMR":
        return "amr", "AMR (3GPP narrow/wide-band speech codec)"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "matroska", "Matroska/WebM"
    if head[:4] == b"caff":
        return "caf", "Apple CAF"
    if head[:4] == b".snd":
        return "au", "Sun/NeXT AU"
    if head[:4] == b"0&\xb2u":
        return "asf", "ASF/WMA"
    if head[:4] == b"wvpk":
        return "wavpack", "WavPack"
    if head[:4] == b"MAC ":
        return "ape", "Monkey's Audio"
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xF6) == 0xF0:
        return "aac", "ADTS AAC"
    if _is_mpeg_audio_sync(head):
        return "mp3", "MPEG audio (MP3) without ID3 tag"
    return "unknown", "unrecognised byte signature"


# --------------------------------------------------------------------------------------------------
# RIFF / WAV chunk walk
# --------------------------------------------------------------------------------------------------


def walk_riff(path: str, max_chunks: int = 512) -> Optional[Dict[str, Any]]:
    """Walk the chunk list of a RIFF/RF64 WAVE file and check its bookkeeping.

    Returns chunk table, parsed fmt / LIST-INFO / bext, and a list of structural problems
    (size fields that disagree with the file length, truncation, trailing bytes, garbage chunk ids).
    """
    size = os.path.getsize(path)
    out: Dict[str, Any] = {"form": None, "file_size": size, "riff_size": None, "chunks": [], "problems": [],
                           "fmt": {}, "info": {}, "bext": {}, "has_ixml": False}
    with open(path, "rb") as f:
        hdr = f.read(12)
        if len(hdr) < 12 or hdr[:4] not in (b"RIFF", b"RF64", b"BW64", b"RIFX") or hdr[8:12] != b"WAVE":
            return None
        e = ">" if hdr[:4] == b"RIFX" else "<"
        out["form"] = hdr[:4].decode()
        riff_size = struct.unpack(e + "I", hdr[4:8])[0]
        out["riff_size"] = riff_size
        ds64_data = None
        # Walk only inside the declared RIFF form; bytes beyond it are reported as appended data below.
        form_end = size if hdr[:4] != b"RIFF" or riff_size + 8 > size else riff_size + 8
        pos = 12
        while pos + 8 <= form_end and len(out["chunks"]) < max_chunks:
            f.seek(pos)
            ch = f.read(8)
            cid_b, csz = ch[:4], struct.unpack(e + "I", ch[4:8])[0]
            cid = cid_b.decode("latin-1")
            if not all(32 <= c < 127 for c in cid_b):
                out["problems"].append(f"unreadable chunk id {cid_b!r} at byte {pos} (file structure is corrupt or was spliced)")
                break
            if cid == "data" and csz == 0xFFFFFFFF and ds64_data is not None:
                csz = ds64_data
            body = pos + 8
            truncated = body + csz > size
            out["chunks"].append({"id": cid, "offset": pos, "size": csz, "truncated": truncated})
            if truncated:
                out["problems"].append(f"chunk '{cid}' at byte {pos} declares {csz:,} bytes but only {size - body:,} remain (truncated file or forged size field)")
            if csz <= (1 << 20) and not truncated:
                f.seek(body)
                data = f.read(csz)
                try:
                    _parse_small_chunk(cid, data, e, out)
                    if cid == "ds64" and len(data) >= 16:
                        ds64_data = struct.unpack("<Q", data[8:16])[0]
                except Exception as ex:  # a malformed sub-chunk is itself a finding, never a crash
                    out["problems"].append(f"malformed '{cid}' chunk ({type(ex).__name__})")
            if truncated:
                break
            pos = body + csz + (csz & 1)  # RIFF chunks are word-aligned
        if pos < form_end and not any(c["truncated"] for c in out["chunks"]) and form_end - pos > 1:
            out["problems"].append(f"{form_end - pos:,} unparsed bytes after the last chunk inside the RIFF form")
    # RIFF size bookkeeping (RF64 legitimately stores 0xFFFFFFFF here).
    if out["form"] == "RIFF" and riff_size + 8 != size and not any(c["truncated"] for c in out["chunks"]):
        if riff_size + 8 > size:
            out["problems"].append(f"RIFF header claims {riff_size + 8:,} bytes but the file has {size:,} (truncated)")
        elif size - (riff_size + 8) > 1:
            out["problems"].append(f"{size - riff_size - 8:,} bytes appended after the end of the RIFF form "
                                   f"(header claims {riff_size + 8:,} bytes, file has {size:,})")
    ids = [c["id"] for c in out["chunks"]]
    for need in ("fmt ", "data"):
        if need not in ids:
            out["problems"].append(f"required '{need.strip()}' chunk missing")
    for dup in ("fmt ", "data"):
        if ids.count(dup) > 1:
            out["problems"].append(f"{ids.count(dup)} '{dup.strip()}' chunks (only one is legal; typical of naive splicing)")
    fmt = out["fmt"]
    if fmt and fmt.get("tag") == 1 and fmt.get("bits"):
        ba = fmt["channels"] * ((fmt["bits"] + 7) // 8)
        if fmt["block_align"] != ba or fmt["byte_rate"] != fmt["sr"] * fmt["block_align"]:
            out["problems"].append("fmt chunk arithmetic is inconsistent (byte rate / block align do not match sample rate x channels x bit depth)")
        data_c = next((c for c in out["chunks"] if c["id"] == "data"), None)
        if data_c and fmt["block_align"] and data_c["size"] % fmt["block_align"]:
            out["problems"].append("data chunk length is not a whole number of sample frames")
    return out


def _parse_small_chunk(cid: str, data: bytes, e: str, out: Dict[str, Any]) -> None:
    if cid == "fmt " and len(data) >= 16:
        tag, ch, sr, br, ba, bits = struct.unpack(e + "HHIIHH", data[:16])
        fmt = {"tag": tag, "channels": ch, "sr": sr, "byte_rate": br, "block_align": ba, "bits": bits}
        if tag == 0xFFFE and len(data) >= 26:
            fmt["valid_bits"] = struct.unpack(e + "H", data[18:20])[0]
            fmt["subformat"] = struct.unpack(e + "H", data[24:26])[0]
        eff = fmt.get("subformat", tag)
        fmt["format_name"] = WAV_FORMAT_TAGS.get(eff, f"0x{eff:04X}")
        out["fmt"] = fmt
    elif cid == "LIST" and len(data) >= 4:
        ltype = data[:4].decode("latin-1")
        out.setdefault("list_types", []).append(ltype)
        if ltype == "INFO":
            p = 4
            while p + 8 <= len(data):
                sid = data[p:p + 4].decode("latin-1")
                ssz = struct.unpack(e + "I", data[p + 4:p + 8])[0]
                out["info"][sid] = _clean(data[p + 8:p + 8 + ssz])
                p += 8 + ssz + (ssz & 1)
    elif cid == "bext" and len(data) >= 346:
        out["bext"] = {"description": _clean(data[0:256]), "originator": _clean(data[256:288]),
                       "originator_reference": _clean(data[288:320]), "origination_date": _clean(data[320:330]),
                       "origination_time": _clean(data[330:338]), "coding_history": _clean(data[602:], 300)}
    elif cid == "iXML":
        out["has_ixml"] = True
        m = re.search(rb"<(?:PROJECT|TAPE|NOTE|SCENE)>([^<]{1,80})<", data)
        if m:
            out["ixml_hint"] = _clean(m.group(1))


# --------------------------------------------------------------------------------------------------
# MPEG / MP4 / Ogg / FLAC tags via mutagen, plus a raw look at the MP3 Xing/Info/LAME header
# --------------------------------------------------------------------------------------------------


def mutagen_info(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import mutagen
        m = mutagen.File(path)
    except Exception as ex:
        return {"error": f"{type(ex).__name__}"}
    if m is None:
        return {}
    out["type"] = type(m).__name__
    info = getattr(m, "info", None)
    for k in ("sample_rate", "bitrate", "channels", "length", "bits_per_sample", "encoder_info", "encoder_settings",
              "sketchy", "codec", "codec_description"):
        v = getattr(info, k, None)
        if v not in (None, ""):
            out[k] = v if isinstance(v, (int, float, bool)) else _clean(v)
    bm = getattr(info, "bitrate_mode", None)
    if bm is not None:
        out["bitrate_mode"] = str(bm).split(".")[-1]
    tags: Dict[str, str] = {}
    try:
        if m.tags is not None:
            vendor = getattr(m.tags, "vendor", None)  # Vorbis comment / FLAC vendor string
            if vendor:
                tags["vendor"] = _clean(vendor)
            for k, v in list(m.tags.items())[:80]:
                key = str(k)
                if key.startswith(("APIC", "covr", "METADATA_BLOCK_PICTURE")):
                    continue  # embedded artwork is irrelevant and large
                val = v.text if hasattr(v, "text") else v
                if isinstance(val, (list, tuple)):
                    val = "; ".join(_clean(x, 80) for x in val[:4])
                tags[key] = _clean(val)
    except Exception:
        pass
    out["tags"] = tags
    return out


def mp3_header_info(path: str) -> Dict[str, Any]:
    """Look inside the first MPEG frame for a Xing/Info/VBRI header and the encoder string of a LAME tag."""
    with open(path, "rb") as f:
        head = f.read(10)
        start = _id3v2_size(head)
        f.seek(start)
        buf = f.read(4096)
    i = next((k for k in range(0, max(0, len(buf) - 4)) if _is_mpeg_audio_sync(buf[k:k + 2])), None)
    if i is None:
        return {"id3v2_bytes": start}
    frame = buf[i:i + 600]
    out = {"id3v2_bytes": start, "leading_junk_bytes": i}
    for tagname in (b"Xing", b"Info", b"VBRI"):
        if tagname in frame:
            out["vbr_header"] = tagname.decode()
    m = re.search(rb"(LAME\d[\d.a-z]*|Lavc\d+\.\d+|Lavf\d+\.\d+|GOGO|FhG)", frame)
    if m:
        out["lame_tag_encoder"] = _clean(m.group(1))
    return out


# --------------------------------------------------------------------------------------------------
# exiftool (fallback for formats not covered above)
# --------------------------------------------------------------------------------------------------

_EXIF_SKIP_GROUPS = ("ExifTool", "System", "File")


def exiftool_tags(path: str, timeout: float = 15.0) -> Dict[str, Any]:
    if not EXIFTOOL:
        return {"error": "exiftool not installed"}
    try:
        r = subprocess.run([EXIFTOOL, "-j", "-G1", "-a", "-s", "-api", "LargeFileSupport=1", path],
                           capture_output=True, text=True, timeout=timeout)
        rows = json.loads(r.stdout or "[]")
    except Exception as ex:
        return {"error": f"{type(ex).__name__}"}
    if not rows:
        return {}
    tags = {}
    for k, v in rows[0].items():
        grp = k.split(":", 1)[0]
        if k == "SourceFile" or grp in _EXIF_SKIP_GROUPS or "Binary data" in str(v):
            continue
        tags[k] = _clean(v)
        if len(tags) >= 80:
            break
    return tags


# --------------------------------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------------------------------

_TIME_KEYS = re.compile(r"(creation_time|date|time|icrd|tdrc|tden|tdor|tdtg|tyer|©day|createdate|datetimeoriginal|"
                        r"mediacreatedate|trackcreatedate|origination)", re.I)


def parse_time(s: str) -> Optional[_dt.datetime]:
    """Parse the zoo of metadata date formats into an aware UTC datetime (None if unparseable)."""
    s = _clean(s, 60)
    if not s or not re.match(r"^\d{4}", s):
        return None
    t = s.replace("Z", "+00:00")
    t = re.sub(r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", t)       # exif style 2024:01:31 12:00:00
    t = re.sub(r"^(\d{4}-\d{2}-\d{2})[ T]?(\d{2})[:-](\d{2})[:-](\d{2})", r"\1T\2:\3:\4", t)
    t = re.sub(r"(\.\d{6})\d+", r"\1", t)
    for cand in (t, t[:19], t[:10], t[:7] + "-01" if len(t) >= 7 else "", t[:4] + "-01-01"):
        try:
            d = _dt.datetime.fromisoformat(cand)
            return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
        except Exception:
            continue
    return None


def embedded_times(pr: Dict[str, Any], riff: Optional[Dict[str, Any]], mut: Dict[str, Any],
                   exif: Dict[str, Any]) -> List[Tuple[str, str, Optional[_dt.datetime]]]:
    """(source, raw value, parsed) for every creation/recording timestamp found inside the file."""
    found = []
    for src, tags in (("ffprobe format", pr.get("format", {}).get("tags", {})),
                      ("ffprobe stream", audio_stream(pr).get("tags", {})),
                      ("mutagen", (mut or {}).get("tags", {})),
                      ("exiftool", exif or {})):
        for k, v in (tags or {}).items():
            if _TIME_KEYS.search(str(k)) and not re.search(r"(modify|access|inodechange|filemodify)", str(k), re.I):
                found.append((f"{src}:{k}", _clean(v, 40), parse_time(str(v))))
    if riff:
        if riff["info"].get("ICRD"):
            found.append(("RIFF INFO ICRD", riff["info"]["ICRD"], parse_time(riff["info"]["ICRD"])))
        b = riff.get("bext") or {}
        if b.get("origination_date"):
            raw = f"{b['origination_date']} {b.get('origination_time', '')}".strip()
            found.append(("bext origination", raw, parse_time(raw.replace(" ", "T"))))
    # de-duplicate identical raw values reported by several parsers
    seen, uniq = set(), []
    for src, raw, d in found:
        if raw not in seen:
            seen.add(raw)
            uniq.append((src, raw, d))
    return uniq


# --------------------------------------------------------------------------------------------------
# The expert
# --------------------------------------------------------------------------------------------------


def _match_signatures(strings: List[Tuple[str, str]]) -> List[Tuple[str, str, str, str]]:
    """[(source, value, label, kind)] for every encoder/software string that matches a known signature."""
    hits = []
    for src, val in strings:
        for pat, label, kind in ENCODER_SIGNATURES:
            if re.search(pat, val, re.I):
                hits.append((src, val, label, kind))
                break
    return hits


FEATURE_DEFAULTS = {k: 0.0 for k in (
    "container_is_pcm", "container_is_lossy", "container_native_sr", "container_bits_per_sample", "container_channels",
    "container_duration_s", "container_bitrate_kbps", "container_ext_mismatch", "container_n_chunks",
    "container_nonstandard_chunks", "container_structure_problems", "container_duration_mismatch_s",
    "container_encoder_ffmpeg", "container_encoder_tts", "container_device_provenance", "container_n_descriptive_tags",
    "container_n_embedded_times", "container_time_inconsistent", "container_tts_profile", "container_vocoder_rate",
    "container_laundered")}


@expert("container", "container", cost="cheap")
def analyze(ctx: Context) -> Evidence:
    """Container / metadata chain-of-custody check: real format vs extension, ffprobe stream parameters, RIFF chunk walk
    (INFO/bext/iXML, size-field consistency), ID3/LAME/MP4/Vorbis tags, exiftool fallback, encoder signatures of
    transcoders and TTS services, embedded timestamps vs filesystem MAC times, and a SHA-256 of the exact bytes."""
    ev = Evidence(expert="container", category="container", features=dict(FEATURE_DEFAULTS))
    path = ctx.path
    if not path or not os.path.isfile(path):
        ev.applicable = False
        ev.findings.append(Finding("No file on disk (in-memory audio), so container and metadata forensics do not apply.", 0, 0.0))
        return ev

    notes: List[str] = []
    st = os.stat(path)
    size = st.st_size
    if size == 0:
        ev.features.update({"container_structure_problems": 1.0})
        ev.findings.append(Finding(f"Custody record: {os.path.basename(path)} is an empty (0-byte) file; there is no audio "
                                   "or metadata to examine.", 0, 0.0))
        return ev
    sha = _sha256(path)
    family, family_desc = sniff_format(path)
    ext = os.path.splitext(path)[1].lower()
    ext_family = EXT_FAMILY.get(ext)
    ext_mismatch = bool(ext_family and family != "unknown" and ext_family != family)

    # ---- stream parameters (ffprobe) -----------------------------------------------------------
    pr = get_probe(ctx)
    ast = audio_stream(pr)
    fmt_tags = {str(k): _clean(v) for k, v in (pr.get("format", {}).get("tags") or {}).items()}
    st_tags = {str(k): _clean(v) for k, v in (ast.get("tags") or {}).items()}
    codec = ast.get("codec_name", "") or ""
    native_sr = _num(ast.get("sample_rate"), int) or _num(ctx.native.get("sr"), int)
    channels = _num(ast.get("channels"), int)
    bits = _num(ast.get("bits_per_raw_sample"), int) or _num(ast.get("bits_per_sample"), int)
    duration = _num(pr.get("format", {}).get("duration")) or _num(ast.get("duration"))
    bitrate = _num(ast.get("bit_rate")) or _num(pr.get("format", {}).get("bit_rate"))
    lossy = codec in LOSSY_CODECS
    if not pr:
        notes.append("ffprobe unavailable or failed")

    # ---- structure-specific parsing --------------------------------------------------------------
    riff = walk_riff(path) if family == "wav" else None
    mut = mutagen_info(path) if family in ("mp3", "mp4", "ogg", "flac", "aiff", "asf", "wavpack", "ape") else {}
    mp3h = mp3_header_info(path) if family == "mp3" else {}
    need_exif = family not in ("wav", "mp3", "mp4", "ogg", "flac") or (family != "wav" and (not mut or "error" in mut))
    exif = exiftool_tags(path) if need_exif else {}

    if riff and riff["fmt"]:
        bits = bits or riff["fmt"].get("bits", 0)
    if not bits and codec.startswith("pcm_"):
        m = re.search(r"(\d+)", codec)
        bits = int(m.group(1)) if m else 0

    # ---- collect every software / encoder string ------------------------------------------------
    enc_strings: List[Tuple[str, str]] = []
    for k in ("encoder", "encoded_by", "software", "handler_name", "vendor_id", "comment", "tool"):
        if fmt_tags.get(k):
            enc_strings.append((f"ffprobe format tag '{k}'", fmt_tags[k]))
        if st_tags.get(k):
            enc_strings.append((f"ffprobe stream tag '{k}'", st_tags[k]))
    if riff:
        for k in ("ISFT", "IENG", "ISRC", "ICMT", "IART", "ITCH"):
            if riff["info"].get(k):
                enc_strings.append((f"RIFF INFO {k}", riff["info"][k]))
        b = riff.get("bext") or {}
        for k in ("originator", "description", "coding_history"):
            if b.get(k):
                enc_strings.append((f"bext {k}", b[k]))
    for k, v in (mut.get("tags") or {}).items():
        if re.match(r"(TSSE|TENC|TOOL|©too|vendor|ENCODER|encoder|ENCODED.?BY|COMM|©cmt|TXXX)", k):
            enc_strings.append((f"tag {k}", v))
    if mut.get("encoder_info"):
        enc_strings.append(("MP3 LAME tag", mut["encoder_info"]))
    if mp3h.get("lame_tag_encoder"):
        enc_strings.append(("MP3 Xing/Info header", mp3h["lame_tag_encoder"]))
    for k, v in exif.items():
        if re.search(r"(software|encoder|writer|make|model|creatortool|handler|vendor|comment)", k, re.I):
            enc_strings.append((f"exiftool {k}", v))
    enc_strings = list({v: (s, v) for s, v in reversed(enc_strings)}.values())[::-1]  # de-duplicate by value, keep first source
    sig_hits = _match_signatures(enc_strings)
    kinds = {h[3] for h in sig_hits}

    # ---- provenance vs laundering ----------------------------------------------------------------
    chunk_ids = [c["id"] for c in riff["chunks"]] if riff else []
    provenance = [PROVENANCE_CHUNKS[c] for c in chunk_ids if c in PROVENANCE_CHUNKS]
    nonstandard = [c for c in chunk_ids if c not in STANDARD_WAV_CHUNKS and c not in PROVENANCE_CHUNKS]
    descriptive_tags = {k: v for k, v in {**(riff["info"] if riff else {}), **(mut.get("tags") or {}), **fmt_tags}.items()
                        if not re.match(r"(ISFT|TSSE|TENC|©too|vendor|encoder|major_brand|minor_version|compatible_brands)$", k)}
    device = bool(provenance) or "device" in kinds or bool(riff and riff.get("bext"))
    only_ffmpeg = bool(sig_hits) and kinds <= {"transcoder"} and any("FFmpeg" in h[2] for h in sig_hits)
    laundered = only_ffmpeg and not device and not descriptive_tags

    # ---- known TTS output profiles (weak, heuristic) ---------------------------------------------
    profiles: List[Tuple[float, str]] = []
    if "tts" in kinds:
        h = next(h for h in sig_hits if h[3] == "tts")
        profiles.append((0.9, f"the {h[0]} names a speech-synthesis product ('{h[1]}' -> {h[2]})"))
    if family == "mp3" and native_sr == 44100 and channels == 1 and 120e3 <= bitrate <= 136e3 \
            and mp3h.get("id3v2_bytes", 0) == 0 and "vbr_header" not in mp3h:
        profiles.append((0.35, "bare 44.1 kHz / 128 kbps mono CBR MP3 with no ID3 tag and no Xing/LAME header matches the "
                               "ElevenLabs API default output profile"))
    if family == "mp3" and native_sr == 24000 and channels == 1:
        profiles.append((0.25, "24 kHz mono MP3 is the default delivery format of several TTS APIs (e.g. PlayHT, OpenAI)"))
    if family == "wav" and riff and native_sr in (22050, 24000) and set(chunk_ids) <= {"fmt ", "data", "fact", "PEAK"}:
        profiles.append((0.15, f"minimal fmt+data WAV at {native_sr} Hz, the rate of most neural vocoders, as written by Python "
                               "audio libraries (weak: real corpora such as LJSpeech share this format)"))
    if family == "wav" and riff and riff["fmt"].get("format_name") == "IEEE float" and "PEAK" in chunk_ids:
        profiles.append((0.15, "32-bit float WAV with a libsndfile PEAK chunk, typical of ML/Python output rather than recorders"))
    tts_profile = max((p for p, _ in profiles), default=0.0)

    # ---- timestamps vs MAC times -----------------------------------------------------------------
    mtime, ctime = st.st_mtime, st.st_ctime
    btime = getattr(st, "st_birthtime", None)
    etimes = embedded_times(pr, riff, mut, exif)
    now = _dt.datetime.now(_dt.timezone.utc)
    mdt = _dt.datetime.fromtimestamp(mtime, _dt.timezone.utc)
    time_issues: List[str] = []
    for src, raw, d in etimes:
        if d is None:
            continue
        if d.year < 1971:
            time_issues.append(f"{src} = '{raw}' is an epoch placeholder (timestamp never set by a recorder)")
        elif d > now + _dt.timedelta(days=1):
            time_issues.append(f"{src} = '{raw}' lies in the future")
        elif d > mdt + _dt.timedelta(days=1):
            time_issues.append(f"{src} = '{raw}' is later than the file's last-modified time ({_utc(mtime)})")
    decoded_dur = len(ctx.audio) / ctx.sr if ctx.audio is not None and ctx.sr else 0.0
    dur_mismatch = abs(duration - decoded_dur) if duration and decoded_dur else 0.0
    problems = list(riff["problems"]) if riff else []
    if duration and decoded_dur and dur_mismatch > max(0.25, 0.05 * duration):
        problems.append(f"header duration {duration:.2f} s disagrees with decoded audio length {decoded_dur:.2f} s")

    # ---- features --------------------------------------------------------------------------------
    f = ev.features
    f["container_is_pcm"] = float(codec.startswith("pcm_") and codec not in ("pcm_alaw", "pcm_mulaw"))
    f["container_is_lossy"] = float(lossy)
    f["container_native_sr"] = float(native_sr)
    f["container_bits_per_sample"] = float(bits)
    f["container_channels"] = float(channels)
    f["container_duration_s"] = float(duration)
    f["container_bitrate_kbps"] = float(bitrate / 1000.0)
    f["container_ext_mismatch"] = float(ext_mismatch)
    f["container_n_chunks"] = float(len(chunk_ids))
    f["container_nonstandard_chunks"] = float(len(nonstandard))
    f["container_structure_problems"] = float(len(problems))
    f["container_duration_mismatch_s"] = float(dur_mismatch)
    f["container_encoder_ffmpeg"] = float(any("FFmpeg" in h[2] for h in sig_hits))
    f["container_encoder_tts"] = float("tts" in kinds)
    f["container_device_provenance"] = float(device)
    f["container_n_descriptive_tags"] = float(len(descriptive_tags))
    f["container_n_embedded_times"] = float(len(etimes))
    f["container_time_inconsistent"] = float(len(time_issues))
    f["container_tts_profile"] = float(tts_profile)
    f["container_vocoder_rate"] = float(native_sr in (22050, 24000))
    f["container_laundered"] = float(laundered)

    # ---- findings (chain-of-custody note) ----------------------------------------------------------
    stream_desc = (f"{codec or family} {'%d-bit ' % bits if bits and not lossy else ''}"
                   f"{'mono' if channels == 1 else '%d-ch' % channels if channels else ''} {native_sr/1000:g} kHz"
                   f"{', %d kbps' % round(bitrate / 1000) if lossy and bitrate else ''}, {duration:.2f} s")
    identity = (f"Custody record: {os.path.basename(path)}, {size:,} bytes, SHA-256 {sha[:16]}...; content is {family_desc} "
                f"({stream_desc}). Filesystem times on this machine: modified {_utc(mtime)}"
                f"{', created ' + _utc(btime) if btime else ''}, inode changed {_utc(ctime)}. "
                "MAC times describe this copy, not the original recording.")
    ranked: List[Finding] = []

    if ext_mismatch:
        ranked.append(Finding(
            f"Extension mismatch: the file is named '{ext}' but its bytes are {family_desc}. It was renamed rather than "
            "converted, a pattern common in TTS download pipelines that save an API's MP3 response under a .wav name. "
            "This does not prove synthesis but breaks the claimed chain of custody.", +1, 0.35))
    if "tts" in kinds:
        h = next(h for h in sig_hits if h[3] == "tts")
        ranked.append(Finding(f"Synthesis software fingerprint: {h[0]} reads '{h[1]}' ({h[2]}). Metadata can be forged, but "
                              "genuine recordings do not normally carry a TTS product's name.", +1, 0.9))
    elif profiles:
        p, why = max(profiles)
        ranked.append(Finding(f"Output-profile match: {why}. This is a weak, circumstantial indicator of a TTS pipeline.", +1, p))
    if device:
        what = ", ".join(provenance) or next((f"{h[0]} '{h[1]}'" for h in sig_hits if h[3] == "device"), "device metadata")
        b = (riff or {}).get("bext") or {}
        extra = f" (originator '{b.get('originator')}', recorded {b.get('origination_date')} {b.get('origination_time')})" \
            if b.get("originator") or b.get("origination_date") else ""
        ranked.append(Finding(f"Recorder provenance present: {what}{extra}. Such records are written by recording hardware or "
                              "professional software and are rarely reproduced by TTS pipelines; they can be forged, so "
                              "treat as supporting evidence only.", -1, 0.4))
    if problems:
        ranked.append(Finding("Structural anomalies in the container: " + "; ".join(problems[:3]) + ". Inconsistent size "
                              "bookkeeping indicates the file was edited, spliced or hand-assembled after it was written.",
                              +1, min(0.5, 0.2 + 0.1 * len(problems))))
    if time_issues:
        ranked.append(Finding("Timestamp contradictions: " + "; ".join(time_issues[:2]) + ". Embedded dates that conflict with "
                              "the filesystem suggest metadata was rewritten or never came from a recorder.", +1, 0.25))
    if laundered:
        enc = next(h[1] for h in sig_hits if "FFmpeg" in h[2])
        ranked.append(Finding(f"Provenance laundered: the container was last written by FFmpeg ('{enc}') and carries no recorder, "
                              "device or date metadata. Any original provenance was stripped by that re-encode, so container "
                              "evidence is uninformative for this file (if a whole collection carries this identical header, "
                              "it was normalised by a single script).", 0, 0.0))
    elif sig_hits and not ({"tts", "device"} & kinds):
        tools = "; ".join(f"{h[0]} '{h[1]}' ({h[2]})" for h in sig_hits[:3])
        ranked.append(Finding(f"Software chain: {tools}. Conversion/editing tools say nothing about whether the voice is real, "
                              "only that the file is not a first-generation recording.", 0, 0.05))
    elif not sig_hits and not device and family == "wav":
        ranked.append(Finding("No encoder, software or recorder metadata at all (bare fmt+data WAV). Such files are typically "
                              "produced by programming libraries; this is neutral on its own.", 0, 0.05))

    ranked.sort(key=lambda fd: -fd.strength)
    ev.findings = [Finding(identity, 0, 0.0)] + ranked[:3]
    ev.score = None  # metadata is too easy to forge/strip to act as a standalone score; fusion weighs the features

    # ---- artifacts + cache ----------------------------------------------------------------------------
    ev.artifacts = {
        "sha256": sha, "file_size": size, "sniffed_format": family, "sniffed_description": family_desc,
        "extension": ext, "codec": codec, "stream": {"sr": native_sr, "channels": channels, "bits": bits,
                                                       "bitrate": bitrate, "duration": duration},
        "riff_chunks": riff["chunks"][:64] if riff else [], "riff_fmt": riff["fmt"] if riff else {},
        "riff_info": riff["info"] if riff else {}, "bext": riff.get("bext") if riff else {},
        "mutagen": mut, "mp3_header": mp3h, "exiftool": exif, "ffprobe_format_tags": fmt_tags,
        "encoder_strings": enc_strings, "signature_hits": sig_hits, "tts_profiles": profiles,
        "embedded_times": [(s, r, d.isoformat() if d else None) for s, r, d in etimes],
        "mac_times": {"modified": _utc(mtime), "inode_changed": _utc(ctime), "born": _utc(btime)},
        "problems": problems, "notes": notes,
    }
    ctx.cache["container"] = {"family": family, "codec": codec, "lossy": lossy, "native_sr": native_sr,
                              "channels": channels, "bitrate": bitrate, "duration": duration, "laundered": laundered,
                              "ext_mismatch": ext_mismatch, "sha256": sha}
    return ev
