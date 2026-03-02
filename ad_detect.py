#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from mutagen.id3 import ID3


AD_PATTERNS = [
    r"\bsponsor(?:ed|ship)?\b",
    r"\bbrought to you by\b",
    r"\bthanks to\b",
    r"\bsupport(?:ed)? by\b",
    r"\bpromo code\b",
    r"\buse code\b",
    r"\bvisit\b.{0,30}\bcom\b",
    r"\bslash\b",
    r"\bdiscount\b",
    r"\bfree trial\b",
    r"\bsubscribe\b",
    r"\bterms and conditions\b",
    r"\bvoid where prohibited\b",
    r"\baffiliate\b",
    r"\bthis episode is\b.{0,20}\bby\b",
]


@dataclass
class TranscriptSeg:
    start: float
    end: float
    text: str


@dataclass
class ScoredSeg:
    start: float
    end: float
    text: str
    score: float
    reasons: list[str]
    pattern_hits: int


def run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {out.stderr.strip()}")
    return out.stdout


def ffprobe_duration(audio_path: str) -> float:
    output = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
    ).strip()
    return float(output)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def count_pattern_hits(text: str) -> int:
    return sum(1 for p in AD_PATTERNS if re.search(p, text))


def load_whisper_model(model: str, compute_type: str, device: str):
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError(
            "faster-whisper is required. Install with `pip install faster-whisper`."
        ) from e
    return WhisperModel(model, device=device, compute_type=compute_type)


def transcribe(audio_path: str, wm) -> list[TranscriptSeg]:
    segments, _ = wm.transcribe(audio_path, vad_filter=True, beam_size=3)
    rows: list[TranscriptSeg] = []
    for s in segments:
        if s.end <= s.start:
            continue
        text = (s.text or "").strip()
        if not text:
            continue
        rows.append(TranscriptSeg(float(s.start), float(s.end), text))
    return rows


def transcribe_windows(audio_path: str, windows: list[tuple[float, float]], wm) -> list[TranscriptSeg]:
    rows: list[TranscriptSeg] = []
    for ws, we in windows:
        if we - ws < 0.4:
            continue
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_path = tf.name
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    f"{ws:.3f}",
                    "-to",
                    f"{we:.3f}",
                    "-i",
                    audio_path,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    tmp_path,
                ]
            )
            for s in transcribe(tmp_path, wm):
                rows.append(TranscriptSeg(s.start + ws, s.end + ws, s.text))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    return rows


def score_segments(segments: Iterable[TranscriptSeg], total_seconds: float) -> list[ScoredSeg]:
    scored: list[ScoredSeg] = []
    for s in segments:
        text_norm = normalize_text(s.text)
        dur = max(0.2, s.end - s.start)
        words = max(1, len(re.findall(r"\b[\w']+\b", text_norm)))
        wps = words / dur
        mid = (s.start + s.end) / 2.0
        pos = 0.0 if total_seconds <= 0 else mid / total_seconds

        hits = count_pattern_hits(text_norm)
        has_url_style = bool(re.search(r"\b[a-z0-9-]+\.(com|org|net|io|co)\b", text_norm))
        has_code = bool(re.search(r"\bcode\b.{0,15}\b[a-z0-9]{3,}\b", text_norm))
        cta = bool(re.search(r"\bvisit|download|sign up|subscribe|try\b", text_norm))

        pos_edge_score = 0.0
        if pos <= 0.12:
            pos_edge_score = clamp01((0.12 - pos) / 0.12)
        elif pos >= 0.88:
            pos_edge_score = clamp01((pos - 0.88) / 0.12)

        midroll_score = 1.0 if 0.35 <= pos <= 0.75 else 0.0
        fast_talk_score = clamp01((wps - 2.8) / 1.7)

        linear = (
            -2.0
            + 1.1 * min(3, hits)
            + 1.0 * (1 if has_url_style else 0)
            + 0.7 * (1 if has_code else 0)
            + 0.4 * (1 if cta else 0)
            + 0.7 * pos_edge_score
            + 0.5 * midroll_score
            + 0.4 * fast_talk_score
        )
        prob = sigmoid(linear)

        reasons: list[str] = []
        if hits:
            reasons.append(f"ad_phrases={hits}")
        if has_url_style:
            reasons.append("url_text")
        if has_code:
            reasons.append("promo_code_style")
        if pos_edge_score > 0.2:
            reasons.append("pre_or_post_roll_position")
        if midroll_score:
            reasons.append("midroll_position")
        if fast_talk_score > 0.3:
            reasons.append("fast_speech")

        scored.append(ScoredSeg(s.start, s.end, s.text, prob, reasons, hits))
    return scored


def load_audio_mono_f32(path: str, sr: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "-",
    ]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}: {p.stderr.decode(errors='ignore')}")
    data = np.frombuffer(p.stdout, dtype=np.float32)
    if data.size == 0:
        raise RuntimeError(f"No audio samples decoded for {path}")
    return data


def normalized_xcorr(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    n = signal.size
    m = kernel.size
    if n < m:
        return np.array([], dtype=np.float32)

    k = kernel.astype(np.float64)
    x = signal.astype(np.float64)
    k = k - k.mean()
    k_energy = np.sum(k * k)
    if k_energy <= 1e-12:
        raise RuntimeError("Cue segment has near-zero energy.")

    nfft = 1
    while nfft < (n + m - 1):
        nfft <<= 1

    x_fft = np.fft.rfft(x, nfft)
    k_rev_fft = np.fft.rfft(k[::-1], nfft)
    corr_full = np.fft.irfft(x_fft * k_rev_fft, nfft)
    corr = corr_full[m - 1 : m - 1 + (n - m + 1)]

    x2 = x * x
    win = np.ones(m, dtype=np.float64)
    e_full = np.fft.irfft(np.fft.rfft(x2, nfft) * np.fft.rfft(win, nfft), nfft)
    e = e_full[m - 1 : m - 1 + (n - m + 1)]

    denom = np.sqrt(np.maximum(e * k_energy, 1e-12))
    return (corr / denom).astype(np.float32)


def find_peaks(scores: np.ndarray, threshold: float, min_gap_samples: int) -> list[int]:
    peaks: list[int] = []
    i = 0
    n = scores.size
    while i < n:
        if scores[i] < threshold:
            i += 1
            continue
        j = min(n, i + max(1, min_gap_samples))
        peak = i + int(np.argmax(scores[i:j]))
        peaks.append(peak)
        i = peak + max(1, min_gap_samples)
    return peaks


def pair_boundaries(peaks_sec: list[float], cue_sec: float, max_gap_sec: float) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    i = 0
    while i + 1 < len(peaks_sec):
        start_cue = peaks_sec[i]
        end_cue = peaks_sec[i + 1]
        gap = end_cue - start_cue
        if 0 < gap <= max_gap_sec:
            cut_start = start_cue + cue_sec
            cut_end = end_cue
            if cut_end - cut_start > 0.5:
                windows.append((cut_start, cut_end))
            i += 2
        else:
            i += 1
    return windows


def windows_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a1 > b0 and b1 > a0


def cue_candidate_windows(
    audio_path: str,
    cue_path: str,
    *,
    sample_rate: int,
    cue_start: float,
    cue_length: float,
    cue_threshold: float,
    cue_min_peak_gap: float,
    cue_max_pair_gap: float,
) -> tuple[list[tuple[float, float]], list[float], float]:
    episode = load_audio_mono_f32(audio_path, sample_rate)
    cue_audio = load_audio_mono_f32(cue_path, sample_rate)
    cue_start_idx = int(max(0.0, cue_start) * sample_rate)
    cue_len_idx = int(max(0.2, cue_length) * sample_rate)
    cue_seg = cue_audio[cue_start_idx : cue_start_idx + cue_len_idx]
    if cue_seg.size < 800:
        raise RuntimeError("Cue segment too short after crop settings.")

    scores = normalized_xcorr(episode, cue_seg)
    min_gap_samples = int(max(1.0, cue_min_peak_gap) * sample_rate)
    peaks = find_peaks(scores, cue_threshold, min_gap_samples)
    peaks_sec = [p / sample_rate for p in peaks]
    cue_sec = cue_seg.size / sample_rate
    windows = merge_windows(pair_boundaries(peaks_sec, cue_sec, cue_max_pair_gap), gap=0.5)
    return windows, peaks_sec, cue_sec


def merge_windows(windows: list[tuple[float, float]], gap: float = 3.0) -> list[tuple[float, float]]:
    if not windows:
        return []
    windows = sorted(windows)
    merged = [windows[0]]
    for s, e in windows[1:]:
        ls, le = merged[-1]
        if s <= le + gap:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def invert_windows(total: float, cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in cuts:
        s = max(0.0, min(total, s))
        e = max(0.0, min(total, e))
        if s > cursor:
            keep.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < total:
        keep.append((cursor, total))
    return [k for k in keep if k[1] - k[0] > 0.25]


def ffmpeg_concat_filter(keep: list[tuple[float, float]]) -> str:
    parts = []
    n = len(keep)
    for i, (s, e) in enumerate(keep):
        parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
    labels = "".join(f"[a{i}]" for i in range(n))
    parts.append(f"{labels}concat=n={n}:v=0:a=1[outa]")
    return ";".join(parts)


def apply_cuts(audio_path: str, output_path: str, keep: list[tuple[float, float]]) -> None:
    if not keep:
        raise RuntimeError("No keep windows; refusing to write empty output.")
    filt = ffmpeg_concat_filter(keep)
    cmd_direct = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-filter_complex",
        filt,
        "-map",
        "[outa]",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        output_path,
    ]
    try:
        run(cmd_direct)
    except Exception:
        # Fallback for rare libmp3lame/MP3 frame edge cases:
        # render filtered audio to WAV first, then encode to MP3.
        tmp_wav = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_wav = tf.name
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    audio_path,
                    "-filter_complex",
                    filt,
                    "-map",
                    "[outa]",
                    "-c:a",
                    "pcm_s16le",
                    tmp_wav,
                ]
            )
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    tmp_wav,
                    "-codec:a",
                    "libmp3lame",
                    "-q:a",
                    "2",
                    output_path,
                ]
            )
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass
    copy_id3_tags(audio_path, output_path)


def copy_id3_tags(src_path: str, dst_path: str) -> None:
    """Copy all ID3 frames (including APIC artwork) from source to destination."""
    try:
        src = ID3(src_path)
    except Exception:
        return

    try:
        dst = ID3(dst_path)
    except Exception:
        dst = ID3()

    dst.clear()
    for frame in src.values():
        dst.add(copy.deepcopy(frame))
    dst.save(dst_path, v2_version=3)


def fmt_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect and remove likely podcast ad segments.")
    parser.add_argument("audio", help="Input audio file path")
    parser.add_argument("--output", default=None, help="Output cleaned audio path")
    parser.add_argument("--json-report", default=None, help="Write full detection report JSON path")
    parser.add_argument("--model", default="base", help="faster-whisper model size (tiny/base/small/...)")
    parser.add_argument("--compute-type", default="int8", help="faster-whisper compute type")
    parser.add_argument("--device", default="cpu", help="faster-whisper device")
    parser.add_argument("--threshold", type=float, default=0.72, help="Ad probability threshold [0-1]")
    parser.add_argument(
        "--verify-threshold",
        type=float,
        default=0.55,
        help="Verification threshold used for cue-derived candidate windows.",
    )
    parser.add_argument("--pad-seconds", type=float, default=0.7, help="Padding around detected cuts")
    parser.add_argument("--min-cut-seconds", type=float, default=8.0, help="Drop tiny ad windows")
    parser.add_argument("--cue", default=None, help="Cue audio that plays before and after ad blocks.")
    parser.add_argument(
        "--cue-is-ad-clip",
        action="store_true",
        help="Treat cue as the ad clip itself and cut each matched occurrence.",
    )
    parser.add_argument("--cue-threshold", type=float, default=0.62, help="Cue correlation threshold [0-1].")
    parser.add_argument("--cue-min-peak-gap", type=float, default=12.0, help="Min seconds between cue matches.")
    parser.add_argument("--cue-max-pair-gap", type=float, default=420.0, help="Max seconds between cue pair.")
    parser.add_argument("--cue-start", type=float, default=0.0, help="Cue crop start seconds.")
    parser.add_argument("--cue-length", type=float, default=4.0, help="Cue crop length seconds.")
    parser.add_argument("--cue-sample-rate", type=int, default=16000, help="Sample rate for cue matching.")
    parser.add_argument("--summary-only", action="store_true", help="Print only length summary.")
    parser.add_argument("--apply", action="store_true", help="Apply cuts and write output audio")
    args = parser.parse_args()

    audio_path = str(Path(args.audio).expanduser().resolve())
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return 2

    total = ffprobe_duration(audio_path)

    cue_matches: list[float] = []
    mode = "ml"
    if args.cue:
        cue_path = str(Path(args.cue).expanduser().resolve())
        if not os.path.exists(cue_path):
            print(f"Cue file not found: {cue_path}", file=sys.stderr)
            return 2

        mode = "cue+ml"
        candidates, cue_matches, cue_sec = cue_candidate_windows(
            audio_path,
            cue_path,
            sample_rate=args.cue_sample_rate,
            cue_start=args.cue_start,
            cue_length=args.cue_length,
            cue_threshold=args.cue_threshold,
            cue_min_peak_gap=args.cue_min_peak_gap,
            cue_max_pair_gap=args.cue_max_pair_gap,
        )
        if args.cue_is_ad_clip:
            candidates = merge_windows(
                [(s, min(total, s + cue_sec)) for s in cue_matches],
                gap=1.0,
            )
            cuts = [(s, e) for s, e in candidates if (e - s) >= float(args.min_cut_seconds)]
            scored: list[ScoredSeg] = []
        else:
            wm = load_whisper_model(args.model, args.compute_type, args.device)
            segments = transcribe_windows(audio_path, candidates, wm)
            scored = score_segments(segments, total)
            cuts = []
            for ws, we in candidates:
                overlapping = [s for s in scored if windows_overlap(s.start, s.end, ws, we)]
                best_score = max((s.score for s in overlapping), default=0.0)
                pattern_hits = sum(s.pattern_hits for s in overlapping)
                if best_score >= args.verify_threshold or pattern_hits > 0:
                    cuts.append((ws, we))
            cuts = [(s, e) for s, e in cuts if (e - s) >= float(args.min_cut_seconds)]
    else:
        wm = load_whisper_model(args.model, args.compute_type, args.device)
        segments = transcribe(audio_path, wm)
        scored = score_segments(segments, total)
        raw_cuts: list[tuple[float, float]] = []
        for s in scored:
            if s.score >= args.threshold:
                raw_cuts.append((max(0.0, s.start - args.pad_seconds), min(total, s.end + args.pad_seconds)))
        cuts = [
            (s, e)
            for s, e in merge_windows(raw_cuts, gap=3.0)
            if (e - s) >= float(args.min_cut_seconds)
        ]

    keep = invert_windows(total, cuts)

    report = {
        "audio": audio_path,
        "mode": mode,
        "duration_seconds": total,
        "threshold": args.threshold,
        "verify_threshold": args.verify_threshold,
        "cue_matches_seconds": cue_matches,
        "cut_windows": [{"start": s, "end": e, "duration": e - s} for s, e in cuts],
        "keep_windows": [{"start": s, "end": e, "duration": e - s} for s, e in keep],
        "high_score_segments": [
            {
                "start": s.start,
                "end": s.end,
                "score": round(s.score, 4),
                "reasons": s.reasons,
                "text": s.text,
            }
            for s in scored
            if s.score >= args.threshold
        ],
    }

    trimmed_estimate = sum((e - s) for s, e in keep)
    if not args.summary_only:
        print(f"Audio: {audio_path}")
        print(f"Duration: {fmt_time(total)}")
        print(f"Mode: {mode}")
        if cue_matches:
            print(f"Cue matches found: {len(cue_matches)}")
        print(f"Detected cut windows: {len(cuts)}")
        for i, (s, e) in enumerate(cuts, start=1):
            print(f"  {i}. {fmt_time(s)} -> {fmt_time(e)} ({e - s:.1f}s)")

    if args.json_report:
        report_path = str(Path(args.json_report).expanduser().resolve())
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        if not args.summary_only:
            print(f"Report: {report_path}")

    out_path = args.output
    if out_path is None:
        p = Path(audio_path)
        out_path = str(p.with_name(f"{p.stem}.cleaned{p.suffix}"))

    if args.apply:
        apply_cuts(audio_path, out_path, keep)
        final_trimmed = ffprobe_duration(out_path)
        print(f"Length: {total / 60.0:.2f} min -> {final_trimmed / 60.0:.2f} min")
    else:
        print(f"Length: {total / 60.0:.2f} min -> {trimmed_estimate / 60.0:.2f} min")
        if not args.summary_only:
            cmd_preview = (
                f"python3 ad_detect.py {shlex.quote(audio_path)} --apply --output {shlex.quote(out_path)} "
                f"--threshold {args.threshold} --pad-seconds {args.pad_seconds} --min-cut-seconds {args.min_cut_seconds}"
            )
            print("Dry run only. To apply cuts:")
            print(f"  {cmd_preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
