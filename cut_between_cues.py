#!/usr/bin/env python3
import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {p.stderr.strip()}")
    return p.stdout


def ffprobe_duration(path: str) -> float:
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            path,
        ]
    ).strip()
    return float(out)


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

    # Local energy for normalization.
    x2 = x * x
    win = np.ones(m, dtype=np.float64)
    e_full = np.fft.irfft(np.fft.rfft(x2, nfft) * np.fft.rfft(win, nfft), nfft)
    e = e_full[m - 1 : m - 1 + (n - m + 1)]

    denom = np.sqrt(np.maximum(e * k_energy, 1e-12))
    score = corr / denom
    return score.astype(np.float32)


def find_peaks(scores: np.ndarray, threshold: float, min_gap_samples: int) -> list[int]:
    peaks: list[int] = []
    i = 0
    n = scores.size
    while i < n:
        if scores[i] < threshold:
            i += 1
            continue
        j = min(n, i + max(1, min_gap_samples))
        local_idx = int(np.argmax(scores[i:j]))
        peak = i + local_idx
        peaks.append(peak)
        i = peak + max(1, min_gap_samples)
    return peaks


def pair_boundaries(peaks_sec: list[float], cue_sec: float, max_gap_sec: float) -> list[tuple[float, float]]:
    cuts: list[tuple[float, float]] = []
    i = 0
    while i + 1 < len(peaks_sec):
        start_cue = peaks_sec[i]
        end_cue = peaks_sec[i + 1]
        gap = end_cue - start_cue
        if gap <= 0:
            i += 1
            continue
        if gap <= max_gap_sec:
            cut_start = start_cue + cue_sec
            cut_end = end_cue
            if cut_end - cut_start > 0.5:
                cuts.append((cut_start, cut_end))
            i += 2
        else:
            i += 1
    return cuts


def merge_intervals(intervals: list[tuple[float, float]], gap: float = 0.5) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le + gap:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def invert_intervals(total: float, cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    keep: list[tuple[float, float]] = []
    cur = 0.0
    for s, e in cuts:
        s = max(0.0, min(total, s))
        e = max(0.0, min(total, e))
        if s > cur:
            keep.append((cur, s))
        cur = max(cur, e)
    if cur < total:
        keep.append((cur, total))
    return [(s, e) for s, e in keep if e - s > 0.2]


def ffmpeg_concat_filter(keep: list[tuple[float, float]]) -> str:
    parts = []
    n = len(keep)
    for i, (s, e) in enumerate(keep):
        parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
    labels = "".join(f"[a{i}]" for i in range(n))
    parts.append(f"{labels}concat=n={n}:v=0:a=1[outa]")
    return ";".join(parts)


def apply_cuts(input_audio: str, output_audio: str, keep: list[tuple[float, float]]) -> None:
    if not keep:
        raise RuntimeError("No keep windows left after cuts.")
    filt = ffmpeg_concat_filter(keep)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_audio,
            "-filter_complex",
            filt,
            "-map",
            "[outa]",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            output_audio,
        ]
    )


def fmt(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60.0
    if h:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Cut audio between repeated cue sound occurrences.")
    ap.add_argument("episode", help="Episode audio file to clean")
    ap.add_argument("--cue", required=True, help="Cue audio that appears before and after ads")
    ap.add_argument("--output", default=None, help="Output cleaned file")
    ap.add_argument("--json-report", default=None, help="Write detection details JSON")
    ap.add_argument("--sample-rate", type=int, default=16000, help="Working sample rate")
    ap.add_argument("--cue-start", type=float, default=0.0, help="Cue crop start seconds")
    ap.add_argument("--cue-length", type=float, default=4.0, help="Cue crop length seconds")
    ap.add_argument("--threshold", type=float, default=0.62, help="Normalized correlation threshold")
    ap.add_argument("--min-peak-gap", type=float, default=12.0, help="Min seconds between cue matches")
    ap.add_argument("--max-pair-gap", type=float, default=420.0, help="Max seconds between start/end cue pair")
    ap.add_argument("--apply", action="store_true", help="Write cleaned output")
    args = ap.parse_args()

    episode = str(Path(args.episode).expanduser().resolve())
    cue = str(Path(args.cue).expanduser().resolve())
    out = args.output
    if out is None:
        p = Path(episode)
        out = str(p.with_name(f"{p.stem}.cuecut{p.suffix}"))

    ep = load_audio_mono_f32(episode, args.sample_rate)
    cue_arr = load_audio_mono_f32(cue, args.sample_rate)
    cue_start_idx = int(max(0.0, args.cue_start) * args.sample_rate)
    cue_len_idx = int(max(0.2, args.cue_length) * args.sample_rate)
    cue_seg = cue_arr[cue_start_idx : cue_start_idx + cue_len_idx]
    if cue_seg.size < 800:
        raise RuntimeError("Cue segment too short after crop settings.")

    scores = normalized_xcorr(ep, cue_seg)
    min_gap_samples = int(max(1.0, args.min_peak_gap) * args.sample_rate)
    peaks = find_peaks(scores, args.threshold, min_gap_samples)
    peaks_sec = [p / args.sample_rate for p in peaks]
    cue_sec = cue_seg.size / args.sample_rate
    cuts = merge_intervals(pair_boundaries(peaks_sec, cue_sec, args.max_pair_gap), gap=0.5)

    duration = ffprobe_duration(episode)
    keep = invert_intervals(duration, cuts)

    print(f"Episode: {episode}")
    print(f"Cue: {cue}")
    print(f"Cue matches found: {len(peaks_sec)}")
    print(f"Ad windows to cut: {len(cuts)}")
    for i, (s, e) in enumerate(cuts, start=1):
        print(f"  {i}. {fmt(s)} -> {fmt(e)} ({e - s:.1f}s)")

    report = {
        "episode": episode,
        "cue": cue,
        "threshold": args.threshold,
        "cue_matches_seconds": peaks_sec,
        "cut_windows": [{"start": s, "end": e, "duration": e - s} for s, e in cuts],
        "keep_windows": [{"start": s, "end": e, "duration": e - s} for s, e in keep],
    }
    if args.json_report:
        rp = str(Path(args.json_report).expanduser().resolve())
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report: {rp}")

    if args.apply:
        apply_cuts(episode, out, keep)
        print(f"Wrote cleaned output: {out}")
    else:
        print("Dry run. To apply:")
        print(
            "  "
            + " ".join(
                [
                    "python3",
                    "cut_between_cues.py",
                    shlex.quote(episode),
                    "--cue",
                    shlex.quote(cue),
                    "--apply",
                    "--output",
                    shlex.quote(out),
                    "--threshold",
                    str(args.threshold),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

