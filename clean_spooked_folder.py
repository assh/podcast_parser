#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def ffprobe_duration(path: str) -> float:
    p = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {p.stderr.strip()}")
    return float(p.stdout.strip())


def is_episode_file(path: Path) -> bool:
    if path.suffix.lower() != ".mp3":
        return False
    name = path.name.lower()
    if name.endswith(".cleaned.mp3") or name.endswith(".adclipcut.mp3") or name.endswith(".adfree.mp3"):
        return False
    return True


def process_file(
    src_path: str,
    detector_path: str,
    adclip_path: str,
    cue_length: float,
    cue_threshold: float,
) -> tuple[str, str, str]:
    src = Path(src_path)
    root, ext = os.path.splitext(str(src))
    tmp_out = f"{root}.adfree{ext or '.mp3'}"
    cmd = [
        sys.executable,
        detector_path,
        str(src),
        "--apply",
        "--summary-only",
        "--cue",
        adclip_path,
        "--cue-is-ad-clip",
        "--cue-length",
        str(cue_length),
        "--cue-threshold",
        str(cue_threshold),
        "--verify-threshold",
        "0.0",
        "--output",
        tmp_out,
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        try:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass
        stderr = (p.stderr or "").strip()
        return ("fail", src.name, stderr or "ad_detect failed")

    try:
        os.replace(tmp_out, str(src))
        stdout = (p.stdout or "").strip()
        return ("ok", src.name, stdout)
    except Exception as e:
        try:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass
        return ("fail", src.name, f"replace failed: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove repeated ad-clip occurrences from all MP3 files in a folder and replace originals."
    )
    parser.add_argument(
        "--folder",
        default="podcasts/Spooked",
        help="Folder containing downloaded Spooked episodes.",
    )
    parser.add_argument(
        "--adclip",
        default="/Users/asishpanda/Downloads/Adclip.mp3",
        help="Path to ad clip used for match-and-cut.",
    )
    parser.add_argument("--cue-threshold", type=float, default=0.55, help="Cue correlation threshold.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel files to process.")
    parser.add_argument("--workers-note", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    adclip = Path(args.adclip).expanduser().resolve()
    detector = Path(__file__).with_name("ad_detect.py").resolve()

    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 2
    if not adclip.exists():
        print(f"Ad clip not found: {adclip}", file=sys.stderr)
        return 2
    if not detector.exists():
        print(f"ad_detect.py not found: {detector}", file=sys.stderr)
        return 2

    cue_length = ffprobe_duration(str(adclip))
    files = sorted([p for p in folder.iterdir() if p.is_file() and is_episode_file(p)])
    if not files:
        print(f"No episode MP3 files found in {folder}")
        return 0

    print(f"Processing {len(files)} files in {folder}")
    print(f"Using ad clip: {adclip} ({cue_length:.2f}s)\n")

    ok = 0
    failed = 0

    workers = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                process_file,
                str(src),
                str(detector),
                str(adclip),
                cue_length,
                float(args.cue_threshold),
            )
            for src in files
        ]
        for fut in as_completed(futs):
            status, name, message = fut.result()
            if status == "ok":
                ok += 1
                print(f"[OK] {name}: {message}" if message else f"[OK] {name}")
            else:
                failed += 1
                print(f"[FAIL] {name}: {message}")

    print(f"\nDone. Updated: {ok}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
