#!/usr/bin/env python3
import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

from mutagen.id3 import ID3


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "command failed")


def copy_id3_tags(src_path: str, dst_path: str) -> None:
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


def is_episode_mp3(path: Path) -> bool:
    if path.suffix.lower() != ".mp3":
        return False
    name = path.name.lower()
    if name.endswith(".cleaned.mp3") or name.endswith(".adclipcut.mp3") or name.endswith(".adfree.mp3"):
        return False
    return True


def bytes_to_mb(n: int) -> float:
    return n / (1024.0 * 1024.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Shrink MP3 files in place (lower bitrate/sample-rate) while preserving ID3 metadata/art."
    )
    parser.add_argument(
        "--folder",
        default="podcasts/Spooked",
        help="Folder with MP3 files to shrink.",
    )
    parser.add_argument(
        "--bitrate",
        default="64k",
        help="Target audio bitrate (e.g. 48k, 64k, 80k).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target sample rate in Hz.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 2),
        default=1,
        help="Target channels. 1=mono, 2=stereo.",
    )
    parser.add_argument(
        "--min-size-mb",
        type=float,
        default=0.0,
        help="Skip files smaller than this size in MB.",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 2

    files = sorted([p for p in folder.iterdir() if p.is_file() and is_episode_mp3(p)])
    if not files:
        print(f"No MP3 files found in {folder}")
        return 0

    print(
        f"Shrinking {len(files)} files in {folder} "
        f"(bitrate={args.bitrate}, sample_rate={args.sample_rate}, channels={args.channels})\n"
    )

    updated = 0
    skipped = 0
    failed = 0

    for src in files:
        before_bytes = src.stat().st_size
        if bytes_to_mb(before_bytes) < args.min_size_mb:
            skipped += 1
            print(f"[SKIP] {src.name}: {bytes_to_mb(before_bytes):.2f} MB < {args.min_size_mb:.2f} MB")
            continue

        root, ext = os.path.splitext(str(src))
        tmp_out = f"{root}.small{ext}"
        try:
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(src),
                    "-map",
                    "0:a:0",
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    args.bitrate,
                    "-ar",
                    str(args.sample_rate),
                    "-ac",
                    str(args.channels),
                    tmp_out,
                ]
            )
            copy_id3_tags(str(src), tmp_out)
            os.replace(tmp_out, str(src))
            after_bytes = src.stat().st_size
            updated += 1
            print(
                f"[OK] {src.name}: {bytes_to_mb(before_bytes):.2f} MB -> {bytes_to_mb(after_bytes):.2f} MB"
            )
        except Exception as e:
            failed += 1
            print(f"[FAIL] {src.name}: {e}")
            try:
                if os.path.exists(tmp_out):
                    os.remove(tmp_out)
            except Exception:
                pass

    print(f"\nDone. Updated: {updated}, Skipped: {skipped}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

