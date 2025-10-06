# Import necessary libraries
import os
import json
import re
import sys
from io import BytesIO
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bs4 import BeautifulSoup

import mutagen
import mutagen.id3
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, COMM, APIC, WOAS
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ----------------------------
# Utilities
# ----------------------------

def sanitize_filename(name: str, max_len: int = 140) -> str:
    """Return a filesystem-safe filename derived from a title.
    Removes/normalizes risky characters and trims length to avoid OS limits.
    """
    # Normalize whitespace
    name = re.sub(r"\s+", " ", name).strip()
    # Replace slashes and other reserved characters
    name = re.sub(r"[\\/\:*?\"<>|]", "-", name)
    # Remove control chars
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    # Trim length preserving extension to be added later
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    # Avoid empty
    return name or "episode"


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_history(history_file: str) -> dict:
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Support legacy list format -> convert to dict
            if isinstance(data, list):
                return {k: None for k in data}
            return dict(data)
        except Exception:
            return {}
    return {}


def save_history(history_file: str, history: dict) -> None:
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def new_session() -> requests.Session:
    """Create a requests session with a retry strategy and user-agent."""
    s = requests.Session()
    retries = Retry(
        total=5,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": "podcast-parser/1.0 (+https://example.local)"
    })
    return s


# ----------------------------
# Core logic
# ----------------------------

def extract_text(tag):
    return tag.get_text(strip=True) if tag else None


def parse_rfc822(dt: str):
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"):
        try:
            return datetime.strptime(dt, fmt)
        except Exception:
            pass
    return None


def guess_author(item, soup):
    # Prefer iTunes author if present
    itunes_author = item.find("itunes:author") or soup.find("itunes:author")
    if itunes_author and itunes_author.text.strip():
        return itunes_author.text.strip()
    # Fallback to <author>
    author = item.find("author")
    if author and author.text.strip():
        return author.text.strip()
    # Channel title as last resort
    channel_title = soup.find("title")
    return channel_title.text.strip() if channel_title else "Unknown Artist"


def episode_image_url(item, soup):
    # Episode image first
    img = item.find("itunes:image")
    if img and img.get("href"):
        return img["href"].strip()
    # Channel image fallback
    ch_img = soup.find("itunes:image") or soup.find("image")
    if ch_img and ch_img.get("href"):
        return ch_img["href"].strip()
    url_tag = soup.find("image")
    if url_tag and url_tag.find("url"):
        return url_tag.find("url").text.strip()
    return None


def add_id3_tags(mp3_path: str, *, title: str, artist: str, album: str,
                 date_text: str | None, link_url: str | None,
                 description: str | None, image_bytes: bytes | None) -> None:
    # Basic tags via EasyID3 (auto-creates header if missing)
    try:
        audio = EasyID3(mp3_path)
    except mutagen.id3.ID3NoHeaderError:
        audio = mutagen.File(mp3_path, easy=True)
        audio.add_tags()
    audio["title"] = title
    audio["artist"] = artist
    audio["album"] = album
    if date_text:
        # Attempt to format date to YYYY-MM-DD if possible
        try:
            dt = datetime.strptime(date_text.strip(), "%a, %d %b %Y %H:%M:%S %z")
            audio["date"] = dt.strftime("%Y-%m-%d")
        except Exception:
            audio["date"] = date_text.strip()
    audio.save()

    id3 = ID3(mp3_path)

    # Description (COMM)
    if description:
        # Remove existing COMM frames to avoid duplicates
        for frame in list(id3.getall("COMM")):
            id3.delall("COMM")
            break
        id3.add(COMM(encoding=3, lang="eng", desc="desc", text=description))

    # Official source URL frame (WOAS)
    if link_url:
        for frame in list(id3.getall("WOAS")):
            id3.delall("WOAS")
            break
        id3.add(WOAS(url=link_url))

    # Cover art (APIC)
    if image_bytes:
        try:
            # Validate image
            Image.open(BytesIO(image_bytes)).verify()
            # Remove existing APIC frames
            id3.delall("APIC")
            id3.add(APIC(
                encoding=3,
                mime="image/jpeg",
                type=3,  # front cover
                desc="Cover",
                data=image_bytes,
            ))
        except Exception:
            # Ignore invalid image data
            pass

    id3.save(v2_version=3)


def download_file(session: requests.Session, url: str, dest_path: str, *,
                  chunk_size: int = 1024 * 256, timeout: int = 30) -> None:
    # Write to temp then atomic rename with a progress bar
    tmp_path = dest_path + ".part"
    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path))
        try:
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        finally:
            pbar.close()
    os.replace(tmp_path, dest_path)


def download_podcast(rss_url, download_folder, history_file):
    session = new_session()

    # Fetch the RSS feed
    try:
        resp = session.get(rss_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch RSS feed: {e}")
        return

    # Parse the RSS feed using BeautifulSoup (XML)
    soup = BeautifulSoup(resp.content, "lxml-xml")
    items = soup.find_all("item")

    # Optional filters applied later via CLI args (since/limit) — set placeholders here
    # Actual values will be passed in via function closure (see process_item usage).

    ensure_dir(download_folder)
    history = load_history(history_file)

    channel_title = extract_text(soup.find("title")) or "Podcast"

    # Apply --since and --limit filters if provided via outer scope
    filtered = []
    if hasattr(download_podcast, "since_dt") and download_podcast.since_dt:
        since_dt = download_podcast.since_dt
    else:
        since_dt = None

    for item in items:
        pub_date = extract_text(item.find("pubDate"))
        if since_dt and pub_date:
            dt = parse_rfc822(pub_date)
            if dt and dt <= since_dt:
                continue
        filtered.append(item)

    if hasattr(download_podcast, "limit") and download_podcast.limit:
        filtered = filtered[: download_podcast.limit]

    items = filtered

    # Per-show subfolder
    channel_dir_name = sanitize_filename(channel_title, max_len=80)
    dest_dir = os.path.join(download_folder, channel_dir_name)
    ensure_dir(dest_dir)

    def process_item(item):
        # Create a per-thread session to avoid sharing sessions across threads
        sess = new_session()

        raw_title = extract_text(item.find("title")) or "episode"
        pub_date = extract_text(item.find("pubDate"))
        link_tag = extract_text(item.find("link"))
        guid_tag = extract_text(item.find("guid"))
        enclosure = item.find("enclosure")
        media_url = enclosure["url"].strip() if enclosure and enclosure.get("url") else None
        media_type = enclosure.get("type") if enclosure else None

        # Use GUID or fallback to media URL/title as key
        episode_key = guid_tag or media_url or raw_title
        if episode_key in history:
            print(f"Skipping already downloaded: {raw_title}")
            return None

        if not media_url:
            print(f"No media URL for: {raw_title} — skipping")
            return None

        if media_type and not media_type.startswith("audio/"):
            print(f"Non-audio enclosure for: {raw_title} ({media_type}) — skipping")
            return None

        # Build destination filename with date and episode number if available
        date_part = ""
        if pub_date:
            dt = parse_rfc822(pub_date)
            if dt:
                date_part = dt.strftime("%Y-%m-%d")

        ep_no = extract_text(item.find("itunes:episode")) or ""
        title_for_name = sanitize_filename(raw_title)
        pieces = [p for p in [date_part, ep_no, title_for_name] if p]
        base_name = " - ".join(pieces) if pieces else title_for_name
        filename = f"{base_name}.mp3"
        dest_path = os.path.join(dest_dir, filename)

        # Handle collisions
        if os.path.exists(dest_path):
            suffix = date_part or "dup"
            filename = f"{title_for_name} - {suffix}.mp3"
            dest_path = os.path.join(dest_dir, filename)

        print(f"Downloading: {raw_title}")
        try:
            download_file(sess, media_url, dest_path)
        except Exception as e:
            print(f"Failed to download '{raw_title}': {e}")
            try:
                if os.path.exists(dest_path + ".part"):
                    os.remove(dest_path + ".part")
            except Exception:
                pass
            return None

        # Prepare tags
        artist = guess_author(item, soup)
        description = extract_text(item.find("description")) or ""
        img_url = episode_image_url(item, soup)
        img_bytes = None
        if img_url:
            try:
                img_resp = sess.get(img_url, timeout=20)
                if img_resp.ok:
                    img_bytes = img_resp.content
            except Exception:
                pass

        try:
            add_id3_tags(
                dest_path,
                title=raw_title,
                artist=artist,
                album=channel_title,
                date_text=pub_date,
                link_url=link_tag,
                description=description,
                image_bytes=img_bytes,
            )
        except Exception as e:
            print(f"Warning: failed to tag '{raw_title}': {e}")

        print(f"Downloaded and tagged: {raw_title}\n")
        return (episode_key, os.path.basename(dest_path))

    # Parallel execution
    workers = getattr(download_podcast, "workers", 1)
    tasks = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for item in items:
            tasks.append(ex.submit(process_item, item))
        for fut in as_completed(tasks):
            res = fut.result()
            if res:
                k, fname = res
                history[k] = fname
                save_history(history_file, history)


# ----------------------------
# CLI entrypoint
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and tag podcast episodes from an RSS feed.")
    parser.add_argument("rss_url", nargs="?", default="https://feeds.simplecast.com/TBotaapn", help="Podcast RSS feed URL")
    parser.add_argument("--out", dest="download_folder", default="podcasts", help="Output folder for downloads")
    parser.add_argument("--history", dest="history_file", default="download_history.json", help="History JSON file path")
    parser.add_argument("--limit", type=int, default=None, help="Max episodes to download")
    parser.add_argument("--since", type=str, default=None, help="Only episodes after YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads")

    args = parser.parse_args()

    # Thread CLI parameters through the function via attributes
    download_podcast.limit = args.limit
    download_podcast.since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.since else None
    download_podcast.workers = max(1, args.workers)

    try:
        download_podcast(args.rss_url, args.download_folder, args.history_file)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
