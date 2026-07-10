"""
YouTube download helper (yt-dlp). Kept dependency-light: yt_dlp is imported lazily
so the rest of the app can load even before setup.sh installs it.
"""
import os
import glob


def _require_ytdlp():
    try:
        import yt_dlp  # noqa: F401
        return yt_dlp
    except Exception as e:
        raise RuntimeError(
            "yt-dlp chưa được cài. Chạy demo_app/setup.sh (hoặc `pip install yt-dlp`) trước."
        ) from e


def probe(url: str) -> dict:
    """Return {id, title, duration} without downloading."""
    yt_dlp = _require_ytdlp()
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "id": info.get("id", ""),
        "title": info.get("title", ""),
        "duration": info.get("duration", 0),
    }


def download_youtube(url: str, out_dir: str, max_height: int = 480, log=None) -> str:
    """Download ``url`` to ``out_dir`` as an mp4 named ``<video_id>.mp4``.

    Returns the local file path. The filename intentionally has no extra dots so that
    VideoRAG's ``os.path.basename(path).split('.')[0]`` yields a clean video_name.
    """
    yt_dlp = _require_ytdlp()
    os.makedirs(out_dir, exist_ok=True)

    def _hook(d):
        if log and d.get("status") == "downloading":
            pct = d.get("_percent_str", "").strip()
            spd = d.get("_speed_str", "").strip()
            log(f"[download] {pct} {spd}")
        elif log and d.get("status") == "finished":
            log("[download] merging / post-processing…")

    opts = {
        "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [_hook],
        "overwrites": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    vid = info.get("id", "")

    # Resolve the produced file (merge may leave .mp4; be defensive about extension).
    candidate = os.path.join(out_dir, f"{vid}.mp4")
    if os.path.exists(candidate):
        return candidate
    matches = sorted(glob.glob(os.path.join(out_dir, f"{vid}.*")))
    if not matches:
        raise RuntimeError(f"Tải xong nhưng không tìm thấy file video cho id={vid} trong {out_dir}")
    return matches[0]
