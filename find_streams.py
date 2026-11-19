"""Find current YouTube live streams for a query, ranked by YouTube view count.

Set YOUTUBE_API_KEY in the environment. The YouTube API does not provide this
search as a dependable live concurrent-viewer leaderboard.
"""
import argparse
import html
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://www.googleapis.com/youtube/v3/search"


def load_approved_channels(path):
    """Load exact YouTube channel IDs from a JSON allowlist."""
    with open(path, encoding="utf-8") as source:
        data = json.load(source)
    channel_ids = data.get("channel_ids", []) if isinstance(data, dict) else data
    if not isinstance(channel_ids, list):
        raise ValueError("Approved creator file must contain a channel_ids list.")
    cleaned = {str(channel_id).strip() for channel_id in channel_ids if str(channel_id).strip()}
    return cleaned


def find_streams(query, api_key, limit=25, approved_channel_ids=None):
    params = urllib.parse.urlencode({
        "part": "snippet", "eventType": "live", "type": "video", "q": query,
        "maxResults": min(max(limit, 1), 50), "order": "viewCount", "key": api_key,
    })
    request = urllib.request.Request(f"{API}?{params}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    streams = []
    for item in payload.get("items", []):
        snippet = item.get("snippet", {})
        video_id = item.get("id", {}).get("videoId")
        if not video_id:
            continue
        channel_id = snippet.get("channelId", "")
        if approved_channel_ids is not None and channel_id not in approved_channel_ids:
            continue
        streams.append({
            "title": html.unescape(snippet.get("title", "")),
            "channel": snippet.get("channelTitle", ""),
            "channel_id": channel_id,
            "published_at": snippet.get("publishedAt"),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "rank_signal": "YouTube search order=viewCount; not current live viewers",
        })
    return streams


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="GTA 6")
    parser.add_argument("--max", type=int, default=25, help="candidate count (maximum 50)")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--approved-channels", help="JSON allowlist; only these exact YouTube channel IDs are returned")
    parser.add_argument("--approved-only", action="store_true", help="require an allowlist and return only approved creators")
    args = parser.parse_args()
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        sys.exit("Set YOUTUBE_API_KEY in your environment before searching.")
    if args.approved_only and not args.approved_channels:
        parser.error("--approved-only requires --approved-channels.")
    try:
        approved = load_approved_channels(args.approved_channels) if args.approved_channels else None
        streams = find_streams(args.query, api_key, args.max, approved_channel_ids=approved)
    except urllib.error.HTTPError as error:
        sys.exit(f"YouTube API returned HTTP {error.code}; check the API key and quota.")
    except Exception as error:
        sys.exit(f"Stream search failed ({type(error).__name__}). Check the network and API key.")
    if args.json:
        print(json.dumps(streams, ensure_ascii=False, indent=2))
    else:
        for rank, stream in enumerate(streams, 1):
            print(f"[{rank}] {stream['channel']} — {stream['title']}\n    {stream['url']}")
            print("    Rank signal: YouTube view-count ordering; not current concurrent viewers.")
        if not streams:
            print("No live results found. Try another search phrase.")


if __name__ == "__main__":
    main()
