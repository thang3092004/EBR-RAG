#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DENO="$ROOT/.tools/deno/bin/deno"

if [[ ! -x "$DENO" ]]; then
  echo "Missing Deno JavaScript runtime: $DENO" >&2
  echo "Install it with:" >&2
  echo "  curl -fsSL https://deno.land/install.sh -o /tmp/deno-install.sh" >&2
  echo "  DENO_INSTALL=$ROOT/.tools/deno sh /tmp/deno-install.sh --no-modify-path" >&2
  exit 1
fi

cd "$ROOT/longervideos"

if [[ ! -x "$ROOT/.venv/bin/yt-dlp" ]]; then
  echo "Missing yt-dlp in .venv." >&2
  exit 1
fi

"$ROOT/.venv/bin/python" prepare_data.py

for collection in \
  0-fights-in-animal-kingdom \
  6-daubechies-wavelet-lecture \
  11-primetime-emmy-awards
do
  mkdir -p "$collection/videos"
  "$ROOT/.venv/bin/yt-dlp" \
    --js-runtimes "deno:$DENO" \
    --remote-components ejs:github \
    --no-playlist \
    --progress \
    -f "bv*[vcodec^=avc1][height<=720]+ba[acodec^=mp4a]" \
    --merge-output-format mp4 \
    -o "%(id)s.%(ext)s" \
    -a "$collection/videos.txt" \
    -P "$collection/videos"
done

"$ROOT/.venv/bin/python" "$ROOT/scripts/verify_minimal_ablation_data.py"

echo "Collections 0, 6, and 11 are downloaded."
