#!/usr/bin/env bash
# Concatenate rendered defense scenes (all under media/videos/defense/<quality>/).
# Usage from paper/defense/manim after: ./render_defense full --quality qh
#   ./scripts/ffmpeg_concat_defense.sh qh
set -euo pipefail
QUALITY="${1:-qm}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT
DIR="$ROOT/media/videos/defense/$QUALITY"

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ "$line" =~ ^#.*$ || -z "${line// }" ]] && continue
  name="$line"
  f="$DIR/${name}.mp4"
  if [[ ! -f "$f" ]]; then
    echo "missing: $f" >&2
    exit 1
  fi
  echo "file '$f'" >>"$LIST"
done <"$ROOT/defense_scene_order.txt"

OUT="$ROOT/media/defense_rehearsal_${QUALITY}.mp4"
ffmpeg -y -f concat -safe 0 -i "$LIST" -c copy "$OUT"
echo "wrote $OUT"
