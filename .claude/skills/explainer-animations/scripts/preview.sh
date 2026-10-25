#!/bin/bash
# Render a scene at 480p and tile chosen timestamps into one contact sheet for review.
#   preview.sh <file.py> <Scene> <cols>x<rows> <t1> [t2 ...]
#   env: MANIM (manim binary, default "manim"), MEDIA (render folder, default /tmp/explainer-media),
#        SHEET (output image, default $MEDIA/sheet.png)
set -u
MANIM=${MANIM:-manim}; MEDIA=${MEDIA:-/tmp/explainer-media}; SHEET=${SHEET:-$MEDIA/sheet.png}
file=$1; scene=$2; tiles=$3; shift 3
out=$(COLUMNS=200 "$MANIM" -ql --disable_caching --media_dir "$MEDIA" "$file" "$scene" 2>&1)
if echo "$out" | grep -qE "Error|Traceback"; then
  echo "render failed:"; echo "$out" | grep -E "Error|$(basename "$file"):[0-9]+" | head -8; exit 1
fi
video="$MEDIA/videos/$(basename "${file%.py}")/480p30/$scene.mp4"
echo "duration $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$video") s"
frames=$(mktemp -d)
i=0
for t in "$@"; do i=$((i + 1)); ffmpeg -v error -ss "$t" -i "$video" -frames:v 1 "$frames/f_$i.png"; done
ffmpeg -v error -y -i "$frames/f_%d.png" -filter_complex "tile=$tiles:padding=6:color=white" -frames:v 1 "$SHEET"
rm -rf "$frames"
echo "sheet $SHEET"
