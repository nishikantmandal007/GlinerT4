#!/bin/bash
# Render a scene at 1080p and export the MP4 plus an 640 px, 12 fps GIF into <out_dir>.
#   export.sh <file.py> <Scene> <out_dir>
#   env: MANIM (manim binary, default "manim"), MEDIA (render folder, default /tmp/explainer-media)
set -eu
MANIM=${MANIM:-manim}; MEDIA=${MEDIA:-/tmp/explainer-media}
file=$1; scene=$2; out=$3; name=$(basename "${file%.py}")
mkdir -p "$out"
"$MANIM" -qh --disable_caching --media_dir "$MEDIA" "$file" "$scene" 2>&1 | grep -E "Error|Traceback" && exit 1
video="$MEDIA/videos/$name/1080p30/$scene.mp4"
cp "$video" "$out/$name.mp4"
ffmpeg -v error -y -i "$video" -vf "fps=12,scale=640:-1:flags=lanczos,palettegen=stats_mode=diff" "$MEDIA/palette.png"
ffmpeg -v error -y -i "$video" -i "$MEDIA/palette.png" \
  -lavfi "fps=12,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  "$out/$name.gif"
ls -la "$out/$name.mp4" "$out/$name.gif"
