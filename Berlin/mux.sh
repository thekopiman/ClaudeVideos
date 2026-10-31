#!/bin/sh
# mux.sh VIDEO_NO_AUDIO.mp4 AUDIO.wav OUT.mp4 [DURATION_SECONDS]
# Adds the soundtrack, normalizes to -14 LUFS (true peak -1.5 dBTP), converts mono -> stereo, pads/trims to the picture length.
set -e
IN="$1"; WAV="$2"; OUT="$3"; DUR="${4:-}"
if [ -n "$DUR" ]; then T="-t $DUR"; else T="-shortest"; fi
ffmpeg -y -loglevel error -i "$IN" -i "$WAV" -c:v copy -c:a aac -b:a 192k -ac 2 \
  -af "loudnorm=I=-14:TP=-1.5:LRA=11,apad" $T -movflags +faststart "$OUT"
echo "wrote $OUT"
