#!/bin/bash

AUDIO="$1"

MODEL="large-v3-turbo-q5_0"
LANG="ca"
WHISPER="$HOME/compu/whisper.cpp/build/bin/whisper-cli"
MODEL_DIR="$HOME/compu/whisper.cpp/models"

BLOC=900   # 15 minuts = 900 segons

if [ -z "$AUDIO" ]; then
  echo "Ús: ./transcriu_blocs.sh audio.m4a"
  exit 1
fi

mkdir -p chunks

echo "✂️ Tallant en blocs de 15 minuts..."

ffmpeg -y -i "$AUDIO" \
  -f segment \
  -segment_time $BLOC \
  -ar 16000 -ac 1 \
  chunks/chunk_%03d.wav

for f in chunks/chunk_*.wav; do
  echo "🧠 Transcrivint $f..."

  "$WHISPER" \
    -m "$MODEL_DIR/ggml-${MODEL}.bin" \
    -f "$f" \
    -l "$LANG" \
    --output-srt

done

echo "✅ Tots els blocs transcrits. Revisa carpeta chunks/"
