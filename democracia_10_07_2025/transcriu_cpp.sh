#!/bin/bash

# --- CONFIGURACIÓ ---
AUDIO="$1"
MODEL="small"  # Pots canviar a tiny, base, small, large
LANG="ca"       # Català

# --- VALIDACIONS ---
if [ -z "$AUDIO" ]; then
  echo "❌ Has d'indicar un fitxer d'àudio."
  echo "Exemple: ./transcriu_cpp.sh audio.mp3"
  exit 1
fi

# --- COMPROVAR ffmpeg ---
if ! command -v ffmpeg &> /dev/null
then
    echo "❌ No tens instal·lat 'ffmpeg'. Instal·la-ho amb:"
    echo "   sudo apt install ffmpeg"
    exit 1
fi

# --- DIRECTORIS I FITXERS ---
WHISPER_DIR="/home/romanoide/compu/whisper.cpp/build/bin"
MODEL_PATH="/home/romanoide/compu/whisper.cpp/models/ggml-${MODEL}.bin"

if [ ! -f "$WHISPER_DIR/whisper-cli" ]; then
  echo "❌ No trobo l'executable 'whisper-cli' a $WHISPER_DIR"
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "❌ No trobo el model a $MODEL_PATH"
  exit 1
fi

# --- FITXER TEMPORAL ---
WAV="temp_input.wav"

echo "🔄 Convertint àudio a format compatible (.wav mono, 16kHz)..."
ffmpeg -y -i "$AUDIO" -ar 16000 -ac 1 "$WAV"

echo "🧠 Transcrivint amb whisper.cpp i model '$MODEL'..."
"$WHISPER_DIR/whisper-cli" -m "$MODEL_PATH" -f "$WAV" -l "$LANG" --output-srt

# --- RENOMENAR SORTIDA ---
SRT_OUT="${AUDIO%.*}.srt"
if [ -f "temp_input.wav.srt" ]; then
  mv "temp_input.wav.srt" "$SRT_OUT"
  echo "✅ Subtítols generats: $SRT_OUT"
else
  echo "⚠️ No s'ha generat el fitxer de subtítols"
fi

# --- NETEJA ---
rm "$WAV"
