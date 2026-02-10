#!/bin/bash

# --- CONFIGURACIÓ ---
AUDIO="$1"
OPTION="$2"

MODEL="small"
MODEL2="small.en-tdrz" # per diaritzacio
LANG="ca"

DIARIZE=false

if [ "$OPTION" = "-diaritzacio" ]; then
  DIARIZE=true
  MODEL="$MODEL2" # canviem model a small.en-tdrz
fi

# --- VALIDACIONS ---
if [ -z "$AUDIO" ]; then
  echo "❌ Has d'indicar un fitxer d'àudio."
  echo "Exemple:"
  echo "  ./transcriu_cpp.sh audio.m4a"
  echo "  ./transcriu_cpp.sh audio.m4a -diaritzacio"
  exit 1
fi

# --- COMPROVAR ffmpeg ---
if ! command -v ffmpeg &> /dev/null
then
  echo "❌ No tens instal·lat 'ffmpeg'."
  exit 1
fi

# --- DIRECTORIS ---
WHISPER_DIR="$HOME/compu/whisper.cpp/build/bin"
MODEL_PATH="$HOME/compu/whisper.cpp/models/ggml-${MODEL}.bin"

if [ ! -f "$WHISPER_DIR/whisper-cli" ]; then
  echo "❌ No trobo whisper-cli a $WHISPER_DIR"
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "❌ No trobo el model a $MODEL_PATH"
  exit 1
fi

# --- FITXER TEMPORAL ---
WAV="temp_input.wav"


echo "🔄 Convertint àudio a WAV 16kHz..."
ffmpeg -y -i "$AUDIO" -ar 16000 -ac 1 "$WAV"

# --- TRANSCRIPCIÓ ---
CMD=(
  "$WHISPER_DIR/whisper-cli"
  -m "$MODEL_PATH"
  -f "$WAV"
  -l "$LANG"
  --output-srt
)

if [ "$DIARIZE" = true ]; then
  echo "🧠 Diarització activada"
fi

"${CMD[@]}"

# --- RENOMENAR SORTIDA ---
SRT_OUT="${AUDIO%.*}.srt"

if [ -f "temp_input.wav.srt" ]; then
  mv "temp_input.wav.srt" "$SRT_OUT"
  echo "✅ Subtítols generats: $SRT_OUT"
else
  echo "⚠️ No s'ha generat el fitxer .srt"
fi

# --- NETEJA ---
rm "$WAV"
