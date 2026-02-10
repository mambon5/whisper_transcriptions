#!/usr/bin/env python3
import os
import re

# ------------------------
# CONFIGURACIÓ
# ------------------------
CHUNK_DIR = "chunks"
FINAL_SRT = "final.srt"
FINAL_TEXT = "final_text.txt"
FINAL_PLAIN = "final_plain.txt"
BLOCK_SECONDS = 900           # 15 min per bloc
LINE_BREAK_SECONDS = 300      # 5 min per salt de línia
PLAIN_MAX_CHARS = 128         # màxim chars abans de salt de línia

# ------------------------
# Funcions utilitats
# ------------------------
def time_to_ms(t):
    h, m, s = t.split(":")
    s, ms = s.split(",")
    return (int(h)*3600 + int(m)*60 + int(s))*1000 + int(ms)

def ms_to_time(ms):
    s, ms = divmod(ms,1000)
    m, s = divmod(s,60)
    h, m = divmod(m,60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# ------------------------
# Preparar sortida
# ------------------------
chunks = sorted([f for f in os.listdir(CHUNK_DIR) if f.endswith(".srt")])
offset_ms = 0
index = 1
plain_buffer = ""
plain_accum_ms = 0

with open(FINAL_SRT, "w", encoding="utf-8") as out_srt, \
     open(FINAL_TEXT, "w", encoding="utf-8") as out_text, \
     open(FINAL_PLAIN, "w", encoding="utf-8") as out_plain:

    for chunk_file in chunks:
        path = os.path.join(CHUNK_DIR, chunk_file)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            if lines[i].strip().isdigit():
                # Numeració SRT
                out_srt.write(f"{index}\n")
                index += 1
                i += 1

                # Temps
                m = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", lines[i])
                if not m:
                    i += 1
                    continue
                start_ms = time_to_ms(m[1]) + offset_ms
                end_ms   = time_to_ms(m[2]) + offset_ms
                start_str = ms_to_time(start_ms)
                end_str   = ms_to_time(end_ms)
                out_srt.write(f"{start_str} --> {end_str}\n")
                i += 1

                # Línies de text
                text_lines = []
                while i < len(lines) and lines[i].strip() != "":
                    text_lines.append(lines[i].strip())
                    i += 1
                text_block = " ".join(text_lines)

                # Escriure SRT
                out_srt.write(text_block + "\n\n")

                # Escriure final_text.txt
                out_text.write(f"{start_str} --> {end_str}  {text_block}\n")

                # Escriure final_plain.txt
                plain_buffer += text_block + " "
                plain_accum_ms += (end_ms - start_ms)

                # Salt de línia cada 128 chars
                while len(plain_buffer) >= PLAIN_MAX_CHARS:
                    split_pos = plain_buffer.find(" ", PLAIN_MAX_CHARS)
                    if split_pos == -1:
                        split_pos = PLAIN_MAX_CHARS
                    out_plain.write(plain_buffer[:split_pos] + "\n")
                    plain_buffer = plain_buffer[split_pos+1:]

                # Salt doble si punt i ja han passat 5 min
                if "." in text_block and plain_accum_ms >= LINE_BREAK_SECONDS*1000:
                    out_plain.write("\n\n")
                    plain_accum_ms = 0
            else:
                i += 1

        # Incrementar offset pel següent bloc
        offset_ms += BLOCK_SECONDS * 1000

    # escriure qualsevol residu
    if plain_buffer.strip():
        out_plain.write(plain_buffer.strip() + "\n")

print(f"✅ Fitxers generats: {FINAL_SRT}, {FINAL_TEXT}, {FINAL_PLAIN}")
