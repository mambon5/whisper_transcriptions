#!/usr/bin/env python3
import os
import re

# ------------------------
# CONFIGURACIÓ
# ------------------------
CHUNK_DIR = "chunks"      # directori on estan els .srt dels blocs
FINAL_SRT = "final.srt"   # fitxer SRT final
BLOCK_SECONDS = 900        # durada de cada bloc en segons (15 min)

# ------------------------
# Funcions utilitats
# ------------------------
print("inici d'unio de fitxers srt")

def time_to_ms(t):
    """Converteix 'HH:MM:SS,mmm' a mil·lisegons"""
    h, m, s = t.split(":")
    s, ms = s.split(",")
    return (int(h)*3600 + int(m)*60 + int(s))*1000 + int(ms)

def ms_to_time(ms):
    """ Converteix mil·lisegons a 'HH:MM:SS,mmm' """
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# ------------------------
# Llistar i ordenar blocs
# ------------------------
chunks = sorted([f for f in os.listdir(CHUNK_DIR) if f.endswith(".srt")])
offset_ms = 0
index = 1

with open(FINAL_SRT, "w", encoding="utf-8") as out:
    for chunk_file in chunks:
        path = os.path.join(CHUNK_DIR, chunk_file)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            # línia de número
            if lines[i].strip().isdigit():
                out.write(f"{index}\n")
                index += 1
                i += 1

                # línia de temps
                m = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", lines[i])
                if m:
                    start = time_to_ms(m[1]) + offset_ms
                    end   = time_to_ms(m[2]) + offset_ms
                    out.write(f"{ms_to_time(start)} --> {ms_to_time(end)}\n")
                    i += 1

                    # línies de text
                    while i < len(lines) and lines[i].strip() != "":
                        out.write(lines[i])
                        i += 1
                    out.write("\n")
            else:
                i += 1

        # incrementar offset pel següent bloc
        offset_ms += BLOCK_SECONDS * 1000

print(f"✅ SRT final creat: {FINAL_SRT}")
