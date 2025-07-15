import sys
import os
from langdetect import detect
import re

def detect_language(text):
    try:
        lang = detect(text)
        if lang == "ca":
            return "[CAT]"
        elif lang == "es":
            return "[ES]"
        else:
            return "[??]"
    except:
        return "[??]"

if len(sys.argv) < 2:
    print("Ús: python script.py fitxer.srt")
    sys.exit(1)

input_file = sys.argv[1]
base, ext = os.path.splitext(input_file)
output_file = f"{base}_amb_idioma{ext}"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        if re.match(r"^\d+$", line.strip()) or "-->" in line or line.strip() == "":
            outfile.write(line)
        else:
            lang_tag = detect_language(line.strip())
            if not line.startswith(lang_tag):
                line = f"{lang_tag} {line}"
            outfile.write(line)
