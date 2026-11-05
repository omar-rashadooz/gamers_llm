"""One-time clean-up of the wiki.txt you already downloaded.

    python scripts/filter_text.py

Applies textclean.clean_line to every line, drops documents that became too
short, and keeps a backup of the original as data/raw/wiki.unfiltered.txt.
(Pages fetched from now on are cleaned while downloading, so run this ONCE.)
"""
import sys
from pathlib import Path

from textclean import clean_line

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
EOT = "<|endoftext|>"


def non_ascii(text):
    return sum(ord(c) > 127 for c in text)


def main():
    src = RAW / "wiki.txt"
    bak = RAW / "wiki.unfiltered.txt"
    if bak.exists():
        sys.exit(f"{bak.name} already exists: the filter was already run.\n"
                 "Delete the backup first if you really want to run it again.")
    text = src.read_text(encoding="utf-8")

    docs_in = docs_out = lines_in = lines_out = 0
    kept = []
    for doc in text.split(EOT):
        doc = doc.strip()
        if not doc:
            continue
        docs_in += 1
        lines = []
        for line in doc.split("\n"):
            lines_in += 1
            c = clean_line(line)
            if c is not None and len(c) >= 40:
                lines.append(c)
                lines_out += 1
        out = "\n".join(lines)
        if len(out) >= 500:
            kept.append(out)
            docs_out += 1

    result = "".join(d + "\n" + EOT + "\n" for d in kept)
    src.rename(bak)
    src.write_text(result, encoding="utf-8")

    print(f"documents: {docs_in:,} -> {docs_out:,}")
    print(f"lines:     {lines_in:,} -> {lines_out:,}")
    print(f"characters: {len(text):,} -> {len(result):,}")
    print(f"non-ASCII characters: {non_ascii(text):,} -> {non_ascii(result):,}")
    print(f"original saved as {bak.name}")


if __name__ == "__main__":
    main()
