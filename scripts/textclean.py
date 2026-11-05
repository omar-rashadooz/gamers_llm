"""Line-level text cleaning shared by fetch_wiki.py and filter_text.py.

Problem: game articles are full of Japanese / Korean / Russian text, e.g.
    "Final Fantasy (ファイナルファンタジー, Fainaru Fantajī) is a media franchise ..."
Our tokenizer works on bytes, so these characters waste vocabulary slots and the
model learns to babble in broken bytes. We keep English (plus accented Latin
letters like the é in Pokémon) and drop the rest.
"""
import re

# fancy punctuation -> plain ASCII (fewer distinct symbols for a tiny model)
_TRANS = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
    "…": "...", "×": "x",
})

# characters we accept: Latin (up to U+024F) and general punctuation (U+2000-206F)
_FOREIGN = re.compile(r"[^\x00-ɏ -⁯]")
# a (...) group that contains at least one foreign character, e.g. a Japanese title
_FOREIGN_PAREN = re.compile(r"\s*\([^()]*[^\x00-ɏ -⁯][^()]*\)")
_EMPTY_PAREN = re.compile(r"\s*\(\s*[;,]?\s*\)")


def clean_line(s):
    """Return the cleaned line, or None if the line should be dropped."""
    s = s.translate(_TRANS)
    s = _FOREIGN_PAREN.sub("", s)          # remove "(ファイナル..., Fainaru ...)"
    s = _EMPTY_PAREN.sub("", s)
    if _FOREIGN.search(s):                 # foreign text outside parentheses: drop the line
        return None
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s or None
