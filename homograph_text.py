"""Shared text helpers for the homograph pipeline (no torch needed).

Everything here works on *character spans* so that the runtime can change only the
homograph letters of the original text and leave every other character untouched.
"""
import re

ZWNJ = "‌"
KASRA, HAMZA_ABOVE = "ِ", "ٔ"
DIAC_RE = re.compile("[ً-ٰٟـ]")        # harakat, hamza-above, superscript alef, tatweel
EDGE_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)             # punctuation around a token
_ARABIC = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "_": ZWNJ})   # 1:1 character mapping
_KEY = str.maketrans({"آ": "ا", "أ": "ا", "إ": "ا", "ة": "ه"})       # matching ignores these variants (1:1)


def strip_diacritics(s):
    return DIAC_RE.sub("", s)


def norm_chars(s):
    """Normalise letters 1:1 (no length change). Diacritics must already be absent."""
    return s.translate(_ARABIC)


def normalize(s):
    """Same normalisation the classifier was trained with."""
    return re.sub(r"\s+", " ", norm_chars(strip_diacritics(str(s)))).strip()


def match_key(s):
    return s.translate(_KEY)


def core_span(token):
    """(start, end) of the token without leading/trailing punctuation."""
    m = EDGE_RE.match(token)
    return m.start(2), m.end(2)


def match_score(core, h):
    """0 exact, 1 ZWNJ-part exact, 2 startswith, 3 ZWNJ-part startswith, 4 endswith, 5 contained."""
    core, h = match_key(core), match_key(h)
    parts = core.split(ZWNJ)
    if core == h: return 0
    if h in parts: return 1
    if core.startswith(h): return 2
    if any(p.startswith(h) for p in parts): return 3
    if core.endswith(h): return 4
    if h in core: return 5
    return None


def find_span(core, h, score):
    """(a, b) of the homograph inside the (normalised) core for a given match score."""
    core, h = match_key(core), match_key(h)
    n = len(h)
    if score == 0: return 0, len(core)
    if score in (1, 3):
        pos = 0
        for p in core.split(ZWNJ):
            if (p == h) if score == 1 else p.startswith(h):
                return pos, pos + n
            pos += len(p) + 1
    if score == 2: return 0, n
    if score == 4: return len(core) - n, len(core)
    i = core.find(h)
    return i, i + n


def find_target(tokens, h, max_score=5, strict=False):
    """Strictest/earliest token matching h -> (score, index) or None. tokens are normalised tokens.
    strict=True returns None when several DIFFERENT words tie for the best match (e.g. فکش / فکر for 'فک')."""
    best, ties = None, set()
    for i, t in enumerate(tokens):
        a, b = core_span(t)
        sc = match_score(t[a:b], h)
        if sc is None or sc > max_score:
            continue
        if best is None or sc < best[0]:
            best, ties = (sc, i), {match_key(t[a:b])}
        elif sc == best[0]:
            ties.add(match_key(t[a:b]))
    if strict and len(ties) > 1:
        return None
    return best


def to_units(s):
    """'تَرْک' -> [('ت','َ'), ('ر','ْ'), ('ک','')] ; marks attach to the preceding letter."""
    units = []
    for ch in s:
        if DIAC_RE.fullmatch(ch):
            if units:
                units[-1][1] += ch
        else:
            units.append([ch, ""])
    return [(c, m) for c, m in units]


def from_units(units):
    return "".join(c + m for c, m in units)


# Mapped phoneme notation -> plain notation (labels are stored in the plain notation)
_MAPPED = {"/": "a", "a": "A", "$": "S", "c": "C", ";": "Z", "@": "?", "1": ""}
def mapped_to_plain(m):
    return "".join(_MAPPED.get(c, c) for c in str(m))
