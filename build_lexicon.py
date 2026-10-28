#!/usr/bin/env python3
"""Build the homograph lexicon used at inference time.

It answers two questions from your own (already diacritized) training data:

  1. HEADS: for every (word, reading), what does the diacritized homograph look like?
     e.g. ترک + tark -> تَرْک.  The most common form across the dataset wins, so a few
     bad rows cannot change it. Ezafe marks at the end (گَردِ) are removed, because ezafe
     depends on the sentence, not on the reading.
  2. SURFACE FORMS: which written forms of each word may be touched at runtime
     (ترک, ترکی, پرهای, می‌راند ...). Only forms that were actually seen in the data are
     allowed, so e.g. 'روباه' or 'روز' can never be mistaken for the word 'رو'.

Input CSVs need the columns used by the classifier notebook (Grapheme, Homograph Grapheme,
Homograph Phoneme or Mapped Homograph Phoneme) plus the diacritized sentence column that
diacritize_aligned.py produced (default "Diacritized Grapheme" and "... Cost").

    python build_lexicon.py data_with_diacritics.csv --clf-dir homograph_clf
Outputs: homograph_lexicon.json (used at runtime), lexicon_review.csv and surface_forms.csv (for you to review).
"""
import argparse, csv, json, os, sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from homograph_text import (DIAC_RE, KASRA, HAMZA_ABOVE, normalize, norm_chars, strip_diacritics, core_span,
                            match_score, find_span, find_target, to_units, from_units, mapped_to_plain)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--clf-dir", default=None, help="only keep words the trained classifier knows (reads meta.json)")
    ap.add_argument("--diac-col", default="Diacritized Grapheme")
    ap.add_argument("--cost-col", default="Diacritized Grapheme Cost")
    ap.add_argument("--max-cost", type=float, default=0.5, help="only rows aligned this well vote for the diacritized form")
    ap.add_argument("--min-surface-count", type=int, default=2,
                    help="any form other than the bare word must appear this many times to be allowed at runtime")
    ap.add_argument("--exclude-file", default=None, help="text file, one surface form per line, to remove from the lexicon")
    ap.add_argument("--out", default="homograph_lexicon.json")
    ap.add_argument("--review", default="lexicon_review.csv")
    ap.add_argument("--surfaces-review", default="surface_forms.csv")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in args.inputs], ignore_index=True).fillna("")
    for c in ("Grapheme", "Homograph Grapheme", args.diac_col):
        assert c in df, f"missing column '{c}'"
    plain = df["Homograph Phoneme"] if "Homograph Phoneme" in df else pd.Series([""] * len(df))
    mapped = df["Mapped Homograph Phoneme"] if "Mapped Homograph Phoneme" in df else pd.Series([""] * len(df))
    df["reading"] = [p.strip() or mapped_to_plain(m).strip() for p, m in zip(plain, mapped)]
    df["cost"] = pd.to_numeric(df[args.cost_col], errors="coerce").fillna(99) if args.cost_col in df else 0.0

    allowed = None
    if args.clf_dir:
        allowed = set(json.load(open(os.path.join(args.clf_dir, "meta.json"), encoding="utf-8"))["words"])

    votes = defaultdict(Counter)        # (word, reading) -> Counter(head)
    surf = Counter()                    # (core, word, a, b) -> count
    surf_score = {}                     # (core, word, a, b) -> min score seen
    skipped = Counter()

    for _, r in df.iterrows():
        word, reading = normalize(r["Homograph Grapheme"]), r["reading"]
        if not word or not reading or (allowed is not None and word not in allowed):
            skipped["word/reading not usable"] += 1; continue
        ntoks = norm_chars(strip_diacritics(r["Grapheme"])).strip().split(" ")
        hit = find_target(ntoks, word, strict=True)
        if hit is None:
            skipped["homograph missing or ambiguous in sentence"] += 1; continue
        sc, i = hit
        tok = ntoks[i]
        cs, ce = core_span(tok)
        core = tok[cs:ce]
        a, b = find_span(core, word, sc)
        key = (core, word, a, b)
        surf[key] += 1
        surf_score[key] = min(sc, surf_score.get(key, 9))

        if r["cost"] > args.max_cost:
            skipped["cost too high for voting"] += 1; continue
        dtoks = r[args.diac_col].split(" ")
        if len(dtoks) != len(ntoks) or strip_diacritics(dtoks[i]) != tok:
            skipped["diacritized tokens do not line up"] += 1; continue
        units = to_units(dtoks[i])
        units = units[cs:ce]                       # core only (punctuation has no marks)
        head = units[a:b]
        if b == len(units) and head:               # head ends the word: drop ezafe marks on the last letter
            c, m = head[-1]
            head[-1] = (c, m.replace(KASRA, "").replace(HAMZA_ABOVE, ""))
        votes[(word, reading)][from_units(head)] += 1

    # ---------------- heads ----------------
    heads, rows = {}, []
    for (word, reading), cnt in sorted(votes.items()):
        best, n_best = cnt.most_common(1)[0]
        total = sum(cnt.values())
        heads[f"{word}|{reading}"] = best
        rows.append({"word": word, "reading": reading, "head": best, "votes": total, "agreement": round(n_best / total, 2),
                     "other_variants": " ; ".join(f"{h} ({c})" for h, c in cnt.most_common(4)[1:])})
    by_word = defaultdict(list)
    for r in rows: by_word[r["word"]].append(r)
    for r in rows:
        clash = [o["reading"] for o in by_word[r["word"]] if o is not r and o["head"] == r["head"]]
        r["same_head_as"] = ",".join(clash)                    # readings that cannot be told apart by diacritics
    rows.sort(key=lambda r: (r["same_head_as"] == "", r["agreement"], r["word"]))
    pd.DataFrame(rows).to_csv(args.review, index=False, encoding="utf-8-sig")

    # ---------------- surface forms ----------------
    by_core = defaultdict(list)
    words_with_heads = {k.split("|")[0] for k in heads}
    for (core, word, a, b), n in surf.items():
        sc = surf_score[(core, word, a, b)]
        if word not in words_with_heads:
            continue
        if sc == 0 or n >= args.min_surface_count:
            by_core[core].append((n, sc, word, a, b))
    surfaces, srows = {}, []
    for core, lst in by_core.items():
        lst.sort(key=lambda x: (x[1], -x[0]))                  # stricter match first, then more frequent
        n, sc, word, a, b = lst[0]
        surfaces[core] = [word, a, b]
        srows.append({"surface": core, "word": word, "span": f"{a}-{b}", "count": n, "match_score": sc, "alternatives": len(lst) - 1})
    pd.DataFrame(srows).sort_values(["alternatives", "match_score"], ascending=[False, False]).to_csv(
        args.surfaces_review, index=False, encoding="utf-8-sig")

    if args.exclude_file:
        for ln in open(args.exclude_file, encoding="utf-8").read().split("\n"):
            surfaces.pop(norm_chars(strip_diacritics(ln.strip())), None)
    json.dump({"heads": heads, "surfaces": surfaces}, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(heads)} readings, {len(surfaces)} surface forms -> {args.out}")
    print("skipped rows:", dict(skipped))
    low = [r for r in rows if r["agreement"] < 0.8 or r["same_head_as"]]
    print(f"{len(low)} readings need a look in {args.review} (low agreement, or identical diacritics to another reading)")


if __name__ == "__main__":
    main()
