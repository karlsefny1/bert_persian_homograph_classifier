#!/usr/bin/env python3
"""Diacritize ONLY the homograph words of a Persian text before it goes to the TTS.

    from homograph_preprocessor import HomographDiacritizer
    prep = HomographDiacritizer("homograph_clf", "homograph_lexicon.json")      # CPU by default
    text_for_tts = prep("او اتاق را ترک کرد.")                                 # -> او اتاق را تَرْک کرد.

How it stays safe
  * Only tokens whose written form was seen as a homograph in the training data are candidates
    (homograph_lexicon.json -> "surfaces"), so 'روباه', 'روز', ... are never touched.
  * A candidate is changed only if the classifier is confident (min_conf). Otherwise it is left
    exactly as written and the TTS reads it its own way.
  * Tokens that already contain diacritics are left alone.
  * Only the homograph letters get diacritic marks added; every other character, including
    spaces and punctuation, is copied unchanged. At the end the output is checked: with
    diacritics stripped it must equal the input stripped, otherwise the ORIGINAL text is returned.

Command line:
    python homograph_preprocessor.py --text "او اتاق را ترک کرد."  [-v]
    python homograph_preprocessor.py --file in.txt --out out.txt
"""
import argparse, json, os, re, sys

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from homograph_text import DIAC_RE, core_span, norm_chars, strip_diacritics, to_units

END_RE = re.compile(r"[.!?؟؛…]\W*$")           # token that ends a sentence


class _Classifier(nn.Module):
    # identical to HomographClassifier in the training notebook (same parameter names)
    def __init__(self, encoder, word_mask, open_id, close_id, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        H = encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * H, word_mask.shape[1])
        self.register_buffer("word_mask", word_mask.clone())
        self.open_id, self.close_id = open_id, close_id

    def forward(self, input_ids, attention_mask, word_ids):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        B, T, _ = h.shape
        pos = torch.arange(T, device=h.device)[None, :]
        start = (input_ids == self.open_id).float().argmax(1)
        end_hit = input_ids == self.close_id
        end = torch.where(end_hit.any(1), end_hit.float().argmax(1), attention_mask.sum(1) - 1)
        span = (pos > start[:, None]) & (pos < end[:, None])
        span_vec = (h * span[..., None]).sum(1) / span.sum(1, keepdim=True).clamp(min=1)
        start_vec = h[torch.arange(B, device=h.device), start]
        logits = self.head(self.drop(torch.cat([start_vec, span_vec], dim=-1))).float()
        return logits.masked_fill(~self.word_mask[word_ids], -1e9)


class HomographDiacritizer:
    def __init__(self, clf_dir, lexicon_path, device="cpu", min_conf=0.80, batch_size=32, num_threads=None):
        if num_threads:
            torch.set_num_threads(num_threads)
        meta = json.load(open(os.path.join(clf_dir, "meta.json"), encoding="utf-8"))
        self.tok = AutoTokenizer.from_pretrained(clf_dir)
        self.open_tag, self.close_tag = meta["mark_open"], meta["mark_close"]
        o, c = self.tok.convert_tokens_to_ids([self.open_tag, self.close_tag])
        self.open_id, self.close_id = o, c
        enc = AutoModel.from_config(AutoConfig.from_pretrained(clf_dir))
        self.model = _Classifier(enc, torch.zeros(len(meta["words"]), len(meta["classes"]), dtype=torch.bool), o, c)
        self.model.load_state_dict(torch.load(os.path.join(clf_dir, "model.pt"), map_location="cpu"))
        self.model.to(device).eval()
        self.device, self.min_conf, self.bs, self.max_len = device, min_conf, batch_size, meta["max_len"]
        self.classes = [tuple(x) for x in meta["classes"]]
        self.word2id = {w: i for i, w in enumerate(meta["words"])}
        lex = json.load(open(lexicon_path, encoding="utf-8"))
        self.heads = lex["heads"]
        self.surfaces = {k: v for k, v in lex["surfaces"].items() if v[0] in self.word2id}

    # ------------------------------------------------------------------ candidates
    def _candidates(self, text):
        toks = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        segs, cur, prev_end = [], [], 0
        for s, e in toks:                                   # split into sentences
            if cur and "\n" in text[prev_end:s]:
                segs.append(cur); cur = []
            cur.append((s, e)); prev_end = e
            if END_RE.search(text[s:e]):
                segs.append(cur); cur = []
        if cur: segs.append(cur)

        cands = []
        for seg in segs:
            ntoks = [re.sub(r"\s+", " ", norm_chars(strip_diacritics(text[s:e]))) for s, e in seg]
            for j, (s, e) in enumerate(seg):
                raw = text[s:e]
                cs, ce = core_span(raw)
                ocore = raw[cs:ce]
                if not ocore or DIAC_RE.search(ocore):
                    continue                                 # empty, or already diacritized by the author
                ncore = norm_chars(ocore)
                ent = self.surfaces.get(ncore)
                if ent is None:
                    continue
                word, a, b = ent
                pre, post = norm_chars(strip_diacritics(raw[:cs])), norm_chars(strip_diacritics(raw[ce:]))
                marked = " ".join(ntoks[:j] + [f"{pre} {self.open_tag} {ncore} {self.close_tag} {post}"] + ntoks[j + 1:])
                cands.append({"start": s + cs, "end": s + ce, "ocore": ocore, "ncore": ncore, "word": word,
                              "a": a, "b": b, "marked": re.sub(r"\s+", " ", marked).strip()})
        return cands

    # ------------------------------------------------------------------ classifier
    @torch.inference_mode()
    def _classify(self, cands):
        for i in range(0, len(cands), self.bs):
            chunk = cands[i:i + self.bs]
            enc = self.tok([c["marked"] for c in chunk], truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt")
            wid = torch.tensor([self.word2id[c["word"]] for c in chunk])
            probs = torch.softmax(self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device),
                                             wid.to(self.device)), -1).cpu()
            for k, c in enumerate(chunk):
                ids = enc["input_ids"][k]
                if not ((ids == self.open_id).any() and (ids == self.close_id).any()):
                    c["reading"], c["conf"] = None, 0.0      # marker cut off by truncation
                    continue
                p, idx = probs[k].max(0)
                c["reading"], c["conf"] = self.classes[int(idx)][1], float(p)

    # ------------------------------------------------------------------ public API
    def analyze(self, text):
        """Returns (new_text, decisions). Each decision says what happened to one candidate word."""
        cands = self._candidates(text)
        if not cands:
            return text, []
        self._classify(cands)
        edits, decisions = [], []
        for c in cands:
            d = {"token": text[c["start"]:c["end"]], "word": c["word"], "reading": c["reading"],
                 "confidence": round(c["conf"], 3), "changed": False, "result": c["ocore"]}
            decisions.append(d)
            if c["reading"] is None:
                d["why"] = "word cut off by max length"; continue
            if c["conf"] < self.min_conf:
                d["why"] = f"confidence below {self.min_conf}"; continue
            head = self.heads.get(f"{c['word']}|{c['reading']}")
            if head is None:
                d["why"] = "reading not in lexicon"; continue
            units = to_units(head)
            if "".join(ch for ch, _ in units) != c["ncore"][c["a"]:c["b"]]:
                d["why"] = "lexicon form does not match the written form"; continue
            new_span = "".join(oc + m for oc, (_, m) in zip(c["ocore"][c["a"]:c["b"]], units))
            new_core = c["ocore"][:c["a"]] + new_span + c["ocore"][c["b"]:]
            if new_core == c["ocore"]:
                d["why"] = "form has no diacritics"; continue
            edits.append((c["start"], c["end"], new_core))
            d.update(changed=True, result=new_core, why="ok")

        out = text
        for s, e, new in sorted(edits, reverse=True):
            out = out[:s] + new + out[e:]
        if strip_diacritics(out) != strip_diacritics(text):    # last line of defence
            return text, [dict(d, changed=False, why="SAFETY CHECK FAILED, text returned unchanged") for d in decisions]
        return out, decisions

    def __call__(self, text):
        return self.analyze(text)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clf", default="homograph_clf")
    ap.add_argument("--lexicon", default="homograph_lexicon.json")
    ap.add_argument("--text"); ap.add_argument("--file"); ap.add_argument("--out")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--min-conf", type=float, default=0.80)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    prep = HomographDiacritizer(a.clf, a.lexicon, device=a.device, min_conf=a.min_conf)
    lines = [a.text] if a.text is not None else open(a.file, encoding="utf-8").read().split("\n")
    res = []
    for ln in lines:
        new, dec = prep.analyze(ln)
        res.append(new)
        if a.verbose:
            for d in dec: print("  ", d, file=sys.stderr)
    if a.out: open(a.out, "w", encoding="utf-8").write("\n".join(res))
    else: print("\n".join(res))


if __name__ == "__main__":
    main()
