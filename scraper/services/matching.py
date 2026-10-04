"""
Ground-truth matching.

Two complementary measures are recorded for every extracted field:

* exact match — after type-aware canonicalisation (numbers compared as
  decimals, dates as ISO, text case/whitespace-insensitive). This is the
  primary field-level metric (SWDE-style attribute accuracy).
* token-level precision/recall/F1 — SQuAD-style bag-of-words overlap, which
  gives partial credit for near-misses (e.g. a title with an extra suffix).

List fields use multiset matching of canonical values: TP = matched items,
FP = extra predicted items, FN = missed ground-truth items.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation

from .normalizer import canonical_url, clean_text, normalize, parse_number

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _text(value: str) -> str:
    return unicodedata.normalize("NFKC", clean_text(value)).lower()


def canonical(value: str, field_type: str = "string") -> str:
    v = normalize(value or "", field_type)
    if field_type in ("price", "number"):
        number = parse_number(v)
        if number is not None:
            try:
                return format(Decimal(number).normalize(), "f")
            except InvalidOperation:
                pass
    if field_type == "url":
        return canonical_url(v)
    return _text(v)


def exact_match(pred: str, gt: str, field_type: str = "string") -> bool:
    g = canonical(gt, field_type)
    return bool(g) and canonical(pred, field_type) == g


def token_prf(pred: str, gt: str):
    """(precision, recall, f1) or None when the ground truth is empty."""
    g = _TOKEN.findall(_text(gt))
    if not g:
        return None
    p = _TOKEN.findall(_text(pred))
    if not p:
        return (0.0, 0.0, 0.0)
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return (0.0, 0.0, 0.0)
    precision, recall = same / len(p), same / len(g)
    return (precision, recall, 2 * precision * recall / (precision + recall))


def list_match(preds: list[str], gts: list[str], field_type: str = "string"):
    """(tp, fp, fn) by multiset matching of canonical values."""
    pc = Counter(canonical(v, field_type) for v in preds if str(v).strip())
    gc = Counter(canonical(v, field_type) for v in gts if str(v).strip())
    tp = sum((pc & gc).values())
    return tp, sum(pc.values()) - tp, sum(gc.values()) - tp


def value_in_text(gt_value: str, page_text: str, field_type: str = "string") -> bool:
    """Is a ground-truth value visibly present in some page text?"""
    if not gt_value or not gt_value.strip():
        return False
    hay = _text(page_text)
    if _text(gt_value) in hay:
        return True
    if field_type in ("price", "number"):
        target = canonical(gt_value, field_type)
        for m in re.finditer(r"[-+]?\d[\d.,'  ]*\d|\d", clean_text(page_text)):
            if canonical(m.group(0), field_type) == target:
                return True
    return False
