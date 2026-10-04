"""
Simulated layout changes (proposal §4.1: "controlled structural modifications
will be introduced to the locally cached HTML to simulate realistic change
scenarios").

Operators (all seeded and deterministic):

structural — preserve every text node, so ground truth carries over unchanged
  rename_classes   rename a fraction of class tokens everywhere (CSS refactor /
                   hashed CSS-module names)
  rename_ids       rename a fraction of id attributes
  wrap_elements    wrap elements in new <div>/<span> wrappers (deeper nesting)
  retag_elements   change tags among equivalents (div<->section, h1->h2, span->em ...)
  reorder_siblings shuffle the order of sibling elements
  insert_sibling   insert a small new block next to content (shifts positions)
  redesign         rename_classes + wrap_elements + retag_elements together
  insert_banner    insert a promotional banner at the top of the content
content — change values, not structure; ground truth is updated by construction
  perturb_numbers  change numeric field values (e.g. a price) everywhere they occur
control — no change at all (the same page again)

``focus=True`` restricts structural operators to the region around the
elements that hold ground-truth values (those elements and up to three
ancestor levels), modelling a redesign of the content area. Whether a change
actually breaks the deployed selectors is *measured*, never assumed.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

from lxml import etree

from .htmlutil import parse_html, to_html, visible_text
from .matching import _text, canonical, value_in_text

STRUCTURAL_OPS = (
    "rename_classes", "rename_ids", "wrap_elements", "retag_elements",
    "reorder_siblings", "insert_sibling", "redesign", "insert_banner",
)
CONTENT_OPS = ("perturb_numbers",)
CONTROL_OPS = ("control",)
ALL_OPS = STRUCTURAL_OPS + CONTENT_OPS + CONTROL_OPS


def family(op: str) -> str:
    if op in CONTENT_OPS:
        return "content"
    if op in CONTROL_OPS:
        return "control"
    return "structural"


SKIP = {"html", "head", "body", "script", "style", "title", "meta", "link", "noscript", "template"}
NO_WRAP = SKIP | {
    "li", "tr", "td", "th", "thead", "tbody", "tfoot", "option", "optgroup", "dt", "dd",
    "caption", "colgroup", "col", "source", "track", "param", "area", "main",
}
LIST_PARENTS = {"ul", "ol", "table", "tbody", "thead", "tfoot", "tr", "select", "dl"}
PHRASING = {
    "p", "span", "a", "b", "i", "em", "strong", "small", "label", "h1", "h2", "h3",
    "h4", "h5", "h6", "button", "sup", "sub", "time", "code", "abbr", "cite", "q",
}
RETAG = {
    "div": "section", "section": "div", "article": "div", "p": "div",
    "h1": "h2", "h2": "h3", "h3": "h4", "h4": "h5", "strong": "b", "b": "strong",
    "em": "i", "i": "em", "span": "em",
}
NO_REORDER_PARENTS = {"tr", "table", "tbody", "thead", "select", "head", "html"}

BANNER_TEXT = "Free delivery on your first order"
NOTE_TEXT = "Customers also viewed this item"


@dataclass
class MutationResult:
    html: str
    spec: dict
    gt_updates: dict = field(default_factory=dict)  # field -> new raw GT value
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _new_token(rng: random.Random, prefix: str) -> str:
    return f"{prefix}{rng.getrandbits(24):06x}"


def _k(severity: float, n: int) -> int:
    if n <= 0:
        return 0
    return max(1, min(n, math.ceil(severity * n)))


def _elements(root):
    body = root.find("body")
    base = body if body is not None else root
    return [e for e in base.iter() if isinstance(e.tag, str) and e.tag not in SKIP]


def field_nodes(root, gt_fields: dict) -> list:
    """
    Deepest elements whose text contains a ground-truth value.
    ``gt_fields``: {field_name: (values, field_type)}.
    """
    targets = []
    for values, ftype in gt_fields.values():
        for v in list(values)[:10]:
            if v and str(v).strip():
                targets.append((_text(str(v)), canonical(str(v), ftype), ftype))
    if not targets:
        return []
    elements = _elements(root)
    matched: dict = {}
    found = []
    for el in reversed(elements):  # descendants before ancestors
        text = _text(el.text_content() or "")
        hit = False
        for norm, canon, ftype in targets:
            if norm and norm in text:
                hit = True
                break
            if ftype in ("price", "number") and canon and canon in re.sub(r"[^\d.]", " ", text).split():
                hit = True
                break
        matched[el] = hit
        if hit and not any(matched.get(c) for c in el if isinstance(c.tag, str)):
            found.append(el)
    return found


def _region(nodes, depth: int = 3) -> list:
    seen, out = set(), []
    for node in nodes:
        el, level = node, 0
        while el is not None and level <= depth:
            if isinstance(el.tag, str) and el.tag not in SKIP and id(el) not in seen:
                seen.add(id(el))
                out.append(el)
            el = el.getparent()
            level += 1
    return out


def _content_parent(root):
    for tag in ("main", "article", "body"):
        el = root.find(f".//{tag}") if tag != "body" else root.find("body")
        if el is not None:
            return el
    return root


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
def op_rename_classes(root, rng, severity, candidates) -> int:
    tokens = sorted({t for e in candidates for t in (e.get("class") or "").split()})
    if not tokens:
        return 0
    chosen = set(rng.sample(tokens, _k(severity, len(tokens))))
    mapping = {t: _new_token(rng, "c") for t in sorted(chosen)}
    changed = 0
    for el in _elements(root):
        classes = (el.get("class") or "").split()
        if any(c in mapping for c in classes):
            el.set("class", " ".join(mapping.get(c, c) for c in classes))
            changed += 1
    return changed


def op_rename_ids(root, rng, severity, candidates) -> int:
    with_id = [e for e in candidates if e.get("id")]
    if not with_id:
        return 0
    for el in rng.sample(with_id, _k(severity, len(with_id))):
        el.set("id", _new_token(rng, "i"))
    return _k(severity, len(with_id))


def op_wrap_elements(root, rng, severity, candidates) -> int:
    eligible = [
        e for e in candidates
        if e.tag not in NO_WRAP and e.getparent() is not None
        and e.getparent().tag not in LIST_PARENTS
    ]
    if not eligible:
        return 0
    chosen = rng.sample(eligible, _k(severity, len(eligible)))
    for el in chosen:
        parent = el.getparent()
        if parent is None:
            continue
        tag = "span" if (parent.tag in PHRASING or el.tag in PHRASING) else "div"
        wrapper = etree.Element(tag)
        wrapper.set("class", _new_token(rng, "w-"))
        wrapper.tail, el.tail = el.tail, None
        parent.replace(el, wrapper)
        wrapper.append(el)
    return len(chosen)


def op_retag_elements(root, rng, severity, candidates) -> int:
    eligible = [e for e in candidates if e.tag in RETAG]
    if not eligible:
        return 0
    chosen = rng.sample(eligible, _k(severity, len(eligible)))
    for el in chosen:
        el.tag = RETAG[el.tag]
    return len(chosen)


def op_reorder_siblings(root, rng, severity, candidates) -> int:
    parents, seen = [], set()
    for el in candidates:
        p = el.getparent()
        if p is None or id(p) in seen or p.tag in NO_REORDER_PARENTS:
            continue
        kids = [c for c in p if isinstance(c.tag, str)]
        if len(kids) >= 2:
            seen.add(id(p))
            parents.append(p)
    if not parents:
        return 0
    chosen = rng.sample(parents, _k(severity, len(parents)))
    for p in chosen:
        kids = [c for c in p if isinstance(c.tag, str)]
        order = kids[:]
        rng.shuffle(order)
        if order == kids:
            order = kids[1:] + kids[:1]
        for c in kids:
            p.remove(c)
        for c in order:
            p.append(c)
    return len(chosen)


def op_insert_sibling(root, rng, severity, candidates) -> int:
    parents = [e.getparent() for e in candidates
               if e.getparent() is not None and e.getparent().tag not in LIST_PARENTS | SKIP]
    parents = list({id(p): p for p in parents}.values())
    if not parents:
        return 0
    chosen = rng.sample(parents, _k(severity, len(parents)))
    for p in chosen:
        tag = "span" if p.tag in PHRASING else "div"
        note = etree.Element(tag)
        note.set("class", _new_token(rng, "note-"))
        inner = etree.SubElement(note, "span")
        inner.text = NOTE_TEXT
        kids = [c for c in p if isinstance(c.tag, str)]
        index = p.index(kids[rng.randrange(len(kids))]) if kids else 0
        p.insert(index, note)
    return len(chosen)


def op_insert_banner(root, rng, severity, candidates) -> int:
    parent = _content_parent(root)
    banner = etree.Element("div")
    banner.set("class", _new_token(rng, "promo-"))
    para = etree.SubElement(banner, "p")
    para.text = BANNER_TEXT
    banner.tail = parent.text
    parent.text = None
    parent.insert(0, banner)
    return 1


def op_redesign(root, rng, severity, candidates) -> int:
    n = op_rename_classes(root, rng, severity, candidates)
    n += op_wrap_elements(root, rng, max(0.1, severity / 2), candidates)
    n += op_retag_elements(root, rng, max(0.1, severity / 2), candidates)
    return n


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _format_like(old: str, value: float) -> str:
    decimals = len(old.split(".")[1]) if "." in old else 0
    if "," in old.split(".")[0]:
        return f"{value:,.{decimals}f}"
    return f"{value:.{decimals}f}"


def op_perturb_numbers(root, rng, gt_fields: dict, notes: list) -> tuple[int, dict]:
    """Change numeric single-value fields everywhere they occur; return GT updates."""
    updates, changed = {}, 0
    for name, (values, ftype) in gt_fields.items():
        if ftype not in ("price", "number") or len(values) != 1:
            continue
        raw = str(values[0])
        m = _NUMBER.search(raw)
        if not m:
            continue
        old = m.group(0)
        try:
            old_value = float(old.replace(",", ""))
        except ValueError:
            continue
        # Prices move by 5-15%; other numbers (ratings, counts) only 2-5%, so a
        # 4.6/5 rating stays plausible.
        lo, hi = (0.05, 0.15) if ftype == "price" else (0.02, 0.05)
        factor = rng.choice([1 - rng.uniform(lo, hi), 1 + rng.uniform(lo, hi)])
        new = _format_like(old, old_value * factor)
        if new == old:
            continue
        hits = 0
        # Whole-number matches only: "22" must not hit "2022" or "22.5".
        rx = re.compile(r"(?<![\d.,])" + re.escape(old) + r"(?![\d]|[.,]\d)")
        plain_old, plain_new = old.replace(",", ""), new.replace(",", "")
        rx_plain = re.compile(r"(?<![\d.,])" + re.escape(plain_old) + r"(?![\d]|[.,]\d)")
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            # Text nodes, including JSON-LD script text, so structured copies
            # of the value stay consistent with the visible one.
            if el.text and rx.search(el.text):
                el.text = rx.sub(new, el.text)
                hits += 1
            if el.text and plain_old != old and el.tag == "script" and rx_plain.search(el.text):
                el.text = rx_plain.sub(plain_new, el.text)
            if el.tail and rx.search(el.tail):
                el.tail = rx.sub(new, el.tail)
                hits += 1
            for attr in ("content", "data-price", "data-value", "value"):
                if el.get(attr) and rx_plain.search(el.get(attr)):
                    el.set(attr, rx_plain.sub(plain_new, el.get(attr)))
        if hits:
            updates[name] = raw.replace(old, new, 1)
            changed += hits
        else:
            notes.append(f"{name}: value {raw!r} not found in a text node; left unchanged")
    if not updates:
        # No numeric field could be changed: change an unrelated number so the
        # step is still a pure content change. Text that contains any labelled
        # value is never touched, so the ground truth stays valid.
        labelled = [_text(str(v)) for values, _ in gt_fields.values() for v in values if str(v).strip()]
        for el in _elements(root):
            text = el.text or ""
            m = _NUMBER.search(text)
            if not m or any(lv and lv in _text(text) for lv in labelled):
                continue
            old = m.group(0)
            new = _format_like(old, float(old.replace(",", "")) + 1)
            el.text = text[: m.start()] + new + text[m.end():]
            notes.append("no numeric field available; changed an unrelated number")
            changed += 1
            break
        else:
            notes.append("no number found outside labelled values; page unchanged")
    return changed, updates


OPERATORS = {
    "rename_classes": op_rename_classes,
    "rename_ids": op_rename_ids,
    "wrap_elements": op_wrap_elements,
    "retag_elements": op_retag_elements,
    "reorder_siblings": op_reorder_siblings,
    "insert_sibling": op_insert_sibling,
    "insert_banner": op_insert_banner,
    "redesign": op_redesign,
}


# ---------------------------------------------------------------------------
def apply_mutation(html: str, op: str, seed: int, severity: float = 0.5,
                   focus: bool = True, gt_fields: dict | None = None) -> MutationResult:
    """
    Apply one operator. ``gt_fields`` = {field: (values, field_type)} for the
    page being mutated (used for focus and for content perturbation).
    """
    if op not in ALL_OPS:
        raise ValueError(f"unknown mutation operator {op!r}; choose from {', '.join(ALL_OPS)}")
    gt_fields = gt_fields or {}
    spec = {"op": op, "family": family(op), "seed": seed,
            "severity": severity, "focus": focus}
    notes: list = []
    if op == "control":
        spec["affected"] = 0
        return MutationResult(html=html, spec=spec)

    root = parse_html(html)
    if root is None:
        raise ValueError("cannot mutate an empty or unparseable page")
    rng = random.Random(seed)

    gt_updates = {}
    if op == "perturb_numbers":
        before_text = visible_text(root)
        affected, gt_updates = op_perturb_numbers(root, rng, gt_fields, notes)
        # A changed number may also occur inside another field's value; such
        # fields are flagged so their (now stale) label is not scored.
        after_text = visible_text(root)
        preserved = {}
        for name, (values, ftype) in gt_fields.items():
            if name in gt_updates:
                continue
            was = [v for v in values if value_in_text(v, before_text, ftype)]
            preserved[name] = all(value_in_text(v, after_text, ftype) for v in was)
        spec["gt_preserved"] = preserved
        lost = [n for n, ok in preserved.items() if not ok]
        if lost:
            notes.append("other labelled values changed by the perturbation: " + ", ".join(lost))
    else:
        nodes = field_nodes(root, gt_fields) if gt_fields else []
        spec["field_nodes"] = len(nodes)
        if focus and nodes:
            candidates = _region(nodes)
        else:
            if focus:
                notes.append("focus requested but no ground-truth nodes found; used whole page")
            candidates = _elements(root)
        affected = OPERATORS[op](root, rng, severity, candidates)
    spec["affected"] = affected
    out = to_html(root)

    # Verify text preservation for structural operators.
    if family(op) == "structural" and gt_fields:
        before = visible_text(parse_html(html))
        after = visible_text(parse_html(out))
        preserved = {}
        for name, (values, ftype) in gt_fields.items():
            was = [v for v in values if value_in_text(v, before, ftype)]
            preserved[name] = all(value_in_text(v, after, ftype) for v in was)
        spec["gt_preserved"] = preserved
        lost = [n for n, ok in preserved.items() if not ok]
        if lost:
            notes.append("ground-truth text not preserved for: " + ", ".join(lost))
    if notes:
        spec["notes"] = notes
    return MutationResult(html=out, spec=spec, gt_updates=gt_updates, notes=notes)
