"""
Annotated site dataset -> evaluation config, and the matching of recorded
dataset values against saved snapshots.

The dataset (JSON, or a sites/fields CSV pair, or the equivalent workbook)
lists, per site, the URL and category and, per field, its type, whether it is
required, single or list, the value an annotator saw on the live page, whether
the value is dynamic, where it was found (visible text, JSON-LD, meta, URL) and
a description. ``to_config`` turns it into ``eval/sites.yaml`` without
changing any URL, category or field; type names are mapped to the harness's
types and every original attribute is kept on the field for the record.

Recorded values may carry annotator notes ("… (truncated)", "e.g. …",
"(article body, ~816 words)", "false (No)"). ``probes`` turns a recorded value
into the literal strings that should appear on the page, and ``locate``
searches a snapshot for them; nothing here invents a value — whatever cannot be
found literally is left for a person to label.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..services.fetcher import METADATA_ID
from ..services.htmlutil import parse_html
from ..services.matching import _text, canonical, value_in_text
from ..services.normalizer import clean_text, normalize, resolve_url

TYPE_MAP = {
    "text": "string", "string": "string", "price": "price", "number": "number",
    "date": "date", "url": "url", "boolean": "boolean", "other": "mixed",
}
CATEGORY_MAP = {
    "e-commerce": "ecommerce", "ecommerce": "ecommerce", "news": "news",
    "finance": "finance", "jobs": "jobs", "academic": "academic",
}
CARD_MAP = {"single": "single", "list": "many", "many": "many"}
YES = {"yes", "y", "true", "1"}

ELLIPSIS = "…"
_TRAILING_PAREN = re.compile(r"\s*\(([^()]*)\)\s*$")
_SHOWN = re.compile(r"shown\s+['‘’\"]([^'‘’\"]+)['‘’\"]", re.I)
_COUNT_NOTE = re.compile(r"^\s*\d[\d,]*\s+(bullets?|authors?|affiliations?|items?|results?|versions?|more|total|paragraphs?|words?)\b", re.I)
_DASH_COUNT = re.compile(r"\s+[\u2014\u2013-]\s+(\d[\d,]*)\s+\w+\s*$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _norm_header(h) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(h or "").strip().lower()).strip("_")


_FIELD_ALIASES = {"single_list": "cardinality", "current_value": "current_value"}
_SITE_ALIASES = {"approx_fields": "approx_field_count"}


def load_dataset(path, fields_path=None) -> dict:
    """Return {"meta": {...}, "sites": [site dicts with "fields"]}."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = {k: v for k, v in data.items() if k != "sites"}
        return {"meta": meta, "sites": data["sites"]}
    if suffix in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook

        wb = load_workbook(path, read_only=True, data_only=True)
        sheets = {ws.title.lower(): ws for ws in wb.worksheets}
        site_ws = next(ws for t, ws in sheets.items() if "website" in t or "site" in t)
        field_ws = next(ws for t, ws in sheets.items() if "field" in t)

        def rows(ws, aliases):
            values = list(ws.iter_rows(values_only=True))
            header = [aliases.get(_norm_header(h), _norm_header(h)) for h in values[0]]
            return [dict(zip(header, ("" if v is None else v for v in r))) for r in values[1:] if any(r)]

        sites, fields = rows(site_ws, _SITE_ALIASES), rows(field_ws, _FIELD_ALIASES)
        return {"meta": {"source_file": path.name}, "sites": _join(sites, fields)}
    if suffix == ".csv":
        if fields_path is None:
            raise ValueError("a CSV dataset needs both the sites CSV and the fields CSV")

        def rows(p):
            with open(p, encoding="utf-8-sig", newline="") as fh:
                return [{_norm_header(k): v for k, v in r.items()} for r in csv.DictReader(fh)]

        return {"meta": {"source_file": path.name}, "sites": _join(rows(path), rows(fields_path))}
    raise ValueError(f"unsupported dataset file: {path.name} (use .json, .xlsx or two .csv files)")


def _join(sites: list[dict], fields: list[dict]) -> list[dict]:
    by_site: dict = {}
    for f in fields:
        by_site.setdefault(str(f.get("site_id")).strip(), []).append(f)
    out = []
    for s in sites:
        s = dict(s)
        s["fields"] = by_site.get(str(s.get("site_id")).strip(), [])
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Probes: literal strings a recorded value should match on the page
# ---------------------------------------------------------------------------
@dataclass
class Probe:
    kind: str                      # literal | prefix | items | boolean | none
    values: list = field(default_factory=list)   # literal candidates (single)
    items: list = field(default_factory=list)    # list items: [[candidates], ...]
    complete: bool = True          # False when the recorded list/text was abridged
    expected_count: int | None = None
    reason: str = ""


def _strip_note(v: str) -> list[str]:
    """Candidates for a value that may end with a parenthetical note, in order
    of preference: the value as recorded, without the note, then the form the
    note says was shown ("(shown 'Sep 8, 2026')") or a URL given in the note."""
    if " / " in v and "(" in v:  # "A (JSON-LD) / B (meta ...)": two recorded forms
        out = []
        for part in v.split(" / "):
            out.extend(_strip_note(part.strip())[1:] or [part.strip()])
        return _uniq([v] + out)
    out = [v]
    m = _TRAILING_PAREN.search(v)
    if m:
        inner = m.group(1).strip()
        head = v[: m.start()].strip()
        if head:
            out.append(head)
        shown = _SHOWN.search(inner)
        if shown:
            out.append(shown.group(1).strip())
        elif re.match(r"https?://\S+$", inner):
            out.append(inner)   # "InStock (http://schema.org/InStock)"
    return _uniq(out)


def _uniq(values: list[str]) -> list[str]:
    seen, out = set(), []
    for c in values:
        c = c.strip().strip("'\"\u2018\u2019")
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def probes(f: dict) -> Probe:
    """Turn a field's recorded value into literal search probes."""
    v = str(f.get("dataset_value") or "").strip()
    ftype = f.get("type", "string")
    if not v:
        return Probe("none", reason="no recorded value")
    low = v.lower()
    if low.startswith("e.g.") or low.startswith("eg "):
        return Probe("none", reason="recorded value is an example, not the page value")
    if v.startswith("(") and v.endswith(")"):
        return Probe("none", reason=f"recorded value is a note: {v}")
    if ftype == "boolean":
        word = _TRAILING_PAREN.search(v)
        base = v[: word.start()].strip() if word else v
        return Probe("boolean", values=[base], reason=(word.group(1) if word else ""))

    if f.get("cardinality") == "many":
        parts = [p.strip() for p in re.split(r"\s*;\s*", v) if p.strip()]
        expected = None
        dash = _DASH_COUNT.search(parts[-1]) if parts else None
        if dash:  # "... — 7 versions"
            expected = int(dash.group(1).replace(",", ""))
            parts[-1] = parts[-1][: dash.start()].strip()
        tail = _TRAILING_PAREN.search(parts[-1]) if parts else None
        if tail and _COUNT_NOTE.match(tail.group(1)):
            n = re.match(r"\s*(\d[\d,]*)", tail.group(1)).group(1).replace(",", "")
            expected = int(n)
            parts[-1] = parts[-1][: tail.start()].strip()
        complete = not any(ELLIPSIS in p or p == "..." for p in parts) and expected is None
        items = []
        for p in parts:
            if p in (ELLIPSIS, "...") or not p:
                continue
            if ELLIPSIS in p:
                p = p.split(ELLIPSIS)[0].strip()
                if len(p) < 8:
                    continue
                items.append([p + ELLIPSIS])  # prefix item
                continue
            items.append(_strip_note(p))
        if "(collapsed" in low or "+" in v and "more" in low:
            complete = False
        return Probe("items", items=items, complete=complete, expected_count=expected)

    if ELLIPSIS in v:
        prefix = v.split(ELLIPSIS)[0].strip()
        if len(prefix) >= 12:
            return Probe("prefix", values=[prefix], complete=False)
        return Probe("none", reason="abridged value too short to locate")
    return Probe("literal", values=_strip_note(v))


# ---------------------------------------------------------------------------
# Locating recorded values in a snapshot
# ---------------------------------------------------------------------------
_VALUE_ATTRS = ("content", "href", "src", "datetime", "title", "alt", "aria-label", "class", "value")


class PageIndex:
    """Searchable texts of a clean DOM, split into visible content and the
    page-metadata block, with attribute values (URLs resolved)."""

    def __init__(self, clean_html: str, base_url: str = ""):
        self.root = parse_html(clean_html)
        self.base_url = base_url
        meta_el = self.root.get_element_by_id(METADATA_ID, None) if self.root is not None else None
        self.meta_root = meta_el
        self.visible_text = self._text(self.root, skip=meta_el)
        self.meta_text = self._text(meta_el) if meta_el is not None else ""
        self.visible_attrs = self._attrs(self.root, skip=meta_el)
        self.meta_attrs = self._attrs(meta_el) if meta_el is not None else ""
        # Whole node texts and attribute values: a value equal to one of these
        # is a strong match; a value inside a longer text is a partial one.
        self._wholes: set[str] = set()
        if self.root is not None:
            for el in self.root.iter():
                if isinstance(el.tag, str):
                    t = _norm(el.text_content())
                    if t and len(t) <= 600:
                        self._wholes.add(t)
        for blob in (self.visible_attrs, self.meta_attrs):
            self._wholes.update(v.strip() for v in blob.split(" \n ") if v.strip())
        self._typed: dict = {}

    @staticmethod
    def _text(root, skip=None) -> str:
        if root is None:
            return ""
        parts = []
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            if skip is not None and (el is skip or _inside(el, skip)):
                continue
            if el.text:
                parts.append(el.text)
            if el.tail and not (skip is not None and el is skip):
                parts.append(el.tail)
            # Block boundaries become spaces so words do not merge.
            parts.append(" ")
        return re.sub(r"\s+", " ", "".join(parts)).strip()

    def _attrs(self, root, skip=None) -> str:
        if root is None:
            return ""
        vals = []
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            if skip is not None and (el is skip or _inside(el, skip)):
                continue
            for a in _VALUE_ATTRS:
                v = el.get(a)
                if v:
                    vals.append(resolve_url(v, self.base_url) if a in ("href", "src") else v)
        return " \n ".join(vals)

    def is_whole(self, value: str, ftype: str) -> bool:
        """Does ``value`` equal an entire element text or attribute value?"""
        key = ftype if ftype in ("price", "number", "date", "url") else "string"
        if key not in self._typed:
            if key == "string":
                forms = {_text(w) for w in self._wholes}
            elif key == "date":
                forms = {normalize(w, "date") for w in self._wholes if len(w) <= 60}
            elif key == "url":
                forms = {canonical(w, "url") for w in self._wholes if "/" in w and " " not in w}
            else:
                forms = {canonical(w, key) for w in self._wholes
                         if len(w) <= 40 and re.search(r"\d", w)}
            self._typed[key] = forms
        target = _text(value) if key == "string" else (
            normalize(value, "date") if key == "date" else canonical(value, key))
        return bool(target) and target in self._typed[key]

    def where(self, value: str, ftype: str) -> list[str]:
        """Which parts of the page contain ``value``: visible, attribute, metadata."""
        found = []
        if _contains(self.visible_text, value, ftype):
            found.append("visible")
        if _contains(self.visible_attrs, value, ftype):
            found.append("attribute")
        if _contains(self.meta_text, value, ftype) or _contains(self.meta_attrs, value, ftype):
            found.append("metadata")
        return found


def _inside(el, ancestor) -> bool:
    parent = el.getparent()
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.getparent()
    return False


def _contains(hay: str, value: str, ftype: str) -> bool:
    if not hay or not value:
        return False
    if value.endswith(ELLIPSIS):  # prefix probe
        return _text(value[:-1]) in _text(hay)
    if ftype == "url":
        target = canonical(value, "url")
        return any(canonical(tok, "url") == target for tok in re.split(r"\s+", hay) if "/" in tok) \
            or _text(value) in _text(hay)
    if ftype == "date":
        if _text(value) in _text(hay):
            return True
        iso = normalize(value, "date")
        return bool(re.match(r"\d{4}-\d{2}-\d{2}$", iso)) and iso in hay
    t = ftype if ftype in ("price", "number") else "string"
    if value_in_text(value, hay, t):
        return True
    if ftype == "number" and value.strip() in _DIGIT_WORDS:  # class="star-rating Three"
        return re.search(r"\b" + _DIGIT_WORDS[value.strip()] + r"\b", hay, re.I) is not None
    return False


_DIGIT_WORDS = {str(i): w for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"])}


@dataclass
class Located:
    status: str                    # found | found-variant | prefix | list-complete | list-partial | not-found | not-literal | boolean
    value: object = None           # str (single) or list[str] (many) — a candidate label
    where: list = field(default_factory=list)
    detail: str = ""


def locate(f: dict, page: PageIndex) -> Located:
    """Find the recorded value of field ``f`` in a snapshot."""
    pr = probes(f)
    ftype = f.get("type", "string")
    if pr.kind == "none":
        return Located("not-literal", detail=pr.reason)
    if pr.kind == "boolean":
        return Located("boolean", value=normalize(pr.values[0], "boolean"),
                       detail=f"recorded {f.get('dataset_value')!r}; yes/no value, check it on the page")
    if pr.kind == "literal":
        hits = []
        for i, cand in enumerate(pr.values):
            where = page.where(cand, ftype)
            if where:
                hits.append((i, cand, where, page.is_whole(cand, ftype)))
        if not hits:
            return Located("not-found", detail="recorded value not in the saved page")
        # Prefer a form that is an entire element text / attribute value.
        i, cand, where, whole = next((h for h in hits if h[3]), hits[0])
        if i == 0:
            status = "found" if whole else "found-partial"
            detail = "" if whole else "value appears inside a longer text on the page"
        else:
            status = "found-variant"
            detail = (f"recorded {pr.values[0]!r}; matched form {cand!r}"
                      + ("" if whole else " (inside a longer text)"))
        return Located(status, value=cand, where=where, detail=detail)
    if pr.kind == "prefix":
        prefix = pr.values[0]
        full, n = _complete_prefix(page, prefix)
        if full:
            return Located("prefix", value=full, where=page.where(prefix + ELLIPSIS, "string"),
                           detail=f"recorded text was abridged; completed from the page element "
                                  f"starting with it ({n} match{'es' if n != 1 else ''})")
        return Located("not-found", detail="abridged text not found in the saved page")
    # list
    found_items, where = [], set()
    for cands in pr.items:
        hit = None
        for c in cands:
            w = page.where(c, ftype)
            if w:
                hit = c
                where.update(w)
                break
        found_items.append(hit)
    hits = [h for h in found_items if h]
    if pr.complete and hits and all(found_items):
        return Located("list-complete", value=hits, where=sorted(where))
    completed = _complete_list(page, hits)
    if completed and (pr.expected_count is None or len(completed) == pr.expected_count):
        return Located("list-completed", value=completed, where=sorted(where),
                       detail=f"recorded list was abridged; {len(completed)} items read from the "
                              "page container of the recorded items"
                              + (f" (recorded count {pr.expected_count})" if pr.expected_count else ""))
    if hits:
        return Located("list-partial", value=hits, where=sorted(where),
                       detail=f"{len(hits)}/{len(pr.items)} recorded items found"
                              + (f"; page container has {len(completed)} items" if completed else ""))
    return Located("not-found", detail="no recorded list item found in the saved page")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text or "")).strip()


def _complete_prefix(page: PageIndex, prefix: str) -> tuple[str, int]:
    """Text of the deepest element whose text starts with ``prefix``."""
    if page.root is None:
        return "", 0
    key = _text(prefix)
    matches = []
    for el in page.root.iter():
        if not isinstance(el.tag, str):
            continue
        t = _norm(el.text_content())
        if _text(t).startswith(key):
            matches.append((el, t))
    # deepest: drop elements that have a matching descendant
    deep = [(el, t) for el, t in matches
            if not any(other is not el and _inside(other, el) for other, _ in matches)]
    if not deep:
        return "", 0
    texts = {t for _, t in deep}
    return (deep[0][1] if len(texts) == 1 else ""), len(deep)


def _find_node(root, value: str):
    """Deepest element whose text equals ``value`` (or starts with it, for an
    abridged value ending in an ellipsis)."""
    prefix = value.endswith(ELLIPSIS)
    key = _text(value[:-1] if prefix else value)
    best = None
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        t = _text(_norm(el.text_content()))
        if (t.startswith(key) if prefix else t == key):
            best = el  # later in document order = deeper for nested equal texts
    return best


def _complete_list(page: PageIndex, items: list[str]) -> list[str]:
    """Items of the container that holds the recorded items (same relative path)."""
    if page.root is None or not items:
        return []
    nodes = [n for n in (_find_node(page.root, it) for it in items) if n is not None]
    if not nodes:
        return []
    if len(nodes) == 1:
        anc = nodes[0].getparent()
        branch_path = [nodes[0].tag]
    else:
        chains = [list(n.iterancestors()) for n in nodes]
        common = [a for a in chains[0] if all(a in c for c in chains[1:])]
        if not common:
            return []
        anc = common[0]
        branch_path = []
        el = nodes[0]
        while el is not None and el.getparent() is not anc:
            branch_path.insert(0, el.tag)
            el = el.getparent()
        if el is None:
            return []
        branch_path.insert(0, el.tag)
    if anc is None:
        return []
    out = []
    for child in anc:
        if not isinstance(child.tag, str) or child.tag != branch_path[0]:
            continue
        el = child
        for tag in branch_path[1:]:
            el = next((c for c in el if isinstance(c.tag, str) and c.tag == tag), None)
            if el is None:
                break
        if el is not None:
            t = _norm(el.text_content())
            if t:
                out.append(t)
    return out


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------
def _yes(v) -> bool:
    return str(v or "").strip().lower() in YES


def field_config(f: dict) -> dict:
    dtype = str(f.get("type") or "text").strip().lower()
    out = {
        "name": str(f.get("field_name") or f.get("name")).strip().lower(),
        "type": TYPE_MAP.get(dtype, "string"),
        "cardinality": CARD_MAP.get(str(f.get("cardinality") or "single").strip().lower(), "single"),
        "required": _yes(f.get("required")),
        "description": str(f.get("description") or "").strip(),
        "source": str(f.get("source") or "visible").strip().lower(),
        "dynamic": _yes(f.get("dynamic")),
        "dataset_value": "" if f.get("current_value") is None else str(f.get("current_value")),
        "dataset_type": dtype,
    }
    return out


def expect_texts(fields: list[dict], limit: int = 3) -> list[str]:
    """Short literal values of required visible text fields: if none of them is
    on a fetched page, the page is a consent wall, bot check or empty shell."""
    preferred = ("title", "headline", "name", "job_title", "company", "symbol")
    cands = []
    for f in fields:
        if not f["required"] or f["cardinality"] != "single" or f["type"] != "string":
            continue
        if "visible" not in f["source"]:
            continue
        pr = probes(f)
        if pr.kind != "literal" or pr.values[0] != f["dataset_value"].strip():
            continue
        v = pr.values[0]
        if 4 <= len(v) <= 150:
            rank = 0 if any(p in f["name"] for p in preferred) else 1
            cands.append((rank, v))
    cands.sort(key=lambda t: t[0])
    return [v for _, v in cands[:limit]]


def to_config(dataset: dict, *, name: str, repetitions: int, planned_repetitions: int | None,
              provider: str = "groq", model: str | None = None, render: str = "auto",
              user_agent: str | None = None, date_order: str = "mdy",
              max_input_chars: int | None = None, reasoning_effort: str | None = None,
              include_reasoning: bool | None = None, tokens_per_minute: int | None = None) -> dict:
    sites = []
    for s in dataset["sites"]:
        fields = [field_config(f) for f in s.get("fields", [])]
        sites.append({
            "key": str(s.get("site_id")).strip().lower(),
            "name": str(s.get("website") or s.get("site_id")).strip(),
            "url": str(s.get("url")).strip(),
            "category": CATEGORY_MAP.get(str(s.get("category") or "").strip().lower(), "other"),
            "page_type": str(s.get("page_type") or "").strip(),
            "website": str(s.get("website") or "").strip(),
            "notes": str(s.get("notes") or "").strip(),
            "expect_text": expect_texts(fields),
            "fields": fields,
        })
    llm = {"provider": provider}
    if model:
        llm["model"] = model
    if max_input_chars:
        llm["max_input_chars"] = max_input_chars
    if reasoning_effort:
        llm["reasoning_effort"] = reasoning_effort
    if include_reasoning is not None:
        llm["include_reasoning"] = include_reasoning
    if tokens_per_minute:
        llm["tokens_per_minute"] = tokens_per_minute
        llm["completion_counts_in_tpm"] = True
    meta = dataset.get("meta") or {}
    return {
        "experiment": {
            "name": name,
            "description": (f"Self-healing scraper evaluation on {len(sites)} websites "
                            f"(dataset: {meta.get('dataset_name') or meta.get('source_file') or 'annotated'}"
                            f", verified {meta.get('verified_at', 'n/a')})"),
            "llm": llm,
            "health_threshold": 0.80,
            "max_heal_attempts": 3,
            "failure_memory_size": 10,
            "repetitions": repetitions,
            "planned_repetitions": planned_repetitions,
            "seed": 20261004,
            "date_order": date_order,
            # Static (LLM selectors), static (rule selectors), rule-based and
            # LLM-based self-healing, all on the same pages (see COND_INFO).
            "conditions": ["self_healing", "static", "rule_based", "rule_static"],
            "fetch": {"user_agent": user_agent, "timeout": 45, "respect_robots": True,
                      "render": render},
            "list_correct_f1": 0.9,
        },
        "sites": sites,
    }
