"""
Rule-based XPath generator and repairer — the "Rule-Based Self-Healing"
module of the evaluation (no LLM).

It plugs into the same detection / healing loop as the LLM (same threshold,
attempt limit and failure memory) and only replaces how selectors are written.
It works on the full cleaned DOM (it has no token limit) and is deterministic.

Generation rules, tried in order for each field until one yields a valid,
type-correct value on the current page:

1. **Value anchoring** (repair only). The value extracted for the field in the
   previous run is looked up on the changed page; a new locator is written for
   the element that now holds it. This is the classic wrapper-repair idea of
   relocating known content, as used by example-driven scrapers.
2. **Label anchoring.** A short element whose text matches the field's name or
   description ("Tax", "Price (excl. tax)", "Previous Close") is taken as a
   label and the value is read from the cell next to it
   (``//th[normalize-space()="Tax"]/following-sibling::td[1]``).
3. **Semantic attributes.** Class, id and microdata tokens matching the field
   name or its synonyms (``price``, ``author``, ``breadcrumb``...).
4. **Type conventions.** The ``<h1>`` for titles/headlines, ``<time
   @datetime>`` for dates.
5. **Structured metadata.** JSON-LD keys and ``<meta>`` names in the
   page-metadata block whose words match the field name (``datePublished``,
   ``offers/price``, ``og:url``); tried first when the field's description
   mentions JSON-LD / meta / schema.org.

Selectors already known to be broken (failure memory) are never reused.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from ..services.extractor import extract_one
from ..services.htmlutil import parse_html
from ..services.matching import canonical
from .base import FailureHint, FieldSpec, LLMProvider, LLMResult
from .heuristic import FIELD_SYNONYMS

METADATA = '//section[@id="page-metadata"]'
LABEL_TAGS = {"th", "dt", "td", "span", "div", "label", "strong", "b", "p", "li", "h2", "h3",
              "h4", "h5", "h6", "em", "small", "dfn", "abbr"}
SKIP_TAGS = {"html", "body", "head", "script", "style", "section"}
STOP = {"the", "a", "an", "of", "for", "and", "to", "in", "on", "at", "by", "with", "from",
        "value", "field", "page", "text", "shown", "main", "current", "displayed", "number",
        "its", "this", "as", "or", "is", "e", "g", "eg", "id"}
# Abbreviations seen in labels vs. field names.
ALIASES = {
    "excl": "excluding", "incl": "including", "qty": "quantity", "num": "number",
    "pct": "percent", "avg": "average", "vol": "volume", "cap": "capitalization",
    "pe": "p/e", "ttm": "ttm", "yr": "year", "1y": "year", "52": "52",
}
STRUCTURED_HINT = re.compile(r"json-?ld|schema\.org|<meta|\bmeta\b|og:|structured", re.I)
TITLE_WORDS = {"title", "headline", "name"}
# Words that name the page's entity, not the field ("product_title", "job_url").
GENERIC = {"product", "item", "article", "job", "page", "paper", "company", "site", "main"}
NUMBER_WORD = re.compile(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)


def _words(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")          # camelCase
    return [w for w in re.split(r"[^a-z0-9%/]+", text.lower()) if w]


def _key_words(text: str) -> set[str]:
    return {w for w in _words(text) if w not in STOP}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _lit(value: str) -> str:
    """XPath string literal for any value."""
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    parts = value.split('"')
    return "concat(" + ", '\"', ".join(f'"{p}"' for p in parts) + ")"


def _token_match(field_word: str, label_words: set[str]) -> bool:
    alias = ALIASES.get(field_word)
    for w in label_words:
        if w == field_word or (alias and w == alias):
            return True
        if len(field_word) >= 3 and (w.startswith(field_word) or field_word.startswith(w) and len(w) >= 3):
            return True
    return False


def _label_like(text: str) -> bool:
    """A label is words, not a value: mostly letters, no currency amount."""
    letters = sum(c.isalpha() for c in text)
    return letters >= 2 and letters >= 0.5 * len(text.replace(" ", "")) and not re.search(r"[$€£¥]\s?\d", text)


def _in_metadata(el) -> bool:
    parent = el
    while parent is not None:
        if parent.get("id") == "page-metadata":
            return True
        parent = parent.getparent()
    return False


class RuleBasedProvider(LLMProvider):
    name = "rule_based"
    model = "rule-based-v1"
    #: the pipeline passes the full cleaned DOM, not the token-limited LLM view
    wants_clean_dom = True

    def generate_xpaths(self, pruned_html: str, fields: list[FieldSpec],
                        failure_history: Optional[list[FailureHint]] = None) -> LLMResult:
        start = time.perf_counter()
        tree = parse_html(pruned_html)
        if tree is None:
            return LLMResult(xpaths={}, model=self.model)
        hints = failure_history or []
        banned = {h.failed_xpath for h in hints}
        last = {h.field_name: h.last_value for h in hints if getattr(h, "last_value", "")}
        self._elements = [e for e in tree.iter() if isinstance(e.tag, str) and e.tag not in SKIP_TAGS]
        self._text_cache: dict = {}
        out, how = {}, {}
        for spec in fields:
            strategies = []
            if spec.name in last:
                strategies.append(("value-anchor", lambda s=spec: self._value_anchor(tree, s, last[s.name])))
            dom = [("label", self._label), ("attributes", self._attributes), ("type", self._type_rule)]
            structured = [("structured", self._structured)]
            order = structured + dom if STRUCTURED_HINT.search(spec.description or "") else dom + structured
            strategies += [(n, lambda f=f, s=spec: f(tree, s)) for n, f in order]
            for strategy, fn in strategies:
                for xp in fn() or []:
                    if xp in banned:
                        continue
                    if extract_one(tree, spec.name, spec.field_type, spec.cardinality, xp).success:
                        out[spec.name], how[spec.name] = xp, strategy
                        break
                if spec.name in out:
                    break
        return LLMResult(
            xpaths=out, model=self.model,
            raw_response=repr({k: [how[k], v] for k, v in out.items()}),
            latency_ms=int((time.perf_counter() - start) * 1000),
            input_chars=len(pruned_html),
        )

    # ------------------------------------------------------------------
    def _text(self, el) -> str:
        t = self._text_cache.get(el)
        if t is None:
            t = _norm(el.text_content())
            self._text_cache[el] = t
        return t

    @staticmethod
    def _field_words(spec: FieldSpec) -> list[str]:
        return [w for w in _words(spec.name) if w not in STOP]

    # 1. value anchoring ------------------------------------------------
    def _value_anchor(self, tree, spec: FieldSpec, value: str) -> list[str]:
        target = canonical(value, spec.field_type)
        if not target:
            return []
        hits = []
        for el in self._elements:
            if _in_metadata(el):
                continue
            t = self._text(el)
            if t and len(t) <= max(400, 2 * len(value)) and canonical(t, spec.field_type) == target:
                hits.append(el)
        # deepest elements only
        deep = [e for e in hits if not any(c in hits for c in e.iterdescendants())]
        out = []
        for el in deep[:3]:
            if spec.cardinality == "many":
                out += self._group_locators(tree, el)
            else:
                out += self._locators(tree, el)
        return out

    def _locators(self, tree, el) -> list[str]:
        """XPaths that select ``el`` (most robust first)."""
        cands = []
        label = self._label_before(el)
        if label is not None:
            cands.append(self._label_xpath(label, el))
        if el.get("id"):
            cands.append(f"//*[@id={_lit(el.get('id'))}]")
        if el.get("itemprop"):
            cands.append(f"//{el.tag}[@itemprop={_lit(el.get('itemprop'))}]")
        for cls in (el.get("class") or "").split():
            cands.append(f'//{el.tag}[contains(concat(" ", normalize-space(@class), " "), " {cls} ")]')
        cands.append(self._path(el))
        good = []
        for xp in cands:
            try:
                nodes = tree.xpath(xp)
            except Exception:
                continue
            if nodes and nodes[0] is el:
                good.append(xp)
            elif el in nodes:
                good.append(f"({xp})[{nodes.index(el) + 1}]")
        return good

    def _group_locators(self, tree, el) -> list[str]:
        """XPaths that select ``el`` and its same-shaped siblings (list fields)."""
        parent = el.getparent()
        if parent is None:
            return []
        out = []
        for pxp in self._locators(tree, parent):
            out.append(f"{pxp}/{el.tag}")
        return out

    def _path(self, el) -> str:
        """Path from the nearest ancestor with an id or class, with positions."""
        steps = []
        node = el
        while node is not None and isinstance(node.tag, str) and node.tag not in ("html", "body"):
            parent = node.getparent()
            if node is not el and (node.get("id") or node.get("class")):
                if node.get("id"):
                    anchor = f"//*[@id={_lit(node.get('id'))}]"
                else:
                    cls = node.get("class").split()[0]
                    anchor = f'//{node.tag}[contains(concat(" ", normalize-space(@class), " "), " {cls} ")]'
                return anchor + "/" + "/".join(reversed(steps))
            same = [c for c in parent if isinstance(c.tag, str) and c.tag == node.tag] if parent is not None else [node]
            steps.append(f"{node.tag}[{same.index(node) + 1}]" if len(same) > 1 else node.tag)
            node = parent
        return "/html/body/" + "/".join(reversed(steps))

    # 2. label anchoring --------------------------------------------------
    def _label_phrases(self, spec: FieldSpec) -> list[set[str]]:
        phrases = [set(self._field_words(spec))]
        desc = re.split(r"[(,;:]", spec.description or "")[0]
        dw = _key_words(desc)
        if dw and len(dw) <= 6:
            phrases.append(dw)
        return [p for p in phrases if p]

    def _label_candidates(self, spec: FieldSpec):
        phrases = self._label_phrases(spec)
        scored = []
        for el in self._elements:
            if el.tag not in LABEL_TAGS or _in_metadata(el):
                continue
            t = self._text(el)
            if not t or len(t) > 50 or not _label_like(t):
                continue
            if sum(1 for c in el if isinstance(c.tag, str)) > 1:
                continue
            words = _key_words(t)
            if not words:
                continue
            for phrase in phrases:
                hit = sum(1 for w in phrase if _token_match(w, words))
                if hit == len(phrase):
                    extra = len(words) - hit
                    scored.append((extra, len(t), el))
                    break
        scored.sort(key=lambda s: (s[0], s[1]))
        return [el for _, _, el in scored]

    def _value_after(self, label):
        """The element holding the value of a label element."""
        if label.tag == "th":
            sib = label.getnext()
            while sib is not None and not (isinstance(sib.tag, str) and sib.tag == "td"):
                sib = sib.getnext()
            return sib
        if label.tag == "dt":
            sib = label.getnext()
            while sib is not None and not (isinstance(sib.tag, str) and sib.tag == "dd"):
                sib = sib.getnext()
            return sib
        sib = label.getnext()
        while sib is not None and not isinstance(sib.tag, str):
            sib = sib.getnext()
        if sib is not None:
            return sib
        parent = label.getparent()
        if parent is not None and len([c for c in parent if isinstance(c.tag, str)]) == 1:
            nxt = parent.getnext()
            while nxt is not None and not isinstance(nxt.tag, str):
                nxt = nxt.getnext()
            return nxt
        return None

    def _label_before(self, el):
        prev = el.getprevious()
        while prev is not None and not isinstance(prev.tag, str):
            prev = prev.getprevious()
        if prev is not None and prev.tag in LABEL_TAGS and 0 < len(self._text(prev)) <= 50 \
                and sum(1 for c in prev if isinstance(c.tag, str)) <= 1 and _label_like(self._text(prev)):
            return prev
        return None

    def _label_xpath(self, label, value_el) -> str:
        text = self._text(label)
        base = f"//{label.tag}[normalize-space()={_lit(text)}]"
        if value_el.getparent() is label.getparent():
            return f"{base}/following-sibling::{value_el.tag}[1]"
        return f"{base}/../following-sibling::{value_el.tag}[1]"

    def _label(self, tree, spec: FieldSpec) -> list[str]:
        if spec.cardinality == "many":
            return []
        out = []
        for label in self._label_candidates(spec)[:4]:
            value = self._value_after(label)
            if value is not None:
                out.append(self._label_xpath(label, value))
        return out

    # 3. semantic attributes ----------------------------------------------
    def _attributes(self, tree, spec: FieldSpec) -> list[str]:
        words = self._field_words(spec)
        tokens = list(dict.fromkeys(words + [s for w in words for s in FIELD_SYNONYMS.get(w, [])]))
        scored = []
        for el in self._elements:
            if _in_metadata(el):
                continue
            attrs = " ".join(filter(None, [el.get("class"), el.get("id"), el.get("itemprop"),
                                           el.get("data-testid"), el.get("data-test")])).lower()
            if not attrs:
                continue
            parts = set(re.split(r"[^a-z0-9]+", attrs))
            specific = [w for w in words if w not in GENERIC] or words
            hit = sum(1 for w in specific if any(p.startswith(w) or w in p for p in parts if p))
            syn = any(t in attrs for t in tokens if t not in GENERIC)
            if not hit and not syn:
                continue
            t = self._text(el)
            kids = sum(1 for c in el if isinstance(c.tag, str))
            score = 3 * hit + (1 if syn else 0) - 0.3 * min(kids, 5) - (2 if not t else 0) \
                - (1 if len(t) > 400 else 0)
            scored.append((score, el))
        scored.sort(key=lambda s: -s[0])
        out = []
        for _, el in scored[:5]:
            locs = self._group_locators(tree, el) if spec.cardinality == "many" else []
            if spec.cardinality == "many":
                kids = [c for c in el if isinstance(c.tag, str)]
                if len(kids) >= 2 and len({c.tag for c in kids}) == 1:
                    locs = [f"{xp}/{kids[0].tag}" for xp in self._locators(tree, el)]
            locs = locs or self._locators(tree, el)
            if spec.field_type == "number" and not self._text(el) and NUMBER_WORD.search(el.get("class") or ""):
                locs = [f"{xp}/@class" for xp in locs]  # rating widget: class="star-rating Three"
            out += locs
        return out

    # 4. type conventions -------------------------------------------------
    def _type_rule(self, tree, spec: FieldSpec) -> list[str]:
        words = set(self._field_words(spec))
        out = []
        if words & TITLE_WORDS and spec.cardinality == "single":
            out += ["(//main//h1)[1]", "(//article//h1)[1]", "(//h1)[1]"]
        if spec.field_type == "date":
            out += ["(//main//time/@datetime)[1]", "(//time/@datetime)[1]", "(//time)[1]"]
        if spec.field_type == "url" and "image" in words:
            out += ["(//main//img/@src)[1]", "(//img/@src)[1]"]
        return out

    # 5. structured metadata ---------------------------------------------
    def _structured(self, tree, spec: FieldSpec) -> list[str]:
        words = set(self._field_words(spec))
        if not words:
            return []
        expanded = set(words)
        for w in words:
            expanded.update(FIELD_SYNONYMS.get(w, []))
        scored = []
        for el in tree.xpath(f"{METADATA}//div[@data-key]"):
            if any(isinstance(c.tag, str) for c in el):   # leaves only
                continue
            keys = []
            node = el
            while node is not None and node.get("data-key") is not None:
                keys.insert(0, node.get("data-key"))
                node = node.getparent()
            kw = set()
            for k in keys:
                kw |= _key_words(k)
            hit = sum(1 for w in words if _token_match(w, kw))
            if hit == 0 or hit < 0.6 * len(words):
                continue
            xp = f"{METADATA}//" + "/".join(f'div[@data-key={_lit(k)}]' for k in keys)
            scored.append((-hit, len(keys), xp))
        for el in tree.xpath(f"{METADATA}/meta"):
            attr = "property" if el.get("property") else ("name" if el.get("name") else "itemprop")
            name = el.get(attr) or ""
            kw = _key_words(name.replace(":", " "))
            hit = sum(1 for w in expanded if _token_match(w, kw))
            if hit == 0 or sum(1 for w in words if _token_match(w, kw)) < 0.5 * len(words) \
                    and not ({"summary", "description", "standfirst"} & words and "description" in kw):
                continue
            scored.append((-hit, 1, f"{METADATA}/meta[@{attr}={_lit(name)}]/@content"))
        if spec.field_type == "url" and "url" in words:
            scored.append((-1, 0, f'{METADATA}/link[@rel="canonical"]/@href'))
            scored.append((-1, 1, f'{METADATA}/meta[@property="og:url"]/@content'))
        if words & TITLE_WORDS:
            scored.append((-1, 2, f'{METADATA}/div[@data-meta="title"]'))
        scored.sort()
        seen, out = set(), []
        for _, _, xp in scored:
            if xp not in seen:
                seen.add(xp)
                out.append(f"({xp})[1]" if spec.cardinality == "single" and "@" not in xp.split("/")[-1] else xp)
        return out
