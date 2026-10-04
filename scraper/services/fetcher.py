"""
Layer 1 — Fetcher Layer (proposal Section 1.4; cleaning pipeline §4.3 steps 1-2).

Produces two deterministic views of a page:

* ``clean_html`` — the extraction DOM. Noise removed (scripts, styles,
  comments, hidden and ad blocks, page chrome, form controls), attributes
  reduced to a whitelist, whitespace collapsed, empty nodes and attribute-less
  single-child wrappers removed. XPaths are always evaluated against this DOM.

* ``llm_view`` — what the LLM sees: the clean DOM with long text and long
  attribute values truncated and long runs of near-identical siblings
  (e.g. 50 product cards) collapsed to the first few plus a marker comment.
  Element structure, tags and kept attributes are identical to the clean DOM,
  so selectors written against the view work on the clean DOM.

Both are pure functions of the raw HTML (plus the per-site ``keep_tags``), so
the same snapshot always yields the same XPath targets.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests
from django.conf import settings
from lxml import etree

from . import tokens as tok
from .change_detector import dom_profiles
from .htmlutil import decode_html, parse_html, to_html

# Whole subtrees that never carry target data.
DROP_TAGS = {
    "head", "script", "style", "noscript", "template", "svg", "iframe",
    "canvas", "video", "audio", "picture", "source", "track", "object",
    "embed", "map", "link", "meta", "dialog",
}
# Page chrome: removed only when outside <main>/<article> (an article's own
# <header> usually holds the headline, author and date — proposal News fields).
CHROME_TAGS = {"nav", "header", "footer", "aside"}
CONTENT_REGION = {"main", "article"}
# Interactive controls: removed (their visible text is UI, not data).
CONTROL_TAGS = {
    "button", "input", "select", "option", "optgroup", "textarea",
    "datalist", "output", "progress", "meter",
}
# Kept but unwrapped (children survive): prices often live inside <form>.
UNWRAP_TAGS = {"form", "font", "center"}

KEEP_ATTRS = {
    "class", "id", "itemprop", "itemtype", "itemscope", "property", "content",
    "datetime", "href", "src", "title", "alt", "aria-label", "name", "rel",
    "lang", "role", "data-testid", "data-test", "data-qa", "data-field",
    # attributes of the page-metadata block (see _metadata_block)
    "data-key", "data-jsonld", "data-meta",
}
VALUE_ATTRS = ("content", "datetime", "alt", "aria-label", "title")
VALUE_CLASS = re.compile(r"rating|\bstars?\b", re.I)
KEEP_EMPTY = {"img", "br", "hr", "td", "th", "tr", "time", "meta", "link"}
COLLAPSIBLE = {"div", "span", "section"}
INLINE = {
    "a", "abbr", "b", "bdi", "bdo", "cite", "code", "data", "dfn", "em", "i",
    "kbd", "label", "mark", "q", "s", "samp", "small", "span", "strong", "sub",
    "sup", "time", "u", "var",
}

AD_TOKEN = re.compile(
    r"^(ad|ads|adv|advert|advertisement|advertising|adsbygoogle|sponsored|"
    r"sponsor|dfp|gpt-ad|google-ad|ad-slot|ad-unit|ad-container|ad-wrapper|"
    r"ad-banner)$|^(ad|ads|advert)[-_]",
    re.I,
)
HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)
_WS = re.compile(r"\s+")

# LLM-view compression parameters.
TEXT_LIMIT = 200
ATTR_LIMIT = 80
RUN_KEEP = 3
RUN_MIN = 6

BLOCK_MARKERS = re.compile(
    r"captcha|are you a robot|access denied|unusual traffic|"
    r"cf-browser-verification|attention required|verify you are human",
    re.I,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PrunedPage:
    clean_html: str = ""
    llm_view: str = ""
    tag_profile: dict = field(default_factory=dict)
    markup_profile: dict = field(default_factory=dict)
    tokenizer: str = ""
    raw_tokens: int = 0
    clean_tokens: int = 0
    llm_view_tokens: int = 0
    raw_chars: int = 0
    clean_chars: int = 0
    llm_view_chars: int = 0

    @property
    def token_reduction(self) -> float:
        if not self.raw_tokens:
            return 0.0
        return 1.0 - (self.llm_view_tokens / self.raw_tokens)

    @property
    def empty(self) -> bool:
        return not self.clean_html


@dataclass
class FetchResult:
    url: str
    http_status: int
    raw_html: str
    page: PrunedPage
    raw_html_signature: str = ""
    dom_signature: str = ""
    error: str = ""
    fetch_meta: dict = field(default_factory=dict)

    # Convenience accessors used by the pipeline.
    @property
    def pruned_html(self) -> str:  # what the LLM sees
        return self.page.llm_view

    @property
    def clean_html(self) -> str:  # where XPaths are evaluated
        return self.page.clean_html

    @property
    def token_reduction(self) -> float:
        return self.page.token_reduction


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------
def _has_ancestor(el, tags) -> bool:
    parent = el.getparent()
    while parent is not None:
        if isinstance(parent.tag, str) and parent.tag in tags:
            return True
        parent = parent.getparent()
    return False


def _contains(el, tags) -> bool:
    return any(isinstance(d.tag, str) and d.tag in tags for d in el.iterdescendants())


def _remove(el) -> None:
    """Remove an element but keep its tail text attached to the document."""
    parent = el.getparent()
    if parent is None:
        return
    tail = el.tail
    prev = el.getprevious()
    parent.remove(el)
    if tail and tail.strip():
        if prev is not None:
            prev.tail = (prev.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail


def _collapse_ws(text, inline: bool):
    if text is None:
        return None
    collapsed = _WS.sub(" ", text)
    if not collapsed.strip():
        return " " if inline else None
    return collapsed


def _is_hidden(el) -> bool:
    if el.get("hidden") is not None:
        return True
    style = el.get("style")
    return bool(style and HIDDEN_STYLE.search(style))


def _is_ad(el) -> bool:
    tokens = (el.get("class") or "").split()
    el_id = el.get("id")
    if el_id:
        tokens.append(el_id)
    return any(AD_TOKEN.search(t) for t in tokens)


def _clean_tree(root, keep_tags=()) -> None:
    keep = {t.lower() for t in keep_tags}

    # Comments / processing instructions.
    for node in root.xpath("//comment() | //processing-instruction()"):
        parent = node.getparent()
        if parent is not None:
            _remove(node)

    drop = DROP_TAGS - keep
    chrome = CHROME_TAGS - keep
    controls = CONTROL_TAGS - keep

    for el in list(root.iter()):
        if not isinstance(el.tag, str) or el.getparent() is None and el is not root:
            continue
        tag = el.tag.lower()
        if el is root:
            continue
        # Body microdata (<meta itemprop="price" content="...">) is kept: it is
        # often the most stable place a value lives.
        if tag in ("meta", "link") and el.get("itemprop") and tag in drop:
            continue
        if tag in drop or tag in controls:
            _remove(el)
        elif tag in chrome and not _has_ancestor(el, CONTENT_REGION) and not _contains(el, CONTENT_REGION):
            _remove(el)
        elif _is_hidden(el) or _is_ad(el):
            _remove(el)

    for el in list(root.iter()):
        if isinstance(el.tag, str) and el.tag.lower() in UNWRAP_TAGS and el is not root:
            el.drop_tag()

    # Attribute whitelist + whitespace normalisation.
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        for attr in list(el.attrib):
            if attr.lower() not in KEEP_ATTRS:
                del el.attrib[attr]
        inline = el.tag in INLINE
        el.text = _collapse_ws(el.text, inline)
        el.tail = _collapse_ws(el.tail, inline)

    # Remove empty elements bottom-up.
    changed = True
    while changed:
        changed = False
        for el in list(root.iter()):
            if el is root or not isinstance(el.tag, str):
                continue
            if el.tag in KEEP_EMPTY or len(el):
                continue
            if (el.text or "").strip():
                continue
            if any(el.get(a) for a in VALUE_ATTRS):
                continue
            if VALUE_CLASS.search(el.get("class") or ""):
                continue  # rating widgets encode the value in the class ("star-rating Three")
            if el.tag == "a" and el.get("href") and (el.get("title") or el.get("aria-label")):
                continue
            _remove(el)
            changed = True

    # Collapse attribute-less single-child wrappers (div > div > span ...).
    changed = True
    while changed:
        changed = False
        for el in list(root.iter()):
            if el is root or not isinstance(el.tag, str) or el.tag not in COLLAPSIBLE:
                continue
            if el.attrib or (el.text or "").strip() or len(el) != 1:
                continue
            child = el[0]
            if not isinstance(child.tag, str) or (child.tail or "").strip():
                continue
            parent = el.getparent()
            if parent is None:
                continue
            child.tail = el.tail
            parent.replace(el, child)
            changed = True



# ---------------------------------------------------------------------------
# Page metadata block (structured data exposed to XPath)
# ---------------------------------------------------------------------------
METADATA_ID = "page-metadata"
# <meta> names that never carry page data.
_META_SKIP = re.compile(
    r"^(viewport|theme-color|color-scheme|referrer|robots|googlebot|format-detection|"
    r"google-site-verification|msvalidate\.01|yandex-verification|apple-itunes-app|"
    r"apple-mobile-web-app.*|mobile-web-app-capable|msapplication-.*|csrf.*|"
    r"generator|handheldfriendly|mobileoptimized)$",
    re.I,
)
_JSONLD_MAX_NODES = 4000
_JSONLD_MAX_DEPTH = 12


def structured_data_enabled() -> bool:
    try:
        return bool(settings.SCRAPER.get("STRUCTURED_DATA", True))
    except Exception:  # pragma: no cover - settings not configured
        return True


def _collect_metadata(root) -> dict:
    """Read <title>, document-level <meta>/<link rel=canonical> and JSON-LD
    from the raw tree (before cleaning removes them)."""
    title = ""
    t = root.find(".//title")
    if t is not None:
        title = _WS.sub(" ", t.text_content() or "").strip()
    metas, canonical, jsonld = [], "", []
    for el in root.iter("meta"):
        if el.get("itemprop") and _has_ancestor(el, {"body"}):
            continue  # body microdata stays where it is
        key_attr = next((a for a in ("property", "name", "itemprop") if el.get(a)), None)
        content = (el.get("content") or "").strip()
        if not key_attr or not content or _META_SKIP.match(el.get(key_attr).strip()):
            continue
        metas.append((key_attr, el.get(key_attr).strip(), _WS.sub(" ", content)))
    for el in root.iter("link"):
        if "canonical" in (el.get("rel") or "").lower().split() and el.get("href"):
            canonical = el.get("href").strip()
            break
    for el in root.iter("script"):
        if (el.get("type") or "").strip().lower() != "application/ld+json":
            continue
        text = (el.text or "").strip()
        if text.startswith("<!--"):
            text = text[4:]
        if text.endswith("-->"):
            text = text[:-3]
        text = text.strip().removeprefix("//<![CDATA[").removesuffix("//]]>").strip()
        # Numbers are kept as written ("59.00", not 59.0).
        load = lambda t: json.loads(t, parse_float=str, parse_int=str)  # noqa: E731
        try:
            jsonld.append(load(text))
        except ValueError:
            try:  # tolerate trailing commas, a frequent JSON-LD error
                jsonld.append(load(re.sub(r",\s*([}\]])", r"\1", text)))
            except ValueError:
                continue
    return {"title": title, "metas": metas, "canonical": canonical, "jsonld": jsonld}


def _json_nodes(key, value, depth, budget) -> list:
    if budget[0] <= 0 or depth > _JSONLD_MAX_DEPTH or value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_json_nodes(key, item, depth, budget))
        return out
    el = etree.Element("div")
    if key is not None:
        el.set("data-key", xml_safe(str(key))[:80])
    budget[0] -= 1
    if isinstance(value, dict):
        for k, v in value.items():
            for child in _json_nodes(k, v, depth + 1, budget):
                el.append(child)
        return [el] if len(el) else []
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = xml_safe(_WS.sub(" ", str(value))).strip()
    if not text:
        return []
    el.text = text
    return [el]


def _metadata_block(meta: dict):
    """<section id="page-metadata"> holding the document title, meta tags,
    the canonical link and every JSON-LD block as nested div[data-key]."""
    section = etree.Element("section")
    section.set("id", METADATA_ID)
    if meta["title"]:
        t = etree.SubElement(section, "div")
        t.set("data-meta", "title")
        t.text = xml_safe(meta["title"])
    for attr, name, content in meta["metas"]:
        m = etree.SubElement(section, "meta")
        m.set(attr, xml_safe(name))
        m.set("content", xml_safe(content))
    if meta["canonical"]:
        link = etree.SubElement(section, "link")
        link.set("rel", "canonical")
        link.set("href", xml_safe(meta["canonical"]))
    budget = [_JSONLD_MAX_NODES]
    blocks = []
    for i, data in enumerate(meta["jsonld"], start=1):
        if isinstance(data, list):  # a top-level array: one block per object
            blocks.extend((f"{i}.{j}", item) for j, item in enumerate(data, start=1))
        else:
            blocks.append((str(i), data))
    for label, data in blocks:
        nodes = _json_nodes(None, data, 0, budget)
        if not nodes:
            continue
        block = nodes[0]  # the object itself, without a data-key
        block.set("data-jsonld", label)
        section.append(block)
    return section if len(section) else None


def _insert_metadata(root, meta: dict) -> None:
    section = _metadata_block(meta)
    if section is None:
        return
    body = root.find("body")
    if body is None:
        body = root
    section.tail = body.text
    body.text = None
    body.insert(0, section)


def _signature(el) -> tuple:
    return (el.tag, tuple(sorted((el.get("class") or "").split())), el.get("data-key"))


def _truncate(text: str | None, limit: int):
    if text is None or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


# Characters XML 1.0 allows; lxml rejects any other in text, attributes and comments.
_XML_ILLEGAL = re.compile("[^\u0009\u000a\u000d\u0020-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def xml_safe(text: str) -> str:
    """Drop characters that cannot appear in an XML/HTML tree (NUL, control
    characters, lone surrogates) — e.g. from JSON-LD string escapes."""
    return _XML_ILLEGAL.sub("", text)


def _marker_text(n: int, sig: tuple) -> str:
    """Text of the comment that replaces a collapsed run of similar siblings.

    It names the tag, class and data-key of the omitted elements. An XML
    comment may not contain "--" or end with "-", which BEM-style class names
    ("price--sale") or keys can produce; then the marker refers to the
    identical siblings kept just above it instead of quoting the names.
    """
    tag, classes, key = sig
    cls = " ".join(classes)
    text = (f" +{n} more similar <{tag}"
            + (f' class="{cls}"' if cls else "")
            + (f' data-key="{key}"' if key else "")
            + "> elements omitted ")
    if "--" in text or text.rstrip(" ").endswith("-"):
        same = ["tag"] + (["class"] if cls else []) + (["data-key"] if key else [])
        names = same[0] if len(same) == 1 else ", ".join(same[:-1]) + " and " + same[-1]
        text = f" +{n} more similar <{tag}> elements omitted (same {names} as the elements above) "
    return xml_safe(text)


def _llm_view(clean_root) -> str:
    root = copy.deepcopy(clean_root)
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        el.text = _truncate(el.text, TEXT_LIMIT)
        el.tail = _truncate(el.tail, TEXT_LIMIT)
        for attr, value in list(el.attrib.items()):
            if len(value) > ATTR_LIMIT:
                el.attrib[attr] = value[:ATTR_LIMIT] + "…"

    # Collapse long runs of near-identical siblings.
    for parent in list(root.iter()):
        if not isinstance(parent.tag, str):
            continue
        children = [c for c in parent if isinstance(c.tag, str)]
        i = 0
        while i < len(children):
            sig = _signature(children[i])
            j = i
            while j + 1 < len(children) and _signature(children[j + 1]) == sig:
                j += 1
            run = children[i : j + 1]
            if len(run) >= RUN_MIN:
                hidden = run[RUN_KEEP:]
                marker = etree.Comment(_marker_text(len(hidden), sig))
                run[RUN_KEEP - 1].addnext(marker)
                marker.tail = run[RUN_KEEP - 1].tail
                run[RUN_KEEP - 1].tail = None
                for el in hidden:
                    _remove(el)
            i = j + 1
    return to_html(root)


def prune(raw_html: str, keep_tags=(), structured: bool | None = None) -> PrunedPage:
    """Return the clean extraction DOM, the LLM view, profiles and tokens.

    With structured data on (``SCRAPER["STRUCTURED_DATA"]``, default), the
    document title, meta tags, canonical link and JSON-LD blocks — removed with
    <head>/<script> — are re-inserted as a <section id="page-metadata"> at the
    top of <body> so XPath can address them.
    """
    root = parse_html(raw_html)
    if root is None:
        return PrunedPage(tokenizer=tok.tokenizer_name(), raw_chars=len(raw_html or ""),
                          raw_tokens=tok.count_tokens(raw_html or ""))
    if structured is None:
        structured = structured_data_enabled()
    meta = _collect_metadata(root) if structured else None
    _clean_tree(root, keep_tags)
    if meta:
        _insert_metadata(root, meta)
    clean_html = to_html(root)
    view = _llm_view(root)
    tag_profile, markup_profile = dom_profiles(root)
    return PrunedPage(
        clean_html=clean_html,
        llm_view=view,
        tag_profile=tag_profile,
        markup_profile=markup_profile,
        tokenizer=tok.tokenizer_name(),
        raw_tokens=tok.count_tokens(raw_html),
        clean_tokens=tok.count_tokens(clean_html),
        llm_view_tokens=tok.count_tokens(view),
        raw_chars=len(raw_html),
        clean_chars=len(clean_html),
        llm_view_chars=len(view),
    )


def profile_signature(page: PrunedPage) -> str:
    items = sorted(page.tag_profile.items())
    return sha256(repr(items)) if items else ""


def prune_result(url: str, raw_html: str, http_status: int = 200,
                 error: str = "", keep_tags=(), fetch_meta=None) -> FetchResult:
    """Build a FetchResult from already-obtained HTML (snapshots, tests)."""
    page = prune(raw_html, keep_tags)
    return FetchResult(
        url=url,
        http_status=http_status,
        raw_html=raw_html,
        page=page,
        raw_html_signature=sha256(raw_html or ""),
        dom_signature=profile_signature(page),
        error=error,
        fetch_meta=fetch_meta or {},
    )


def prune_html(raw_html: str) -> tuple[str, str]:
    """Backward-compatible helper: (llm_view, tag skeleton string)."""
    page = prune(raw_html)
    if page.empty:
        return "", ""
    clean = parse_html(page.clean_html)
    skeleton = ">".join(
        el.tag for el in clean.iter()
        if isinstance(el.tag, str) and el.tag not in ("html", "body")
    ) if clean is not None else ""
    return page.llm_view, skeleton


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def _browser_headers(user_agent: str) -> dict:
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    }


def robots_allowed(url: str, user_agent: str, timeout: int = 15):
    """True/False per robots.txt, or None if robots.txt could not be read."""
    from urllib import robotparser

    parts = urlparse(url)
    robots_url = urljoin(f"{parts.scheme}://{parts.netloc}", "/robots.txt")
    try:
        resp = requests.get(robots_url, timeout=timeout, headers=_browser_headers(user_agent))
    except requests.RequestException:
        return None
    if resp.status_code >= 400:
        return True  # no robots.txt => no restrictions declared
    parser = robotparser.RobotFileParser()
    parser.parse(resp.text.splitlines())
    return parser.can_fetch(user_agent, url)


def _block_suspected(status: int, html: str) -> bool:
    return status in (401, 403, 429, 503) or bool(BLOCK_MARKERS.search(html[:20000]))


def fetch_raw(url: str, timeout: int = 20, user_agent: str = "SelfHealingScraper/1.0",
              render: str = "static") -> tuple[int, str, dict]:
    """
    Fetch a page and return (http_status, html, meta). Never raises for
    network errors; they are returned in meta["error"] with status 0.
    """
    meta: dict = {"render": render}
    start = time.perf_counter()
    if render == "js":
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            meta["error"] = ("render=js needs Playwright: pip install playwright "
                             "&& python -m playwright install chromium")
            return 0, "", meta
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=user_agent, locale="en-US",
                    viewport={"width": 1366, "height": 900},
                    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                )
                page = context.new_page()
                response = page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
                # Pages with live tickers/ads never go network-idle: bounded wait.
                try:
                    page.wait_for_load_state("networkidle", timeout=12000)
                except Exception:
                    meta["networkidle"] = False
                # Scroll so lazy-loaded sections (reviews, tables) render.
                for _ in range(8):
                    page.mouse.wheel(0, 2400)
                    page.wait_for_timeout(400)
                page.wait_for_timeout(2000)
                html = page.content()
                status = response.status if response else 0
                meta["final_url"] = page.url
                browser.close()
        except Exception as exc:  # pragma: no cover - environment dependent
            meta["error"] = f"render error: {exc}"[:300]
            return 0, "", meta
    else:
        try:
            resp = requests.get(url, timeout=timeout, headers=_browser_headers(user_agent))
        except requests.RequestException as exc:
            meta["error"] = f"fetch error: {exc}"[:300]
            return 0, "", meta
        html, encoding = decode_html(resp.content, resp.headers.get("Content-Type"))
        status = resp.status_code
        meta["final_url"] = resp.url
        meta["encoding"] = encoding
        meta["content_type"] = resp.headers.get("Content-Type", "")
    meta["elapsed_ms"] = int((time.perf_counter() - start) * 1000)
    meta["suspected_block"] = _block_suspected(status, html)
    if status >= 400:
        meta["error"] = f"HTTP {status}"
    return status, html, meta


def fetch(url: str, timeout: int = 20, user_agent: str = "SelfHealingScraper/1.0",
          render: str = "static", keep_tags=()) -> FetchResult:
    status, html, meta = fetch_raw(url, timeout, user_agent, render)
    return prune_result(url, html, http_status=status, error=meta.get("error", ""),
                        keep_tags=keep_tags, fetch_meta=meta)
