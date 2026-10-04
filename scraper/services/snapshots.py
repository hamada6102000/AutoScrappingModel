"""
Snapshot storage (Raw HTML Snapshot — Table 4.1).

Snapshots are written under SCRAPER["DATA_DIR"] as UTF-8 HTML files and
registered in the database with their SHA-256, size and token counts. Every
evaluation run reads its input from a snapshot, which makes results
reproducible and lets external baselines run on byte-identical pages.

For human labelling, baseline/natural snapshots also get a ``.view.html``
companion: the cleaned DOM with no scripts, so the values a person reads are
exactly the values the system sees (a live page may change prices via JS).
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from ..models import GroundTruth, Snapshot, SnapshotKind
from .fetcher import PrunedPage, prune
from .htmlutil import parse_html, searchable_text
from .matching import value_in_text

_CACHE_SIZE = 32
_PAGE_CACHE: "OrderedDict[tuple, PrunedPage]" = OrderedDict()

LABEL_VIEW_HEAD = (
    "<!doctype html><meta charset='utf-8'><title>Labeling view</title>"
    "<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:2em auto;"
    "line-height:1.5;padding:0 1em}img{max-width:120px}"
    "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:2px 6px}"
    ".note{background:#fff8d6;border:1px solid #e5d27a;padding:.6em 1em;margin-bottom:1em}"
    "</style><div class='note'>Labeling view: scripts and page chrome removed. "
    "Copy each ground-truth value exactly as it appears here.</div>"
)


def data_dir() -> Path:
    path = Path(settings.SCRAPER["DATA_DIR"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def slug(text: str, default: str = "site") -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-").lower()
    return value[:60] or default


def experiment_dir(experiment) -> Path:
    base = slug(experiment.name, "experiment") if experiment else "live"
    path = data_dir() / base
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rel(experiment, site_key: str, name: str) -> str:
    base = slug(experiment.name, "experiment") if experiment else "live"
    return f"{base}/snapshots/{slug(site_key)}/{name}"


def file_name(kind: str, label: str = "", repetition=None, step=None) -> str:
    parts = [kind]
    if label:
        parts.append(slug(label))
    if repetition is not None:
        parts.append(f"r{repetition}")
    if step is not None:
        parts.append(f"s{int(step):02d}")
    return "_".join(parts) + ".html"


def write_text(rel: str, text: str) -> Path:
    path = data_dir() / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return path


def read_html(snapshot: Snapshot) -> str:
    with open(data_dir() / snapshot.path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def label_view_path(snapshot: Snapshot) -> Path:
    return data_dir() / snapshot.path.replace(".html", ".view.html")


def pruned_page(snapshot: Snapshot, keep_tags=()) -> PrunedPage:
    """Prune a snapshot (small LRU cache: snapshots are re-read by both conditions)."""
    key = (snapshot.pk, snapshot.sha256, tuple(sorted(keep_tags or ())))
    page = _PAGE_CACHE.get(key)
    if page is None:
        page = prune(read_html(snapshot), keep_tags)
        _PAGE_CACHE[key] = page
        while len(_PAGE_CACHE) > _CACHE_SIZE:
            _PAGE_CACHE.popitem(last=False)
    else:
        _PAGE_CACHE.move_to_end(key)
    return page


def create_snapshot(*, html: str, site_key: str, kind: str, experiment=None, url: str = "",
                    http_status=None, label: str = "", repetition=None, step=None,
                    parent=None, mutation=None, fetch_meta=None, robots_allowed=None,
                    keep_tags=(), label_view: bool = False, name: str | None = None,
                    notes: str = "") -> Snapshot:
    page = prune(html, keep_tags)
    if name is None:
        name = file_name(kind, label, repetition, step)
        if experiment is None and kind == SnapshotKind.LIVE:
            name = f"live_{timezone.now():%Y%m%dT%H%M%S%f}.html"
    rel = _rel(experiment, site_key, name)
    write_text(rel, html)
    max_chars = settings.SCRAPER.get("LLM_MAX_INPUT_CHARS", 60000)
    snap = Snapshot.objects.create(
        experiment=experiment,
        site_key=site_key,
        kind=kind,
        label=label,
        repetition=repetition,
        step=step,
        parent=parent,
        mutation=mutation or {},
        url=url or "",
        http_status=http_status,
        path=rel,
        sha256=hashlib.sha256(html.encode("utf-8", "ignore")).hexdigest(),
        size_bytes=len(html.encode("utf-8", "ignore")),
        tokenizer=page.tokenizer,
        raw_tokens=page.raw_tokens,
        clean_tokens=page.clean_tokens,
        llm_view_tokens=page.llm_view_tokens,
        llm_view_truncated=page.llm_view_chars > max_chars if max_chars else False,
        robots_allowed=robots_allowed,
        fetch_meta=fetch_meta or {},
        notes=notes,
    )
    _PAGE_CACHE[(snap.pk, snap.sha256, tuple(sorted(keep_tags or ())))] = page
    if label_view:
        write_text(rel.replace(".html", ".view.html"), LABEL_VIEW_HEAD + page.clean_html)
    return snap


def save_live(target, fetch_result) -> Snapshot:
    return create_snapshot(
        html=fetch_result.raw_html,
        site_key=target.site_key or slug(target.name),
        kind=SnapshotKind.LIVE,
        url=fetch_result.url,
        http_status=fetch_result.http_status,
        fetch_meta=fetch_result.fetch_meta,
        keep_tags=target.prune_keep or (),
    )


def gt_map(snapshot) -> dict[str, GroundTruth]:
    if snapshot is None:
        return {}
    return {g.field_name: g for g in snapshot.ground_truth.all()}


def compute_field_presence(snapshot: Snapshot, keep_tags=(), field_types=None) -> dict:
    """For each labelled field: is the value in the raw page / clean DOM / LLM view?"""
    field_types = field_types or {}
    raw = read_html(snapshot)
    page = pruned_page(snapshot, keep_tags)
    base = snapshot.url or None
    raw_text = searchable_text(parse_html(raw), base)
    clean_text = searchable_text(parse_html(page.clean_html), base)
    view_text = searchable_text(parse_html(page.llm_view), base)
    presence = {}
    for gt in snapshot.ground_truth.all():
        ftype = field_types.get(gt.field_name, "string")
        values = gt.values
        if not values:
            presence[gt.field_name] = {"labelled_absent": True}
            continue
        # For lists the LLM view intentionally shows only the first few items
        # of a long run, so "visible" means at least one item is shown.
        view_check = any if gt.cardinality == "many" else all
        need = any if gt.alternatives else all   # alternatives: one form is enough
        if gt.alternatives:
            view_check = any
        presence[gt.field_name] = {
            "in_raw": need(value_in_text(v, raw_text, ftype) for v in values),
            "in_clean_dom": need(value_in_text(v, clean_text, ftype) for v in values),
            "in_llm_view": view_check(value_in_text(v, view_text, ftype) for v in values),
        }
    snapshot.field_presence = presence
    snapshot.save(update_fields=["field_presence"])
    return presence
