"""
Metric definitions — pure functions over plain dicts, so every number in the
paper can be recomputed from the exported CSVs.

Field-level counting (SWDE-style, one decision per field per run):

  single-value field, ground truth present:
      extracted and exactly equal (after normalisation) -> TP
      extracted but different                           -> FP and FN
      not extracted                                     -> FN
  single-value field labelled <ABSENT> on that page:
      extracted -> FP ; not extracted -> TN (not counted)
  list field: item-level multiset matching -> TP/FP/FN items

Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = harmonic mean (micro).
"""

from __future__ import annotations

import math
import statistics


def safe_div(a, b):
    return (a / b) if b else None


def prf(tp: int, fp: int, fn: int) -> dict:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    if p is None or r is None:
        f = None
    elif p + r == 0:
        f = 0.0
    else:
        f = 2 * p * r / (p + r)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def record_counts(rec: dict):
    """(tp, fp, fn) for one field record, or None if it has no usable label."""
    if not rec.get("has_gt"):
        return None
    if rec["cardinality"] == "many":
        if rec["gt_available"]:
            return rec.get("list_tp") or 0, rec.get("list_fp") or 0, rec.get("list_fn") or 0
        return 0, (rec.get("n_values") or 0) if rec["success"] else 0, 0
    if rec["gt_available"]:
        if rec["success"] and rec.get("exact_match"):
            return 1, 0, 0
        if rec["success"]:
            return 0, 1, 1
        return 0, 0, 1
    return (0, 1, 0) if rec["success"] else (0, 0, 0)


def is_correct(rec: dict, list_threshold: float = 0.9):
    """Is this field's output correct w.r.t. its label? None if unlabelled."""
    if not rec.get("has_gt"):
        return None
    if not rec["gt_available"]:
        return not rec["success"]
    if rec["cardinality"] == "many":
        counts = record_counts(rec)
        f1 = prf(*counts)["f1"]
        return bool(f1 is not None and f1 >= list_threshold)
    return bool(rec["success"] and rec.get("exact_match"))


def aggregate(records: list[dict]) -> dict:
    tp = fp = fn = 0
    labelled = correct = 0
    token_f1 = []
    for rec in records:
        counts = record_counts(rec)
        if counts is None:
            continue
        tp, fp, fn = tp + counts[0], fp + counts[1], fn + counts[2]
        labelled += 1
        if is_correct(rec):
            correct += 1
        if rec.get("gt_available") and rec.get("token_f1") is not None:
            token_f1.append(rec["token_f1"])
    out = prf(tp, fp, fn)
    out.update({
        "fields": labelled,
        "field_accuracy": safe_div(correct, labelled),
        "token_f1_mean": statistics.fmean(token_f1) if token_f1 else None,
    })
    return out


def describe(values: list) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "median": None, "p95": None,
                "min": None, "max": None}
    ordered = sorted(vals)
    p95 = ordered[min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))]
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "median": statistics.median(vals),
        "p95": p95,
        "min": ordered[0],
        "max": ordered[-1],
    }


def binary_eval(pred: list[bool], label: list[bool]) -> dict:
    tp = sum(1 for p, y in zip(pred, label) if p and y)
    fp = sum(1 for p, y in zip(pred, label) if p and not y)
    fn = sum(1 for p, y in zip(pred, label) if not p and y)
    tn = sum(1 for p, y in zip(pred, label) if not p and not y)
    out = prf(tp, fp, fn)
    out.update({"tn": tn, "n": len(label), "fpr": safe_div(fp, fp + tn),
                "false_alarm_share": safe_div(fp, len(label))})
    return out


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list, y: list):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    rx, ry = _ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs])
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy)


def sign_test(wins: int, losses: int):
    """Two-sided exact binomial (sign) test p-value; ties excluded."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def wilcoxon(diffs: list[float]):
    nonzero = [d for d in diffs if d]
    if len(nonzero) < 5:
        return None
    try:
        from scipy.stats import wilcoxon as _wilcoxon

        return float(_wilcoxon(nonzero).pvalue)
    except Exception:
        return None
