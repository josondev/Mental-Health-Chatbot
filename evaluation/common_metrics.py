import math
import re
import string
from collections import Counter
from typing import Any, Iterable, Sequence


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(prediction: Any, gold: Any) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))


def best_exact_match(prediction: Any, gold_answers: Sequence[Any]) -> float:
    return max((exact_match(prediction, gold) for gold in gold_answers), default=0.0)


def token_f1(prediction: Any, gold: Any) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_token_f1(prediction: Any, gold_answers: Sequence[Any]) -> float:
    return max((token_f1(prediction, gold) for gold in gold_answers), default=0.0)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def canonical_rows(rows: Iterable[Iterable[Any]]) -> list[tuple[str, ...]]:
    normalized = [tuple(normalize_text(cell) for cell in row) for row in rows]
    return sorted(normalized)


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1
