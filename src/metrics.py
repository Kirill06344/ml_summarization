"""ROUGE-1/2/L без внешних зависимостей: едет в ядро Kaggle вместе с пакетом.

Стандартный `rouge_score` оставляет только `[a-z0-9]`, и на кириллице, деванагари или арабском письме любая пара
текстов получает ноль. Здесь токен — непрерывная последовательность букв, знаков-модификаторов и цифр любого
алфавита, а иероглифы идут по одному: без этого весь китайский абзац был бы одним токеном.

Стемминга нет, поэтому числа не равны числам статьи XL-Sum там, где у её скорера есть стеммер (docs/metrics.md).
Определение метрики совпадает с `rouge_score`: F1 по клипованным n-граммам, для L — по длине LCS всего текста.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

METRICS = ('rouge1', 'rouge2', 'rougeL')
PRIMARY = 'rouge2'  # главная метрика статьи XL-Sum и правила выбора чекпойнта


def _is_han(ch: str) -> bool:
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2FFFF
            or 0xF900 <= code <= 0xFAFF)


def tokenize(text: str) -> list[str]:
    tokens, word = [], []
    for ch in unicodedata.normalize('NFKC', text).lower():
        if _is_han(ch):
            if word:
                tokens.append(''.join(word))
                word = []
            tokens.append(ch)
        elif unicodedata.category(ch)[0] in 'LMN':
            word.append(ch)
        elif word:
            tokens.append(''.join(word))
            word = []
    if word:
        tokens.append(''.join(word))
    return tokens


def _f1(overlap: float, predicted: int, reference: int) -> float:
    if not overlap or not predicted or not reference:
        return 0.0
    precision, recall = overlap / predicted, overlap / reference
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def rouge_tokens(predicted: Sequence[str], reference: Sequence[str]) -> dict[str, float]:
    scores = {}
    for name, n in (('rouge1', 1), ('rouge2', 2)):
        p, r = _ngrams(predicted, n), _ngrams(reference, n)
        scores[name] = _f1(sum((p & r).values()), sum(p.values()), sum(r.values()))
    scores['rougeL'] = _f1(_lcs_length(predicted, reference), len(predicted), len(reference))
    return scores


def rouge(predicted: str, reference: str) -> dict[str, float]:
    return rouge_tokens(tokenize(predicted), tokenize(reference))


def score_pairs(predictions: Iterable[str], references: Iterable[str]) -> np.ndarray:
    """Матрица (документы, 3) в порядке METRICS, в долях единицы."""
    rows = [[scores[m] for m in METRICS] for scores in map(rouge, predictions, references)]
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(METRICS))


@dataclass
class Report:
    """Средние по языкам и их среднее (macro). Macro — главная сводка: языки равноправны независимо от размера."""
    per_language: dict[str, dict[str, float]]
    macro: dict[str, float]

    def as_dict(self) -> dict:
        return {'per_language': self.per_language, 'macro': self.macro}


def summarize(per_document: dict[str, np.ndarray]) -> Report:
    per_language = {language: dict(zip(METRICS, map(float, scores.mean(axis=0)))) | {'n': int(len(scores))}
                    for language, scores in per_document.items()}
    macro = {m: float(np.mean([row[m] for row in per_language.values()])) for m in METRICS}
    return Report(per_language, macro)


def bootstrap_macro(per_document: dict[str, np.ndarray], *, metric: str = PRIMARY, iterations: int = 2000,
                    seed: int = 0, baseline: dict[str, np.ndarray] | None = None) -> tuple[float, float, float]:
    """Macro-среднее и 95% CI по документам, с ресэмплингом внутри каждого языка.

    С `baseline` считается парная разность с тем же порядком документов. CI покрывает только случайность выборки
    документов, но не случайность обучения.
    """
    column = METRICS.index(metric)
    rng = np.random.default_rng(seed)
    draws = np.zeros(iterations)
    point = []
    for language, scores in per_document.items():
        values = scores[:, column]
        if baseline is not None:
            if len(baseline[language]) != len(values):
                raise ValueError(f'{language}: разный набор документов, парная разность бессмысленна')
            values = values - baseline[language][:, column]
        point.append(values.mean())
        picks = rng.integers(0, len(values), size=(iterations, len(values)))
        draws += values[picks].mean(axis=1)
    draws /= len(per_document)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(np.mean(point)), float(low), float(high)
