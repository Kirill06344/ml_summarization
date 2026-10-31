"""Экстрактивные бейзлайны: нижняя граница для обучаемых моделей и верхняя оценка любого выбора одного предложения.

Резюме XL-Sum — одно предложение, поэтому все методы отдают ровно одно предложение статьи.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .metrics import rouge_tokens, tokenize

# Конец предложения: латинская и арабская пунктуация, данда, иероглифические точки. Абзац — тоже граница.
_SENTENCE = re.compile(r'(?<=[.!?؟।。！？…])["»”’)\]]*\s+|(?<=[。！？])|\n+')
MAX_SENTENCES = 40


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split(text) if part and part.strip()]


def lead(text: str) -> str:
    sentences = split_sentences(text)
    return sentences[0] if sentences else ''


def oracle(text: str, reference: str) -> str:
    """Лучшее предложение по ROUGE-2 с подсмотром ответа: потолок для методов, которые выбирают одно предложение."""
    target = tokenize(reference)
    best, best_score = '', -1.0
    for sentence in split_sentences(text):
        score = rouge_tokens(tokenize(sentence), target)['rouge2']
        if score > best_score:
            best, best_score = sentence, score
    return best


def _similarity(a: Counter, b: Counter) -> float:
    """Сходство TextRank (Mihalcea, Tarau, 2004): общие токены, нормированные логарифмами длин."""
    size_a, size_b = sum(a.values()), sum(b.values())
    if size_a < 2 or size_b < 2:
        return 0.0
    return sum((a & b).values()) / (math.log(size_a) + math.log(size_b))


def textrank(text: str, damping: float = 0.85, iterations: int = 30) -> str:
    sentences = split_sentences(text)[:MAX_SENTENCES]
    if len(sentences) < 2:
        return sentences[0] if sentences else ''
    bags = [Counter(tokenize(sentence)) for sentence in sentences]
    size = len(bags)
    weights = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            weights[i, j] = weights[j, i] = _similarity(bags[i], bags[j])
    totals = weights.sum(axis=1, keepdims=True)
    transition = np.divide(weights, totals, out=np.full_like(weights, 1 / size), where=totals > 0)
    ranks = np.full(size, 1 / size)
    for _ in range(iterations):
        ranks = (1 - damping) / size + damping * transition.T @ ranks
    return sentences[int(np.argmax(ranks))]
