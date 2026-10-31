"""Данные: чтение архивов XL-Sum, очистка train, загрузка и балансировка по языкам.

Разбиение train/val/test — официальное. Val и test не меняются, чтобы результаты оставались сравнимы с публикацией;
из train удаляются пересечения с ними и мусор (подробности и счётчики — docs/data.md, data/processed/manifest.json).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import tarfile
from pathlib import Path
from typing import Iterable, Iterator

SPLITS = ('train', 'val', 'test')
COLUMNS = ('id', 'url', 'title', 'summary', 'text')
MIN_TEXT_CHARS = 100

_SPACES = re.compile(r'[^\S\n]+')
_NEWLINES = re.compile(r'\s*\n\s*')
_ANY_SPACE = re.compile(r'\s+')


def clean_text(text: str) -> str:
    """Схлопнуть пробелы, сохранив абзацы: они нужны экстрактивным бейзлайнам."""
    return _NEWLINES.sub('\n', _SPACES.sub(' ', text or '')).strip()


def model_input(text: str) -> str:
    """Вид текста для модели: одна строка, как в карточке эталонной модели XL-Sum."""
    return _ANY_SPACE.sub(' ', text).strip()


def fingerprint(text: str) -> str:
    return hashlib.md5(_ANY_SPACE.sub(' ', text).strip().lower().encode()).hexdigest()


def read_archive(path: Path) -> dict[str, list[dict]]:
    """Прочитать сплиты из `<язык>_XLSum_v2.0.tar.bz2` без распаковки на диск."""
    splits: dict[str, list[dict]] = {}
    with tarfile.open(path, 'r:bz2') as archive:
        for member in archive:
            match = re.fullmatch(r'(?:\./)?[a-z_]+_(train|val|test)\.jsonl', member.name)
            if not (match and member.isfile()):
                continue
            stream = archive.extractfile(member)
            rows = [json.loads(line) for line in stream]
            splits[match.group(1)] = [{k: clean_text(row[k]) if k in ('title', 'summary', 'text') else row[k]
                                       for k in COLUMNS} for row in rows]
    missing = set(SPLITS) - set(splits)
    if missing:
        raise ValueError(f'{path.name}: нет сплитов {sorted(missing)}')
    return splits


def clean_train(splits: dict[str, list[dict]]) -> tuple[list[dict], dict[str, int]]:
    """Очистить train. Сравнение — по нормализованному тексту: регистр и пробелы не делают статью новой.

    Совпадение резюме учитывается отдельно от текста: под шаблонным резюме могут быть разные статьи, но модель,
    выучившая такое резюме на train, получила бы на val/test незаслуженный балл.
    """
    held_texts = {fingerprint(row['text']) for split in ('val', 'test') for row in splits[split]}
    held_summaries = {fingerprint(row['summary']) for split in ('val', 'test') for row in splits[split]}
    kept, seen = [], set()
    removed = dict.fromkeys(('short_text', 'summary_not_shorter', 'text_in_val_test', 'summary_in_val_test',
                             'duplicate_text'), 0)
    for row in splits['train']:
        text, summary = fingerprint(row['text']), fingerprint(row['summary'])
        if len(row['text']) < MIN_TEXT_CHARS or not row['summary']:
            removed['short_text'] += 1
        elif len(row['summary']) >= len(row['text']):
            removed['summary_not_shorter'] += 1
        elif text in held_texts:
            removed['text_in_val_test'] += 1
        elif summary in held_summaries:
            removed['summary_in_val_test'] += 1
        elif text in seen:
            removed['duplicate_text'] += 1
        else:
            seen.add(text)
            kept.append(row)
    return kept, removed


def stable_subset(rows: list[dict], limit: int | None) -> list[dict]:
    """Подвыборка, не зависящая от порядка строк и сида: один и тот же набор документов во всех прогонах."""
    if limit is None or limit <= 0 or limit >= len(rows):
        return rows
    return sorted(rows, key=lambda row: hashlib.md5(row['id'].encode()).hexdigest())[:limit]


def load_split(folder: Path, language: str, split: str, limit: int | None = None) -> list[dict]:
    import pandas as pd

    path = Path(folder) / f'{language}_{split}.parquet'
    rows = pd.read_parquet(path, columns=['id', 'title', 'summary', 'text']).to_dict('records')
    for row in rows:
        row['language'] = language
    return stable_subset(rows, limit)


def find_data(root: Path) -> Path:
    """Папка с parquet: Kaggle монтирует датасет не всегда по ожидаемому пути."""
    root = Path(root)
    found = sorted(root.rglob('manifest.json'))
    if not found:
        raise FileNotFoundError(f'manifest.json не найден под {root}')
    return found[0].parent


def epoch_indices(pools: dict[str, list[int]], per_language: int, rng: random.Random) -> list[int]:
    """Индексы одной эпохи: поровну от каждого языка. Без этого английский занял бы половину обучения.

    Большой язык каждую эпоху отдаёт новую случайную порцию, поэтому его данные не пропадают.
    """
    indices = []
    for language in sorted(pools):
        pool = pools[language]
        indices += rng.sample(pool, per_language) if per_language < len(pool) else list(pool)
    rng.shuffle(indices)
    return indices


def length_batches(indices: list[int], lengths: list[int], batch_size: int, rng: random.Random,
                   window: int = 50) -> Iterator[list[int]]:
    """Батчи из текстов близкой длины: меньше паддинга и быстрее шаг, порядок батчей остаётся случайным."""
    batches = []
    for start in range(0, len(indices), batch_size * window):
        chunk = sorted(indices[start:start + batch_size * window], key=lengths.__getitem__)
        batches += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    yield from batches


def collate(rows: Iterable[dict], tokenizer, max_source: int, max_target: int) -> dict:
    rows = list(rows)
    # Символов заведомо больше, чем войдёт в max_source токенов: иначе самые длинные статьи тормозят токенизатор.
    texts = [model_input(row['text'])[:max_source * 12] for row in rows]
    batch = tokenizer(texts, max_length=max_source, truncation=True, padding=True, return_tensors='pt')
    targets = tokenizer(text_target=[row['summary'] for row in rows], max_length=max_target, truncation=True,
                        padding=True, return_tensors='pt')
    labels = targets['input_ids']
    labels[labels == tokenizer.pad_token_id] = -100
    batch['labels'] = labels
    return dict(batch)
