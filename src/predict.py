"""Прогнозы модели на сплите: predictions/<split>/<язык>.jsonl (id, резюме, эталон; текст статьи не хранится)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .data import load_split
from .distributed import Context
from .modeling import generate


def shared_generate(context: Context, model, tokenizer, texts: list[str], args, amp_dtype) -> list[str]:
    """Сгенерировать вместе: процессы берут документы по очереди из списка по длине, поэтому нагрузка у них равная."""
    order = sorted(range(len(texts)), key=lambda i: (-len(texts[i]), i))[context.rank::context.world]
    summaries = generate(model, tokenizer, [texts[i] for i in order], batch_size=args.eval_batch,
                         max_source=args.max_source, max_target=args.max_target, beams=args.beams,
                         length_penalty=args.length_penalty, no_repeat_ngram=args.no_repeat_ngram, amp_dtype=amp_dtype)
    merged = [''] * len(texts)
    for indices, part in context.gather((order, summaries)):
        for i, summary in zip(indices, part):
            merged[i] = summary
    return merged


def write_predictions(context: Context, model, tokenizer, folder: Path, languages: list[str], split: str,
                      limit: int | None, out: Path, amp_dtype, args) -> None:
    target = out / 'predictions' / split
    target.mkdir(parents=True, exist_ok=True)
    for language in languages:
        rows = load_split(folder, language, split, limit)
        started = time.monotonic()
        predictions = shared_generate(context, model, tokenizer, [row['text'] for row in rows], args, amp_dtype)
        if context.is_main:
            with open(target / f'{language}.jsonl', 'w', encoding='utf-8') as file:
                for row, prediction in zip(rows, predictions):
                    file.write(json.dumps({'id': row['id'], 'prediction': prediction, 'reference': row['summary']},
                                          ensure_ascii=False) + '\n')
            print(f'{split}/{language}: {len(rows)} за {(time.monotonic() - started) / 60:.1f} мин', flush=True)


def read_predictions(folder: Path) -> dict[str, list[dict]]:
    """Прогнозы по языкам, отсортированные по id: иначе парные сравнения прогонов сопоставят разные документы."""
    result = {}
    for path in sorted(Path(folder).glob('*.jsonl')):
        rows = [json.loads(line) for line in path.open(encoding='utf-8')]
        result[path.stem] = sorted(rows, key=lambda row: row['id'])
    if not result:
        raise FileNotFoundError(f'нет прогнозов в {folder}')
    return result
