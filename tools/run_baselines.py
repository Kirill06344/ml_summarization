"""Экстрактивные бейзлайны на CPU: прогнозы в формате моделей, runs/baseline-<метод>/predictions/<сплит>/.

    .venv/bin/python tools/run_baselines.py --split val --limit 500
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import lead, oracle, textrank  # noqa: E402
from src.data import load_split  # noqa: E402
from src.languages import NAMES  # noqa: E402

METHODS = ('lead', 'textrank', 'oracle')


def run_row(row: dict) -> dict[str, str]:
    return {'lead': lead(row['text']), 'textrank': textrank(row['text']), 'oracle': oracle(row['text'], row['summary'])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'processed')
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--limit', type=int, default=0, help='документов на язык; 0 — все')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    with Pool(args.workers) as pool:
        for language in NAMES:
            rows = load_split(args.data, language, args.split, args.limit or None)
            results = pool.map(run_row, rows, chunksize=32)
            for method in METHODS:
                folder = ROOT / 'runs' / f'baseline-{method}' / 'predictions' / args.split
                folder.mkdir(parents=True, exist_ok=True)
                with open(folder / f'{language}.jsonl', 'w', encoding='utf-8') as file:
                    for row, result in zip(rows, results):
                        file.write(json.dumps({'id': row['id'], 'prediction': result[method],
                                               'reference': row['summary']}, ensure_ascii=False) + '\n')
            print(f'{language}: {len(rows)}', flush=True)


if __name__ == '__main__':
    main()
