"""Оценить прогнозы прогонов и сравнить их парно: единственный путь от прогнозов к числам отчёта.

    .venv/bin/python tools/score.py --split val baseline-lead baseline-textrank xlsum-mt5-small
    .venv/bin/python tools/score.py --split val --against baseline-lead baseline-textrank

Результат: stdout и runs/<прогон>/scores_<сплит>.json. Test запускать только для моделей, выбранных по val.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import METRICS, bootstrap_macro, score_pairs, summarize  # noqa: E402
from src.predict import read_predictions  # noqa: E402


def load(run: str, split: str) -> dict:
    rows = read_predictions(ROOT / 'runs' / run / 'predictions' / split)
    return {language: score_pairs((r['prediction'] for r in part), (r['reference'] for r in part))
            for language, part in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--against', default=None, help='парная разность ROUGE-2 с этим прогоном')
    args = parser.parse_args()
    reference = load(args.against, args.split) if args.against else None
    for run in args.runs:
        scores = load(run, args.split)
        report = summarize(scores).as_dict()
        point, low, high = bootstrap_macro(scores)
        report['rouge2_ci95'] = [low, high]
        line = ' '.join(f'{m} {report["macro"][m] * 100:6.2f}' for m in METRICS)
        print(f'{run:32} {line}  R2 CI [{low * 100:.2f}, {high * 100:.2f}]  документов {sum(map(len, scores.values()))}')
        if reference is not None:
            if set(scores) != set(reference):
                raise SystemExit(f'{run}: другой набор языков, чем у {args.against}')
            delta, low, high = bootstrap_macro(scores, baseline=reference)
            report['delta_rouge2'] = {'against': args.against, 'delta': delta, 'ci95': [low, high]}
            print(f'{"":32} ΔR2 к {args.against}: {delta * 100:+.2f} [{low * 100:+.2f}, {high * 100:+.2f}]')
        (ROOT / 'runs' / run / f'scores_{args.split}.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
