"""Собрать отчёт из scores_*.json, журналов и сохранённых примеров, без запуска моделей.

    .venv/bin/python tools/collect_results.py --final xlsum-reference

Метрики не пересчитываются. Test допускается только для выбранной по val модели,
эталона и бейзлайнов. Числовые блоки README обновляются между маркерами results:.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.languages import LANGUAGES, NAMES  # noqa: E402
from src.metrics import METRICS  # noqa: E402

MODELS = ('xlsum-reference', 'xlsum-mt5-small', 'xlsum-mt5-base', 'xlsum-mt5-xlsum-finetune')
BASELINES = ('baseline-lead', 'baseline-textrank', 'baseline-oracle')
RUNS = MODELS + BASELINES
LABELS = {
    'xlsum-reference': 'XL-Sum без дообучения',
    'xlsum-mt5-small': 'mT5-small',
    'xlsum-mt5-base': 'mT5-base',
    'xlsum-mt5-xlsum-finetune': 'XL-Sum + дообучение',
    'baseline-lead': 'Первое предложение',
    'baseline-textrank': 'TextRank',
    'baseline-oracle': 'Oracle (с доступом к ответу)',
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def load_scores(root: Path, run: str, split: str, expected_n: int) -> dict:
    report = read_json(root / run / f'scores_{split}.json')
    if set(report['per_language']) != set(NAMES):
        raise ValueError(f'{run}/{split}: требуется ровно 9 языков')
    for language, row in report['per_language'].items():
        if row['n'] != expected_n:
            raise ValueError(f'{run}/{split}/{language}: ожидалось {expected_n} документов')
        for metric in METRICS:
            if not math.isfinite(row[metric]) or not 0 <= row[metric] <= 1:
                raise ValueError(f'{run}/{split}: некорректная метрика')
    for metric in METRICS:
        macro = sum(row[metric] for row in report['per_language'].values()) / len(NAMES)
        if not math.isclose(macro, report['macro'][metric], abs_tol=1e-12):
            raise ValueError(f'{run}/{split}: macro не соответствует языковым оценкам')
    low, high = report['rouge2_ci95']
    if not 0 <= low <= high <= 1:
        raise ValueError(f'{run}/{split}: некорректный доверительный интервал')
    return report


def choose_final(scores: dict[str, dict], requested: str) -> str:
    winner = max(MODELS, key=lambda run: scores[run]['macro']['rouge2'])
    if requested != winner:
        raise ValueError(f'По val выбран {winner}, а не {requested}; test не используется для выбора')
    return winner


def number(value: float, *, percent: bool = True) -> str:
    return f'{value * (100 if percent else 1):.2f}'.replace('.', ',')


def table(headers: list[str], rows: list[list[str]]) -> str:
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |',
                      *['| ' + ' | '.join(row) + ' |' for row in rows]]) + '\n'


def score_table(scores: dict[str, dict]) -> str:
    rows = []
    for run, report in scores.items():
        low, high = report['rouge2_ci95']
        rows.append([LABELS[run], *[number(report['macro'][m]) for m in METRICS],
                     f'[{number(low)}; {number(high)}]', str(sum(x['n'] for x in report['per_language'].values()))])
    return table(['Модель / метод', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', '95% CI ROUGE-2', 'Документов'], rows)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding='utf-8')


def plots(out: Path, scores: dict[str, dict], logs: dict[str, dict]) -> None:
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/xlsum-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = sorted(scores, key=lambda run: scores[run]['macro']['rouge2'])
    values = [scores[r]['macro']['rouge2'] * 100 for r in ordered]
    errors = np.array([[v - scores[r]['rouge2_ci95'][0] * 100 for r, v in zip(ordered, values)],
                       [scores[r]['rouge2_ci95'][1] * 100 - v for r, v in zip(ordered, values)]])
    ax.barh([LABELS[r] for r in ordered], values, xerr=errors,
            color=['#94a3b8' if r in BASELINES else '#2563eb' for r in ordered], capsize=3)
    for i, value in enumerate(values):
        ax.text(value + 0.6, i, f'{value:.2f}', va='center')
    ax.set(xlabel='Macro ROUGE-2 F1 × 100; интервалы 95%', title='Сравнение на val: 500 документов × 9 языков', xlim=(0, max(values) + 3))
    fig.tight_layout()
    fig.savefig(out / 'val_rouge2.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for run, log in logs.items():
        history = [h for h in log['history'] if 'val_rouge2' in h]
        if not history:
            continue
        epochs = list(range(1, len(history) + 1))
        axes[0].plot(epochs, [h['train_loss'] for h in history], marker='o', label=LABELS[run])
        axes[1].plot(epochs, [h['val_rouge2'] * 100 for h in history], marker='o', label=LABELS[run])
    axes[0].set(ylabel='Средний train loss', title='Обучение')
    axes[1].set(ylabel='Macro ROUGE-2 × 100', title='Val в журнале обучения')
    for ax in axes:
        ax.set(xlabel='Проверка в конце эпохи', xticks=[1, 2])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'training.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    values = np.array([[scores[r]['per_language'][lang]['rouge2'] * 100 for r in MODELS] for lang in NAMES])
    heatmap = ax.imshow(values, cmap='Blues', aspect='auto', vmin=0, vmax=35)
    ax.set(xticks=range(len(MODELS)), xticklabels=[LABELS[r] for r in MODELS],
           yticks=range(len(NAMES)), yticklabels=[LANGUAGES[x].title for x in NAMES],
           title='ROUGE-2 на val по языкам; сравнения внутри строки')
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    for i in range(len(NAMES)):
        for j in range(len(MODELS)):
            ax.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center',
                    color='white' if values[i, j] > 22 else '#172554')
    fig.colorbar(heatmap, ax=ax, label='ROUGE-2 F1 × 100')
    fig.tight_layout()
    fig.savefig(out / 'languages.png', dpi=180)
    plt.close(fig)


def collect_examples(root: Path, out: Path) -> None:
    # Правило фиксировано и видно читателю: первые три id русского val,
    # независимо от баллов моделей. Это иллюстрации, а не оценка частоты ошибок.
    rows = {}
    for run in MODELS:
        path = root / run / 'predictions' / 'val' / 'russian.jsonl'
        part = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        by_id = {row['id']: row for row in part}
        if len(part) != 500 or len(by_id) != 500:
            raise ValueError(f'{path}: ожидалось 500 уникальных id')
        rows[run] = by_id
    reference = rows[MODELS[0]]
    for run in MODELS[1:]:
        if {k: v['reference'] for k, v in rows[run].items()} != {k: v['reference'] for k, v in reference.items()}:
            raise ValueError(f'{run}: примеры не сопоставимы по id и эталону')
    examples = []
    blocks = ['# Примеры русскоязычных резюме\n',
              'Первые три документа русского val по лексикографическому порядку id. '
              'Примеры не отбирались по ROUGE и не измеряют распространённость ошибок.\n']
    for key in sorted(reference)[:3]:
        predictions = {r: rows[r][key]['prediction'] for r in MODELS}
        examples.append({'language': 'russian', 'split': 'val', 'id': key,
                         'reference': reference[key]['reference'], 'predictions': predictions})
        blocks += [f'## `{key}`\n', '**Эталон:** ' + reference[key]['reference'] + '\n']
        blocks += [f'**{LABELS[r]}:** {prediction}\n' for r, prediction in predictions.items()]
    write_text(out / 'examples.md', '\n'.join(blocks))
    write_text(out / 'examples.json', json.dumps(examples, ensure_ascii=False, indent=2) + '\n')


def update_readme(path: Path, blocks: dict[str, str]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding='utf-8')
    for name, block in blocks.items():
        start, end = f'<!-- results:{name}:start -->', f'<!-- results:{name}:end -->'
        pattern = re.escape(start) + r'.*?' + re.escape(end)
        text, count = re.subn(pattern, lambda _: f'{start}\n\n{block.rstrip()}\n\n{end}', text, flags=re.S)
        if count != 1:
            raise ValueError(f'README: требуется ровно один блок {name}')
    write_text(path, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final', required=True, choices=MODELS)
    args = parser.parse_args()
    root, out = ROOT / 'runs', ROOT / 'docs' / 'results'
    scores = {r: load_scores(root, r, 'val', 500) for r in RUNS}
    final = choose_final(scores, args.final)
    test_runs = tuple(dict.fromkeys(('xlsum-reference', final, *BASELINES)))
    tests = {r: load_scores(root, r, 'test', 2000) for r in test_runs}
    logs = {r: read_json(root / r / 'training_log.json') for r in MODELS}
    manifest = read_json(ROOT / 'data' / 'processed' / 'manifest.json')
    if any(log['data_manifest'] != manifest['revision'] for log in logs.values()):
        raise ValueError('Прогоны используют разные ревизии данных')
    out.mkdir(parents=True, exist_ok=True)
    blocks = {'val': score_table(scores), 'test': score_table(tests)}
    blocks['languages'] = table(['Язык', *[LABELS[r] for r in MODELS]],
        [[LANGUAGES[lang].title, *[number(scores[r]['per_language'][lang]['rouge2']) for r in MODELS]] for lang in NAMES])
    comparisons = []
    for run in MODELS[1:]:
        delta = scores[run]['delta_rouge2']
        if delta['against'] != 'xlsum-reference':
            raise ValueError(f'{run}: требуется scores_val.json с --against xlsum-reference')
        comparisons.append([LABELS[run], number(delta['delta']),
                            f'[{number(delta["ci95"][0])}; {number(delta["ci95"][1])}]'])
    blocks['comparisons'] = table(['Сравнение с исходным XL-Sum', 'Δ ROUGE-2, п.', 'Парный 95% CI, п.'], comparisons)
    training = []
    for run, log in logs.items():
        if log['args']['eval_only']:
            training.append([LABELS[run], '—', '—', '—', '—', number(log['total_minutes'], percent=False)])
            continue
        last = log['history'][-1]
        if last['final_step'] != last['planned_steps']:
            raise ValueError(f'{run}: обучение завершено не полностью')
        training.append([LABELS[run], f'{last["final_step"]}/{last["planned_steps"]}',
                         str(last['epochs_started']), str(last['skipped_batches']), str(last['overflowed_steps']),
                         number(log['total_minutes'], percent=False)])
    blocks['training'] = table(['Модель', 'Шаги', 'Эпохи', 'Пропущенные батчи', 'Переполнения fp16', 'Всего, мин'], training)
    blocks['data'] = table(['Язык', 'Train после очистки', 'Val (полный)', 'Test (полный)'],
        [[LANGUAGES[lang].title, *[str(manifest['languages'][lang]['kept'][s]) for s in ('train', 'val', 'test')]] for lang in NAMES])
    test_id = tests['xlsum-reference']['per_language']['indonesian']['rouge2']
    test_display = f'{test_id * 100:.4f}'.replace('.', ',')
    difference = f'{(test_id - 0.170181) * 100:+.4f}'.replace('.', ',')
    blocks['sanity'] = (f'Индонезийский ROUGE-2 исходного XL-Sum на test: **{test_display}**. '
                       f'В таблице авторов — 17,0181; разность **{difference} п.** '
                       'Сверка проводится на подвыборке test, поэтому не является точным воспроизведением полного теста авторов.')
    for name, block in blocks.items():
        write_text(out / f'{name}.md', block)
    with (out / 'scores.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['split', 'run', 'language', 'n', *METRICS])
        for split, reports in [('val', scores), ('test', tests)]:
            for run, report in reports.items():
                for lang in NAMES:
                    row = report['per_language'][lang]
                    writer.writerow([split, run, lang, row['n'], *[row[m] for m in METRICS]])
                writer.writerow([split, run, 'macro', sum(r['n'] for r in report['per_language'].values()),
                                 *[report['macro'][m] for m in METRICS]])
    inputs = [root / r / 'scores_val.json' for r in RUNS] + [root / r / 'scores_test.json' for r in test_runs]
    inputs += [root / r / 'training_log.json' for r in MODELS]
    inputs += [root / r / 'predictions/val/russian.jsonl' for r in MODELS]
    inputs += [ROOT / 'data/processed/manifest.json']
    provenance = {'final': final, 'selection_split': 'val', 'primary_metric': 'macro.rouge2',
                  'val': {r: scores[r]['macro']['rouge2'] for r in MODELS}, 'test_runs': test_runs,
                  'data_revision': manifest['revision'],
                  'inputs_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    write_text(out / 'provenance.json', json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    collect_examples(root, out)
    plots(out, scores, logs)
    update_readme(ROOT / 'README.md', blocks)
    print(f'Готово: {out.relative_to(ROOT)}, финальная модель {final}')


if __name__ == '__main__':
    main()
