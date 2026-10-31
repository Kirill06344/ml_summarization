"""Залить папку на Kaggle приватным датасетом и проверить приватность запросом к серверу.

    .venv/bin/python tools/kaggle_upload.py --dir data/processed --slug xlsum-nine-languages

Лицензия XL-Sum (CC BY-NC-SA 4.0) допускает учебное использование; приватность здесь — чтобы случайно не опубликовать
производную копию данных от своего имени. Аккаунт — `~/.kaggle/access_token`, см. docs/pitfalls.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = str(ROOT / '.venv' / 'bin' / 'kaggle')


def run(*args: str, check: bool = True) -> str:
    # Глобальный KAGGLE_API_TOKEN молча увёл бы операцию на другой аккаунт.
    env = {k: v for k, v in os.environ.items() if k != 'KAGGLE_API_TOKEN'}
    result = subprocess.run([KAGGLE, *args], capture_output=True, text=True, env=env)
    output = (result.stdout or '') + (result.stderr or '')
    if check and result.returncode != 0:
        raise SystemExit(f'kaggle {" ".join(args)} упал:\n{output}')
    return output


def verify_private(dataset: str) -> None:
    """`datasets list --mine` индексируется с задержкой, поэтому источник правды — `datasets metadata`.

    Сразу после заливки metadata отвечает ошибкой: это «пока не готов», а не «не приватный». Поэтому повторяем.
    """
    for attempt in range(40):
        with tempfile.TemporaryDirectory() as tmp:
            result = run('datasets', 'metadata', '-p', tmp, dataset, check=False)
            path = Path(tmp) / 'dataset-metadata.json'
            if path.exists():
                info = json.loads(path.read_text())
                info = info.get('info', info)
                if info.get('isPrivate') is True:
                    print('приватность подтверждена сервером')
                    return
                raise SystemExit(f'!!! {dataset}: isPrivate={info.get("isPrivate")!r}, проверить на сайте и удалить')
        print(f'metadata ещё не отвечает ({attempt + 1}/40): {result.strip()[:120]}', flush=True)
        time.sleep(30)
    raise SystemExit(f'!!! не удалось проверить приватность {dataset}: проверить вручную')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, required=True)
    parser.add_argument('--slug', required=True)
    parser.add_argument('--user', default='badzhik')
    parser.add_argument('--message', default='', help='сообщение новой версии; с ним обновляется существующий датасет')
    args = parser.parse_args()
    dataset = f'{args.user}/{args.slug}'
    (args.dir / 'dataset-metadata.json').write_text(json.dumps({
        'title': args.slug.replace('-', ' '), 'id': dataset, 'licenses': [{'name': 'other'}]}, indent=2))
    size = sum(f.stat().st_size for f in args.dir.iterdir()) / 2**20
    print(f'{dataset}: {size:.0f} МБ', flush=True)
    started = time.monotonic()
    # Без --public: датасет создаётся приватным. CLI отвечает кодом 0 и на ошибку создания, поэтому текст проверяется.
    output = (run('datasets', 'version', '-p', str(args.dir), '-m', args.message, '-q') if args.message
              else run('datasets', 'create', '-p', str(args.dir), '-q'))
    print(output)
    if 'error' in output.lower():
        raise SystemExit(f'{dataset}: заливка не удалась')
    print(f'заливка: {(time.monotonic() - started) / 60:.1f} мин')
    verify_private(dataset)


if __name__ == '__main__':
    main()
