"""Скачать архивы XL-Sum зафиксированной ревизии и собрать data/processed: parquet, манифест и счётчики очистки.

    .venv/bin/python tools/prepare_data.py

Загрузочный скрипт датасета (`xlsum.py`) новые версии `datasets` не исполняют, поэтому архивы читаются напрямую. Они
не распаковываются на диск: `src.data.read_archive` читает из tar только ожидаемые jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data import SPLITS, clean_train, read_archive  # noqa: E402
from src.languages import NAMES  # noqa: E402

REVISION = '8d824689b60413430f1782f5bea2a56a4c390b68'
URL = 'https://huggingface.co/datasets/GEM/xlsum/resolve/{revision}/data/{language}_XLSum_v2.0.tar.bz2'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(language: str, folder: Path) -> Path:
    path = folder / f'{language}_XLSum_v2.0.tar.bz2'
    if not path.exists():
        print(f'скачиваю {language}', flush=True)
        partial = path.with_suffix('.part')
        urllib.request.urlretrieve(URL.format(revision=REVISION, language=language), partial)
        partial.rename(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'data' / 'raw')
    parser.add_argument('--out', type=Path, default=ROOT / 'data' / 'processed')
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {'dataset': 'GEM/xlsum', 'version': 'XL-Sum v2.0', 'revision': REVISION, 'languages': {}}
    for language in NAMES:
        archive = download(language, args.raw)
        splits = read_archive(archive)
        original = {split: len(splits[split]) for split in SPLITS}
        splits['train'], removed = clean_train(splits)
        for split in SPLITS:
            pd.DataFrame(splits[split]).to_parquet(args.out / f'{language}_{split}.parquet', compression='zstd',
                                                   index=False)
        manifest['languages'][language] = {
            'archive_sha256': sha256(archive), 'original': original,
            'kept': {split: len(splits[split]) for split in SPLITS}, 'removed_from_train': removed}
        print(language, manifest['languages'][language]['kept'], removed, flush=True)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
