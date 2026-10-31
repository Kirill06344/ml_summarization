"""Резюме текста обученной или опубликованной моделью.

    .venv/bin/python tools/summarize.py --model runs/xlsum-mt5-base/model --file article.txt
    echo "Текст статьи" | .venv/bin/python tools/summarize.py --model csebuetnlp/mT5_multilingual_XLSum
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.modeling import generate, load_model, load_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='папка модели или имя на Hugging Face')
    parser.add_argument('--file', type=Path, default=None, help='по умолчанию текст читается из stdin')
    parser.add_argument('--beams', type=int, default=4)
    args = parser.parse_args()
    text = args.file.read_text(encoding='utf-8') if args.file else sys.stdin.read()
    if not text.strip():
        raise SystemExit('пустой текст')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model).to(device)
    print(generate(model, tokenizer, [text], batch_size=1, beams=args.beams)[0])


if __name__ == '__main__':
    main()
