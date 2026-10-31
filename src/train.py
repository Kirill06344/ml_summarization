"""Дообучение mT5 на нескольких языках. Точка входа ядра Kaggle и локального дыма.

Цикл свой, а не `Trainer`: в образе Kaggle версия transformers меняется без предупреждения, а здесь от библиотеки нужны
только модель и генерация. На каждую GPU — свой процесс (`src/distributed.py`). Правила выбора чекпойнта и сравнения —
docs/experiment_protocol.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# До импорта torch: длинные входы разной длины иначе фрагментируют память T4 и дают ложный OOM.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch

from .codehash import package_hash
from .data import collate, epoch_indices, find_data, length_batches, load_split
from .distributed import Context, setup, teardown
from .languages import NAMES
from .metrics import METRICS, PRIMARY, score_pairs, summarize
from .modeling import SummarizerLoss, keep_feed_forward_output_in_fp32, load_model, load_tokenizer, tiny_model
from .predict import shared_generate, write_predictions

TINY = 'tiny-random'
AMP = {'none': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='папка с parquet и manifest.json (или её родитель)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', default='google/mt5-small')
    parser.add_argument('--revision', default=None, help='коммит весов на Hugging Face')
    parser.add_argument('--tokenizer', default=None, help='по умолчанию из --model')
    parser.add_argument('--languages', default=','.join(NAMES))
    parser.add_argument('--train-per-language', type=int, default=20_000, help='примеров языка за эпоху')
    parser.add_argument('--val-per-language', type=int, default=500)
    parser.add_argument('--test-per-language', type=int, default=0, help='0 — без прогнозов на test; -1 — весь test')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=0, help='шагов оптимизатора; 0 — по эпохам')
    parser.add_argument('--batch', type=int, default=8, help='суммарно на все процессы')
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--max-source', type=int, default=512)
    parser.add_argument('--max-target', type=int, default=84)
    parser.add_argument('--beams', type=int, default=4)
    parser.add_argument('--length-penalty', type=float, default=0.6)
    parser.add_argument('--no-repeat-ngram', type=int, default=2)
    parser.add_argument('--eval-batch', type=int, default=32, help='на процесс')
    parser.add_argument('--eval-every', type=int, default=0, help='шагов оптимизатора; 0 — раз в эпоху')
    parser.add_argument('--amp', choices=sorted(AMP), default='none')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--time-limit-hours', type=float, default=10.0,
                        help='остановить обучение заранее: у ядра Kaggle лимит 12 ч, ещё нужно время на прогнозы')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--processes', type=int, default=0, help='0 — по числу GPU; на CPU для тестов')
    parser.add_argument('--eval-only', action='store_true', help='без обучения: оценить --model как есть')
    parser.add_argument('--require-gpu', action='store_true', help='упасть без GPU, а не обучаться на CPU')
    parser.add_argument('--expect-code-hash', default=None, help='упасть, если на Kaggle приехал другой код')
    return parser


def log(context: Context, message: str) -> None:
    if context.is_main:
        print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def environment() -> dict:
    import transformers

    return {'python': sys.version.split()[0], 'torch': torch.__version__, 'transformers': transformers.__version__,
            'cuda': torch.version.cuda, 'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Линейный разогрев и линейное затухание, как в обучении авторов XL-Sum на одном GPU."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    return max(0.0, (total - step) / max(total - warmup, 1))


def parameter_groups(model, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim < 2 or 'norm' in name.lower() else decay).append(parameter)
    return [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]


class Validator:
    """Фиксированная подвыборка val: один и тот же набор документов на всех проверках всех прогонов."""

    def __init__(self, folder: Path, languages: list[str], per_language: int, args, amp_dtype, context: Context):
        self.sets = {language: load_split(folder, language, 'val', per_language) for language in languages}
        self.args, self.amp_dtype, self.context = args, amp_dtype, context

    def __call__(self, model, tokenizer) -> dict | None:
        scores = {}
        for language, rows in self.sets.items():
            predictions = shared_generate(self.context, model, tokenizer, [row['text'] for row in rows], self.args,
                                          self.amp_dtype)
            scores[language] = score_pairs(predictions, [row['summary'] for row in rows])
        return summarize(scores).as_dict() if self.context.is_main else None


def train(args, model, tokenizer, folder: Path, languages: list[str], validate: Validator, out: Path,
          context: Context, amp_dtype, history: list[dict], save_log) -> None:
    rows, pools = [], {}
    for language in languages:
        loaded = load_split(folder, language, 'train')
        pools[language] = list(range(len(rows), len(rows) + len(loaded)))
        rows += loaded
        log(context, f'train {language}: {len(loaded)}')
    lengths = [len(row['text']) for row in rows]
    per_epoch = sum(min(args.train_per_language, len(pool)) for pool in pools.values())
    steps_per_epoch = max(per_epoch // (args.batch * args.accumulation), 1)
    total = args.max_steps or steps_per_epoch * args.epochs
    warmup = int(total * args.warmup_ratio)
    eval_every = args.eval_every or steps_per_epoch
    log(context, f'{per_epoch} примеров за эпоху, {steps_per_epoch} шагов на эпоху, всего {total}, разогрев {warmup}, '
                 f'{context.world} процесс(а)')

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW(parameter_groups(model, args.weight_decay), lr=args.lr, foreach=True)
    criterion = context.wrap(SummarizerLoss(model, args.label_smoothing, amp_dtype))
    # GradScaler нужен только fp16; при bf16 и none он выключен и ничего не меняет.
    # Стартовый масштаб ниже стандартного 65536: на нём градиенты mT5 переполняются и первые шаги пропускаются.
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.0,
                                  enabled=amp_dtype is torch.float16 and context.device.type == 'cuda')
    rng = random.Random(args.seed)  # один сид у всех процессов: они должны видеть один порядок батчей
    best, step, micro, started = -math.inf, 0, 0, time.monotonic()
    running, seen, skipped, overflowed = 0.0, 0, 0, 0
    stop = False

    def evaluate_and_save(reason: str) -> None:
        nonlocal best
        started_eval = time.monotonic()
        optimizer.zero_grad(set_to_none=True)  # градиенты 2 ГБ не нужны генерации
        torch.cuda.empty_cache()
        report = validate(model, tokenizer)
        if context.is_main:
            score = report['macro'][PRIMARY]
            record = {'step': step, 'reason': reason, 'train_loss': running / max(seen, 1), 'skipped_batches': skipped,
                      'overflowed_steps': overflowed, 'grad_scale': scaler.get_scale(),
                      'minutes': (time.monotonic() - started) / 60,
                      'eval_minutes': (time.monotonic() - started_eval) / 60,
                      **{f'val_{m}': report['macro'][m] for m in METRICS}, 'val_per_language': report['per_language']}
            history.append(record)
            improved = score > best
            log(context, f'шаг {step}/{total} loss {record["train_loss"]:.4f} val ' +
                ' '.join(f'{m} {report["macro"][m] * 100:.2f}' for m in METRICS) + (' — лучший' if improved else ''))
            if improved:
                best = score
                model.save_pretrained(out / 'model')
                tokenizer.save_pretrained(out / 'model')
            save_log()
        context.barrier()
        torch.cuda.empty_cache()
        criterion.train()

    criterion.train()
    epoch = 0
    while not stop:
        epoch += 1
        indices = epoch_indices(pools, args.train_per_language, rng)
        for batch_indices in length_batches(indices, lengths, args.batch, rng):
            if len(batch_indices) < context.world:
                continue  # хвост эпохи: одному из процессов не досталось бы примеров, и обучение зависло бы
            part = batch_indices[context.rank::context.world]
            batch = collate((rows[i] for i in part), tokenizer, args.max_source, args.max_target)
            batch = {k: v.to(context.device, non_blocking=True) for k, v in batch.items()}
            with context.no_sync(criterion, skip=micro + 1 < args.accumulation):
                loss = criterion(**batch)
                if not context.all_true(bool(torch.isfinite(loss))):
                    skipped += 1
                    if skipped > 100:
                        raise SystemExit(f'loss не конечен в {skipped} батчах: обучение разошлось (--amp={args.amp})')
                    optimizer.zero_grad(set_to_none=True)
                    micro = 0
                    continue
                scaler.scale(loss / args.accumulation).backward()
            running += loss.item()
            seen += 1
            micro += 1
            if micro < args.accumulation:
                continue
            micro = 0
            for group in optimizer.param_groups:
                group['lr'] = args.lr * lr_factor(step, warmup, total)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            overflowed += scaler.get_scale() < scale  # масштаб снижается только после пропущенного шага
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                minutes = (time.monotonic() - started) / 60
                log(context, f'шаг {step}/{total} loss {running / max(seen, 1):.4f} {step / minutes:.1f} шаг/мин, '
                             f'пропущено шагов {overflowed}, масштаб {scaler.get_scale():.0f}')
            if step % eval_every == 0 or step >= total:
                evaluate_and_save('epoch' if not args.eval_every else 'steps')
                running, seen = 0.0, 0
            # Решение об остановке общее: часы процессов могут разойтись на границе лимита.
            over_time = context.any_true((time.monotonic() - started) / 3600 > args.time_limit_hours)
            if step >= total or over_time:
                if step < total:
                    log(context, f'лимит времени {args.time_limit_hours} ч: стоп на шаге {step}')
                    evaluate_and_save('time_limit')
                stop = True
                break
    # Остановка записывается: прогон короче плана не должен выглядеть полным в таблице результатов.
    history.append({'final_step': step, 'planned_steps': total, 'epochs_started': epoch, 'skipped_batches': skipped,
                    'overflowed_steps': overflowed})


def worker(rank: int, world: int, args, digest: str) -> None:
    context = setup(rank, world)
    device = context.device
    amp_dtype = AMP[args.amp] if device.type == 'cuda' else None
    # Сид модели общий (DDP всё равно сверит веса), а dropout у процессов должен отличаться.
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    folder = find_data(Path(args.data))
    languages = args.languages.split(',')
    log(context, f'код {digest}, {device}, процессов {world}, amp {args.amp}, языки {languages}, данные {folder}')
    log(context, json.dumps(environment(), ensure_ascii=False))

    tokenizer = load_tokenizer(args.tokenizer or args.model, args.revision if not args.tokenizer else None)
    model = tiny_model() if args.model == TINY else load_model(args.model, args.revision)
    torch.manual_seed(args.seed + rank)
    model = model.to(device)
    log(context, f'{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f} млн параметров')
    if args.amp == 'fp16':
        log(context, f'fp16: {keep_feed_forward_output_in_fp32(model)} слоёв wo считаются в fp32')

    history: list[dict] = []
    payload = {'args': vars(args), 'code_hash': digest, 'environment': environment(),
               'data_manifest': json.loads((folder / 'manifest.json').read_text()).get('revision'), 'history': history}

    def save_log() -> None:
        (out / 'training_log.json').write_text(json.dumps(payload, ensure_ascii=False, indent=1))

    validate = Validator(folder, languages, args.val_per_language, args, amp_dtype, context)
    started = time.monotonic()
    if not args.eval_only:  # эталон только пишет прогнозы: числа считает tools/score.py
        train(args, model, tokenizer, folder, languages, validate, out, context, amp_dtype, history, save_log)
        if context.is_main:
            save_log()
        context.barrier()  # все процессы читают лучший чекпойнт только после его записи
        del model
        torch.cuda.empty_cache()
        model = load_model(str(out / 'model')).to(device)
        if args.amp == 'fp16':
            keep_feed_forward_output_in_fp32(model)

    write_predictions(context, model, tokenizer, folder, languages, 'val', args.val_per_language, out, amp_dtype, args)
    if args.test_per_language:
        # Только запись прогнозов: test оценивается локально и только для заранее выбранных по val моделей.
        limit = None if args.test_per_language < 0 else args.test_per_language
        write_predictions(context, model, tokenizer, folder, languages, 'test', limit, out, amp_dtype, args)
    if context.is_main:
        payload['total_minutes'] = (time.monotonic() - started) / 60
        save_log()
    log(context, 'готово')
    context.barrier()
    teardown(context)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    digest = package_hash(Path(__file__).resolve().parent)
    if args.expect_code_hash and args.expect_code_hash != digest:
        raise SystemExit(f'код ядра {digest}, ожидался {args.expect_code_hash}: датасет кода не обновился')
    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit('GPU нет: квота исчерпана или ядро собрано без ускорителя')
    world = args.processes or max(torch.cuda.device_count(), 1)
    if world > 1:
        torch.multiprocessing.spawn(worker, args=(world, args, digest), nprocs=world)
    else:
        worker(0, 1, args, digest)


if __name__ == '__main__':
    main()
