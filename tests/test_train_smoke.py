"""Сквозной тракт на случайной крошечной mT5: обучение, выбор чекпойнта, прогнозы, журнал. Качество здесь не проверяется."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.modeling import SummarizerLoss, smoothed_loss, keep_feed_forward_output_in_fp32, tiny_model
from src.predict import read_predictions
from src.train import lr_factor, main


def write_data(folder: Path) -> None:
    rows = [{'id': str(i), 'url': '', 'title': '', 'summary': f'краткое резюме номер {i}',
             'text': f'Длинная статья номер {i}. ' * 20} for i in range(12)]
    for language in ('russian', 'hindi'):
        for split in ('train', 'val', 'test'):
            pd.DataFrame(rows).to_parquet(folder / f'{language}_{split}.parquet', index=False)
    (folder / 'manifest.json').write_text(json.dumps({'revision': 'test'}))


def tokenizer_available() -> bool:
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained('google/mt5-small', local_files_only=True, use_fast=True, legacy=False)
        return True
    except Exception:  # noqa: BLE001 — нет в кэше: дым пропускается, а не падает без сети
        return False


@pytest.mark.skipif(not tokenizer_available(), reason='нет токенизатора mt5-small в кэше Hugging Face')
def test_end_to_end(tmp_path):
    data, out = tmp_path / 'data', tmp_path / 'out'
    data.mkdir()
    write_data(data)
    main(['--data', str(data), '--out', str(out), '--model', 'tiny-random', '--tokenizer', 'google/mt5-small',
          '--languages', 'russian,hindi', '--train-per-language', '8', '--val-per-language', '2',
          '--test-per-language', '3', '--epochs', '2', '--batch', '2', '--accumulation', '2',
          '--max-source', '32', '--max-target', '8', '--beams', '1', '--no-repeat-ngram', '0', '--eval-batch', '2'])
    log = json.loads((out / 'training_log.json').read_text())
    assert log['code_hash'] and log['args']['seed'] == 1
    evaluations = [r for r in log['history'] if 'val_rouge2' in r]
    assert [r['step'] for r in evaluations] == [4, 8]
    assert log['history'][-1]['final_step'] == 8 == log['history'][-1]['planned_steps']
    assert (out / 'model' / 'config.json').exists()
    assert {len(v) for v in read_predictions(out / 'predictions' / 'val').values()} == {2}
    assert {len(v) for v in read_predictions(out / 'predictions' / 'test').values()} == {3}


@pytest.mark.skipif(not tokenizer_available(), reason='нет токенизатора mt5-small в кэше Hugging Face')
def test_two_processes_stay_in_sync(tmp_path):
    # На GPU расхождение процессов выглядит как вечное зависание, поэтому проверяется здесь: два процесса на CPU.
    data, out = tmp_path / 'data', tmp_path / 'out'
    data.mkdir()
    write_data(data)
    main(['--data', str(data), '--out', str(out), '--model', 'tiny-random', '--tokenizer', 'google/mt5-small',
          '--processes', '2', '--languages', 'russian,hindi', '--train-per-language', '8', '--val-per-language', '3',
          '--test-per-language', '3', '--epochs', '1', '--batch', '4', '--accumulation', '2', '--max-source', '32',
          '--max-target', '8', '--beams', '1', '--no-repeat-ngram', '0', '--eval-batch', '2'])
    log = json.loads((out / 'training_log.json').read_text())
    assert log['history'][-1]['final_step'] == 2
    predictions = read_predictions(out / 'predictions' / 'test')
    assert {len(rows) for rows in predictions.values()} == {3}
    assert {row['id'] for row in predictions['russian']} == {row['id'] for row in predictions['hindi']}


@pytest.mark.skipif(not tokenizer_available(), reason='нет токенизатора mt5-small в кэше Hugging Face')
def test_eval_only_writes_predictions_without_training(tmp_path):
    data, out = tmp_path / 'data', tmp_path / 'out'
    data.mkdir()
    write_data(data)
    main(['--data', str(data), '--out', str(out), '--model', 'tiny-random', '--tokenizer', 'google/mt5-small',
          '--eval-only', '--languages', 'russian', '--val-per-language', '2', '--test-per-language', '2',
          '--max-source', '32', '--max-target', '8', '--beams', '1', '--no-repeat-ngram', '0', '--eval-batch', '2'])
    assert json.loads((out / 'training_log.json').read_text())['history'] == []
    assert not (out / 'model').exists()
    assert len(read_predictions(out / 'predictions' / 'test')['russian']) == 2


def test_hash_mismatch_stops_before_anything(tmp_path):
    with pytest.raises(SystemExit, match='датасет кода'):
        main(['--data', str(tmp_path), '--out', str(tmp_path / 'o'), '--expect-code-hash', 'wrong'])


def test_lr_schedule():
    assert lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert lr_factor(100, 10, 100) == 0.0


def test_smoothed_loss_ignores_padding():
    logits = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    full = smoothed_loss(logits, torch.tensor([[0, -100]]), 0.0)
    only = smoothed_loss(logits[:, :1], torch.tensor([[0]]), 0.0)
    assert full.item() == pytest.approx(only.item())


def test_loss_on_present_tokens_equals_full_forward():
    model = tiny_model().eval()
    input_ids = torch.tensor([[5, 6, 7, 1], [8, 1, 0, 0]])
    attention_mask = (input_ids != 0).long()
    labels = torch.tensor([[9, 10, 11, 1], [12, 1, -100, -100]])
    logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
    expected = smoothed_loss(logits, labels, 0.1)
    actual = SummarizerLoss(model, 0.1, None)(input_ids, attention_mask, labels)
    assert actual.item() == pytest.approx(expected.item(), rel=1e-6)


def test_fp32_output_layer_is_identical_outside_autocast():
    model = tiny_model().eval()
    ids = torch.tensor([[5, 6, 7, 1]])
    before = model(input_ids=ids, decoder_input_ids=ids).logits
    assert keep_feed_forward_output_in_fp32(model) > 0
    after = model(input_ids=ids, decoder_input_ids=ids).logits
    assert torch.equal(before, after)


def test_fp32_output_layer_uses_replica_weights():
    # Так DataParallel создаёт реплики: копия модуля со своими весами должна считать на них, а не на весах оригинала.
    model = tiny_model()
    keep_feed_forward_output_in_fp32(model)
    layer = next(module for name, module in model.named_modules() if name.endswith('.wo'))
    replica = copy.copy(layer)
    replica._parameters = {'weight': torch.nn.Parameter(torch.zeros_like(layer.weight)), 'bias': None}
    assert torch.equal(replica(torch.ones(1, 32)), torch.zeros(1, 16))


def test_fp32_output_layer_does_not_overflow_half_precision():
    model = tiny_model()
    layer = next(module for name, module in model.named_modules() if name.endswith('.wo'))
    torch.nn.init.constant_(layer.weight, 1e4)  # масштаб реальных весов mT5: выход не помещается в fp16
    hidden = torch.full((1, 2, 32), 1.0)
    with torch.autocast('cpu', dtype=torch.float16):
        assert not torch.isfinite(layer(hidden)).all()
    keep_feed_forward_output_in_fp32(model)
    with torch.autocast('cpu', dtype=torch.float16):
        output = layer(hidden)
    assert output.dtype == torch.float32 and torch.isfinite(output).all()
