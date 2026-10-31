from pathlib import Path

import pytest

from src.config import load_config, train_argv
from src.train import build_parser

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def test_extends_merges_and_slug_is_not_inherited(tmp_path):
    (tmp_path / 'base.yaml').write_text('slug: base\ntrain:\n  lr: 0.1\n  batch: 8\n')
    (tmp_path / 'child.yaml').write_text('extends: base.yaml\ntrain:\n  lr: 0.2\n')
    config = load_config(tmp_path / 'child.yaml')
    assert config['train'] == {'lr': 0.2, 'batch': 8}
    assert config['slug'] == 'xlsum-child'


def test_extends_cycle_and_unknown_section_fail(tmp_path):
    (tmp_path / 'a.yaml').write_text('extends: b.yaml\n')
    (tmp_path / 'b.yaml').write_text('extends: a.yaml\n')
    with pytest.raises(ValueError, match='цикл'):
        load_config(tmp_path / 'a.yaml')
    (tmp_path / 'c.yaml').write_text('trian: {}\n')
    with pytest.raises(ValueError, match='неизвестные секции'):
        load_config(tmp_path / 'c.yaml')


def test_unknown_or_launch_only_train_key_fails():
    with pytest.raises(ValueError, match='неизвестные'):
        train_argv(build_parser(), {'learning_rate': 1})
    with pytest.raises(ValueError, match='ставит запуск'):
        train_argv(build_parser(), {'expect_code_hash': 'x'})


def test_argv_round_trip():
    argv = train_argv(build_parser(), {'lr': 0.1, 'gradient_checkpointing': True, 'languages': ['russian', 'hindi']})
    args = build_parser().parse_args(['--data', 'd', '--out', 'o', *argv])
    assert args.lr == 0.1 and args.gradient_checkpointing and args.languages == 'russian,hindi'


@pytest.mark.parametrize('path', sorted((CONFIGS / 'experiments').glob('*.yaml')), ids=lambda p: p.stem)
def test_every_experiment_config_resolves(path):
    config = load_config(path)
    train_argv(build_parser(), config['train'])
    assert config['kaggle']['data_slug']
    assert config.get('note'), 'у эксперимента должна быть гипотеза'
