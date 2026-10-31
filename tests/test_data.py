import io
import json
import random
import tarfile

import pytest

from src.data import (clean_text, clean_train, epoch_indices, length_batches, model_input, read_archive, stable_subset)


def row(i, text, summary='резюме статьи'):
    return {'id': str(i), 'url': '', 'title': '', 'summary': summary, 'text': text}


LONG = 'длинный текст статьи ' * 10


def test_clean_text_keeps_paragraphs_for_baselines():
    assert clean_text('  a   b \n\n  c\t d ') == 'a b\nc d'
    assert model_input('a b\nc d') == 'a b c d'


def test_clean_train_removes_leaks_and_junk_but_not_val_test():
    splits = {
        'train': [row(1, LONG + 'one'), row(2, (LONG + 'one').upper()), row(3, LONG + 'val'), row(4, 'мало'),
                  row(5, LONG + 'tmpl', 'шаблонное резюме'), row(6, 'абв' * 40, 'абв' * 50), row(7, LONG + 'ok')],
        'val': [row(8, LONG + 'VAL', 'другое')],
        'test': [row(9, LONG + 'test', 'шаблонное резюме')],
    }
    kept, removed = clean_train(splits)
    assert [r['id'] for r in kept] == ['1', '7']
    assert removed == {'short_text': 1, 'summary_not_shorter': 1, 'text_in_val_test': 1, 'summary_in_val_test': 1,
                       'duplicate_text': 1}
    assert len(splits['val']) == len(splits['test']) == 1


def test_no_train_text_survives_in_held_out():
    splits = {'train': [row(i, f'{LONG} {i % 3}', f'резюме {i}') for i in range(30)],
              'val': [row(100, f'{LONG} 0', 'x')], 'test': [row(101, f'{LONG} 1', 'y')]}
    kept, _ = clean_train(splits)
    assert {r['text'] for r in kept} == {f'{LONG} 2'}


def test_stable_subset_ignores_order():
    rows = [{'id': str(i)} for i in range(100)]
    shuffled = rows[:]
    random.Random(0).shuffle(shuffled)
    assert stable_subset(rows, 10) == stable_subset(shuffled, 10)
    assert stable_subset(rows, None) is rows


def test_epoch_is_balanced_and_rotates_big_language():
    pools = {'big': list(range(1000)), 'small': list(range(1000, 1050))}
    rng = random.Random(0)
    first, second = epoch_indices(pools, 100, rng), epoch_indices(pools, 100, rng)
    assert sum(i < 1000 for i in first) == 100
    assert sum(i >= 1000 for i in first) == 50  # малый язык отдаёт всё, что есть, без повторов
    assert len(set(first)) == len(first)
    assert {i for i in first if i < 1000} != {i for i in second if i < 1000}


def test_length_batches_cover_everything_once():
    rng = random.Random(0)
    lengths = [rng.randrange(1000) for _ in range(103)]
    batches = list(length_batches(list(range(103)), lengths, 8, rng, window=2))
    assert sorted(i for batch in batches for i in batch) == list(range(103))
    assert max(map(len, batches)) <= 8


def make_archive(path, members):
    with tarfile.open(path, 'w:bz2') as archive:
        for name, rows in members.items():
            data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_read_archive_takes_only_expected_members(tmp_path):
    path = tmp_path / 'x.tar.bz2'
    good = [row(1, 'текст  с\n\nабзацем')]
    make_archive(path, {'./x_train.jsonl': good, './x_val.jsonl': good, './x_test.jsonl': good,
                        '../evil.jsonl': good, 'x_extra.txt': good})
    splits = read_archive(path)
    assert set(splits) == {'train', 'val', 'test'}
    assert splits['train'][0]['text'] == 'текст с\nабзацем'
    assert not (tmp_path.parent / 'evil.jsonl').exists()


def test_read_archive_fails_on_missing_split(tmp_path):
    path = tmp_path / 'x.tar.bz2'
    make_archive(path, {'x_train.jsonl': [row(1, 'a')]})
    with pytest.raises(ValueError, match='val'):
        read_archive(path)
