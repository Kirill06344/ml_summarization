import json

import pytest

from src.languages import NAMES
from src.metrics import METRICS
from tools.collect_results import MODELS, choose_final, load_scores, update_readme


def test_scores_reject_partial_language_set_and_wrong_macro(tmp_path):
    folder = tmp_path / 'model'
    folder.mkdir()
    path = folder / 'scores_val.json'
    report = {'per_language': {lang: dict.fromkeys(METRICS, 0.2) | {'n': 500} for lang in NAMES},
              'macro': dict.fromkeys(METRICS, 0.2), 'rouge2_ci95': [0.1, 0.3]}
    path.write_text(json.dumps(report))
    assert load_scores(tmp_path, 'model', 'val', 500) == report
    report['per_language'].pop(NAMES[0])
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='9 языков'):
        load_scores(tmp_path, 'model', 'val', 500)
    report['per_language'][NAMES[0]] = dict.fromkeys(METRICS, 0.2) | {'n': 500}
    report['macro']['rouge2'] = 0.8
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='macro'):
        load_scores(tmp_path, 'model', 'val', 500)


def test_final_must_be_chosen_on_validation():
    reports = {run: {'macro': {'rouge2': score}} for run, score in zip(MODELS, [0.16, 0.1, 0.13, 0.155])}
    assert choose_final(reports, MODELS[0]) == MODELS[0]
    with pytest.raises(ValueError, match='test не используется'):
        choose_final(reports, MODELS[3])


def test_update_readme_preserves_surrounding_text_and_is_repeatable(tmp_path):
    path = tmp_path / 'README.md'
    path.write_text('Before\n<!-- results:val:start -->old<!-- results:val:end -->\nAfter\n')
    update_readme(path, {'val': '| data |\n'})
    first = path.read_text()
    update_readme(path, {'val': '| data |\n'})
    assert path.read_text() == first
    assert first.startswith('Before\n') and first.endswith('\nAfter\n') and 'old' not in first
    with pytest.raises(ValueError, match='ровно один блок'):
        update_readme(path, {'missing': 'data'})
