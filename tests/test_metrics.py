import numpy as np
import pytest

from src.metrics import METRICS, bootstrap_macro, rouge, score_pairs, summarize, tokenize


@pytest.mark.parametrize('text, expected', [
    ('Президент прибыл в Бирму.', ['президент', 'прибыл', 'в', 'бирму']),
    ('राष्ट्रपति आज आए।', ['राष्ट्रपति', 'आज', 'आए']),                 # матры — категория M, слово не рвётся
    ('وصل الرئيس، اليوم', ['وصل', 'الرئيس', 'اليوم']),
    ('总统今天', ['总', '统', '今', '天']),
    ('Tổng thống đến', ['tổng', 'thống', 'đến']),
    ('COVID-19 in 2021', ['covid', '19', 'in', '2021']),
    ('', []),
])
def test_tokenize_keeps_every_script(text, expected):
    assert tokenize(text) == expected


def test_nonlatin_scores_are_not_zero():
    # Регресс стандартного rouge_score: на кириллице он отдаёт ноль.
    assert rouge('президент прибыл в бирму', 'президент прибыл в бирму') == dict.fromkeys(METRICS, 1.0)


def test_disjoint_and_empty():
    assert rouge('один два', 'три четыре') == dict.fromkeys(METRICS, 0.0)
    assert rouge('', 'три') == dict.fromkeys(METRICS, 0.0)
    assert rouge('три', '') == dict.fromkeys(METRICS, 0.0)


def test_single_token_has_no_bigrams():
    assert rouge('слово', 'слово')['rouge2'] == 0.0


@pytest.mark.parametrize('prediction, reference', [
    ('the quick brown fox jumps', 'the quick red fox jumps over the dog'),
    ('a b a b c', 'a a b b c c'),
    ('police arrested two men on tuesday', 'two men were arrested by police'),
    ('the the the', 'the cat sat on the mat'),
])
def test_matches_reference_implementation_on_english(prediction, reference):
    rouge_scorer = pytest.importorskip('rouge_score.rouge_scorer')
    expected = rouge_scorer.RougeScorer(list(METRICS), use_stemmer=False).score(reference, prediction)
    actual = rouge(prediction, reference)
    for metric in METRICS:
        assert actual[metric] == pytest.approx(expected[metric].fmeasure)


def test_macro_gives_languages_equal_weight():
    big = np.ones((1000, 3))
    small = np.zeros((10, 3))
    assert summarize({'big': big, 'small': small}).macro['rouge2'] == pytest.approx(0.5)


def test_score_pairs_empty_shape():
    assert score_pairs([], []).shape == (0, 3)


def test_bootstrap_contains_point_and_shrinks_with_data():
    rng = np.random.default_rng(1)
    small = {'a': rng.random((50, 3))}
    large = {'a': rng.random((5000, 3))}
    point, low, high = bootstrap_macro(small)
    assert low <= point <= high
    _, low_large, high_large = bootstrap_macro(large)
    assert high_large - low_large < high - low


def test_paired_bootstrap_of_identical_runs_is_zero():
    scores = {'a': np.random.default_rng(2).random((100, 3))}
    assert bootstrap_macro(scores, baseline=scores) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_rejects_other_documents():
    with pytest.raises(ValueError):
        bootstrap_macro({'a': np.zeros((5, 3))}, baseline={'a': np.zeros((6, 3))})
