import pytest

from src.baselines import lead, oracle, split_sentences, textrank
from src.metrics import rouge


@pytest.mark.parametrize('text, expected', [
    ('Первое. Второе! Третье?', ['Первое.', 'Второе!', 'Третье?']),
    ('Абзац один\nАбзац два', ['Абзац один', 'Абзац два']),
    ('今天下雨。明天晴天！后天呢？', ['今天下雨。', '明天晴天！', '后天呢？']),
    ('आज बारिश है। कल धूप है।', ['आज बारिश है।', 'कल धूप है।']),
    ('هل ستمطر؟ نعم ستمطر.', ['هل ستمطر؟', 'نعم ستمطر.']),
    ('Цена 3.5 рубля.', ['Цена 3.5 рубля.']),
    ('', []),
])
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


def test_lead_takes_first_sentence():
    assert lead('Первое предложение. Второе.') == 'Первое предложение.'
    assert lead('') == ''


def test_textrank_picks_central_sentence():
    text = ('Кошка спит на диване дома. Собака бежит по улице быстро. '
            'Кошка и собака спят на диване дома. Самолёт летит над морем высоко.')
    assert 'Кошка и собака' in textrank(text)


def test_textrank_degenerate_inputs():
    assert textrank('') == ''
    assert textrank('Одно.') == 'Одно.'
    assert textrank('A. B. C.') in ('A.', 'B.', 'C.')


def test_oracle_is_not_worse_than_other_methods():
    text = 'Погода сегодня хорошая. Президент прибыл в Бирму с визитом. Рынки закрылись ростом.'
    reference = 'Президент прибыл в Бирму'
    best = rouge(oracle(text, reference), reference)['rouge2']
    assert best >= rouge(lead(text), reference)['rouge2']
    assert best >= rouge(textrank(text), reference)['rouge2']
    assert best > 0
