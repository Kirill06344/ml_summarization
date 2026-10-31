# Команды

Запускать из корня репозитория.

## 1. Окружение

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

## 2. Данные

```bash
.venv/bin/python tools/prepare_data.py                                             # 1,2 ГБ в data/processed
.venv/bin/python tools/kaggle_upload.py --dir data/processed --slug xlsum-nine-languages
```

`kaggle_upload` создаёт приватный датасет и проверяет приватность запросом к серверу. Новая версия: `--message "что изменилось"`.

## 3. Бейзлайны

```bash
.venv/bin/python tools/run_baselines.py --split val --limit 500
.venv/bin/python tools/score.py --split val baseline-lead baseline-textrank baseline-oracle
```

## 4. Обучение

```bash
.venv/bin/python tools/kaggle_train.py --config configs/experiments/mt5-base.yaml --dry-run
.venv/bin/python tools/kaggle_train.py --config configs/experiments/mt5-base.yaml
.venv/bin/python tools/kaggle_train.py --config configs/experiments/mt5-base.yaml --fetch-only
```

Результат ложится в `runs/<слаг>/`: `model/`, `predictions/`, `training_log.json`, `config.yaml`. Перед запуском проверить квоту и
активные ядра. `--fetch-only` забирает готовый результат и не запускает обучение заново.

## 5. Выбор модели и оценка

Текущие эксперименты уже завершены: по val выбран `xlsum-reference`, test для него и бейзлайнов
уже посчитан. Итоговый отчёт — [README.md](../README.md). Для повторной сборки достаточно
`collect_results.py`, без повторного обучения и подбора по test.

```bash
.venv/bin/python tools/score.py --split val xlsum-reference xlsum-mt5-small xlsum-mt5-base xlsum-mt5-xlsum-finetune \
    --against xlsum-reference
```

Test считается один раз, после выбора по val и только для отобранных моделей:

```bash
.venv/bin/python tools/run_baselines.py --split test --limit 2000
.venv/bin/python tools/score.py --split test baseline-lead baseline-textrank baseline-oracle xlsum-reference <финальная>
.venv/bin/python tools/collect_results.py --final <финальная>
```

Если финальная модель совпадает с эталоном, в список для `score.py` она включается один раз.
Для текущих готовых результатов:

```bash
.venv/bin/python tools/collect_results.py --final xlsum-reference
```

Сборщик создаёт таблицы, CSV, примеры, графики и provenance с хэшами источников в `docs/results/`,
а также обновляет числовые блоки README. Источник метрик — только `runs/*/scores_*.json`.

## 6. Использование модели

```bash
.venv/bin/python tools/summarize.py --model runs/<финальная>/model --file article.txt
```
