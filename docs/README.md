# Документация

Итоговый отчёт по курсовой — [README.md](../README.md). Здесь — рабочие подробности.

| Вопрос | Страница |
|---|---|
| Данные, выбор языков, очистка, длины | [data.md](data.md) |
| Метрика и сравнимость с публикацией | [metrics.md](metrics.md) |
| Модели и причины выбора | [models.md](models.md) |
| Рецепт и устройство обучения | [training.md](training.md) |
| Гипотезы, правила, бюджет, отклонения | [experiment_protocol.md](experiment_protocol.md) |
| Команды от данных до отчёта | [runbook.md](runbook.md) |
| Ошибки, на которые мы уже наступили | [pitfalls.md](pitfalls.md) |
| Сгенерированные таблицы и примеры | [results/](results/) |

## Карта кода

```text
XL-Sum (архивы) → tools/prepare_data.py → data/processed/*.parquet + manifest.json
configs/experiments/*.yaml → src/config.py → tools/kaggle_train.py → src/train.py
    src/data.py + src/modeling.py + src/distributed.py → training_log.json, model/
    src/predict.py → predictions/<сплит>/<язык>.jsonl
tools/run_baselines.py (src/baselines.py) → такие же прогнозы для бейзлайнов
predictions → tools/score.py (src/metrics.py) → tools/collect_results.py → docs/results/
```
