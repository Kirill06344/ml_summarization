"""Девять языков работы. Выбор и обоснование — docs/data.md."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    name: str        # имя в XL-Sum: имя архива и префикс файлов
    title: str       # для отчёта
    family: str
    script: str


LANGUAGES = {
    language.name: language for language in (
        Language('russian', 'Русский', 'индоевропейская (славянская)', 'кириллица'),
        Language('english', 'Английский', 'индоевропейская (германская)', 'латиница'),
        Language('spanish', 'Испанский', 'индоевропейская (романская)', 'латиница'),
        Language('hindi', 'Хинди', 'индоевропейская (индоарийская)', 'деванагари'),
        Language('arabic', 'Арабский', 'афразийская (семитская)', 'арабское письмо'),
        Language('chinese_simplified', 'Китайский (упрощ.)', 'сино-тибетская', 'ханьцзы'),
        Language('turkish', 'Турецкий', 'тюркская', 'латиница'),
        Language('indonesian', 'Индонезийский', 'австронезийская', 'латиница'),
        Language('vietnamese', 'Вьетнамский', 'австроазиатская', 'латиница'),
    )
}
NAMES = tuple(LANGUAGES)
