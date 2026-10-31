"""Перенос README в DOCX с титульным листом и стилями локального референса.

Нужны python-docx и markdown-it-py; схема строится matplotlib.
    .venv/bin/python tools/export_docx.py
Для обновления оглавления и экспорта PDF: LibreOffice / Microsoft Word.
"""
from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'report'
BUILD = Path('/tmp/xlsum-docx')
AUTHOR = 'Ролецкая Софья Николаевна'
FONT = 'Times New Roman'
TABLE_TITLES = [
    'Объём корпуса по языкам', 'Сравниваемые варианты модели', 'Параметры обучения',
    'Продолжительность и ход обучения', 'Качество на валидационной выборке',
    'Разница ROUGE-2 относительно исходного XL-Sum', 'ROUGE-2 по языкам',
    'Качество на тестовой выборке', 'Ошибки моделей в статье о деле Старовойтовой',
]
SOURCE_TITLES = {
    'https://github.com/csebuetnlp/xl-sum': 'BUET CSE NLP Group. XL-Sum: код, данные и модели. Официальный репозиторий.',
    'https://aclanthology.org/W04-3252/': 'Mihalcea R., Tarau P. TextRank: Bringing Order into Text // EMNLP. 2004. P. 404–411.',
    'https://aclanthology.org/2021.naacl-main.41/': 'Xue L. et al. mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer // NAACL-HLT. 2021. P. 483–498.',
    'https://aclanthology.org/2021.findings-acl.413/': 'Hasan T. et al. XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages // Findings of ACL-IJCNLP. 2021. P. 4693–4703.',
    'https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum': 'BUET CSE NLP Group. mT5-multilingual-XLSum. Карточка многоязычной модели.',
    'https://aclanthology.org/W04-1013/': 'Lin C.-Y. ROUGE: A Package for Automatic Evaluation of Summaries // Text Summarization Branches Out. 2004. P. 74–81.',
}


def element(tag: str, **attrs):
    node = OxmlElement(tag)
    for key, value in attrs.items():
        node.set(qn(key), str(value))
    return node


def font(run, size=14, bold=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        run.bold = bold
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn('w:rFonts'))
    if fonts is None:
        fonts = element('w:rFonts')
        rpr.insert(0, fonts)
    for key in ('ascii', 'hAnsi', 'cs', 'eastAsia'):
        fonts.set(qn('w:' + key), FONT)
    language = element('w:lang', **{'w:val': 'ru-RU'})
    rpr.append(language)
    return run


def typography(text):
    """Тире в обычном тексте; дефисы внутри терминов и команд не изменяются."""
    text = text.replace('–', '—')
    text = re.sub(r'(?<=\s)-(?=\s)', '—', text)
    return re.sub(r'(?<![\w/])-(?=\d)', '−', text)


def replace_paragraph(p, text, *, size=14, bold=False, alignment=None):
    p.clear()
    font(p.add_run(text), size, bold)
    if alignment is not None:
        p.alignment = alignment


def field(p, instruction, value='1'):
    run = p.add_run()
    run._r.append(element('w:fldChar', **{'w:fldCharType': 'begin'}))
    instr = element('w:instrText', **{'xml:space': 'preserve'})
    instr.text = ' ' + instruction + ' '
    run._r.append(instr)
    run._r.append(element('w:fldChar', **{'w:fldCharType': 'separate'}))
    font(p.add_run(value))
    end = p.add_run()
    end._r.append(element('w:fldChar', **{'w:fldCharType': 'end'}))


def bookmark(p, name, number):
    p._p.insert(0, element('w:bookmarkStart', **{'w:id': number, 'w:name': name}))
    p._p.append(element('w:bookmarkEnd', **{'w:id': number}))


def draw_diagram(path):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/xlsum-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8, 5.4))
    positions = {'A': (0, 3), 'B': (1, 3), 'C': (2, 3), 'D': (2, 2), 'E': (1, 2),
                 'F': (0, 2), 'G': (0, 1), 'H': (1, 1), 'I': (2, 1)}
    labels = {'A': 'XL-Sum\nОфициальные сплиты', 'B': 'Очистка train\nи parquet',
              'C': 'Балансировка\nязыков', 'D': 'Дообучение\nmT5',
              'E': 'Чекпойнты\nи прогнозы val', 'F': 'Lead / TextRank\nOracle',
              'G': 'ROUGE\nВыбор по val', 'H': 'Test победителя\nи бейзлайнов',
              'I': 'Таблицы, графики\nи отчёт'}
    for key, (x, y) in positions.items():
        ax.add_patch(FancyBboxPatch((x - .39, y - .22), .78, .44,
                                   boxstyle='round,pad=0.03', facecolor='#eef2f6',
                                   edgecolor='#333333', linewidth=1))
        ax.text(x, y, labels[key], ha='center', va='center', fontsize=11, fontfamily='DejaVu Sans')
    for a, b in [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('B', 'F'), ('F', 'E'),
                 ('E', 'G'), ('G', 'H'), ('H', 'I')]:
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        dx, dy = x2 - x1, y2 - y1
        if dy == 0:
            x1 += .43 * (1 if dx > 0 else -1)
            x2 -= .43 * (1 if dx > 0 else -1)
        elif dx == 0:
            y1 -= .27
            y2 += .27
        else:
            y1 -= .27
            y2 += .27
        ax.annotate('', (x2, y2), (x1, y1), arrowprops={'arrowstyle': '-|>', 'color': '#333333', 'lw': 1.2})
    ax.set(xlim=(-.5, 2.5), ylim=(.6, 3.4))
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


class Exporter:
    def __init__(self):
        OUT.mkdir(exist_ok=True, parents=True)
        BUILD.mkdir(exist_ok=True, parents=True)
        self.doc = Document(ROOT / 'Курсовая работа.docx')
        body = self.doc._element.body
        # Сохраняем титульные абзацы и разрыв первой секции из референса.
        cover = [copy.deepcopy(x) for x in list(body)[:34]]
        final_section = copy.deepcopy(body.sectPr)
        for child in list(body):
            body.remove(child)
        for child in cover:
            body.append(child)
        body.append(final_section)
        # Удаляем изображения и ссылки чужой работы из пакета.
        for key, rel in list(self.doc.part.rels.items()):
            if any(kind in rel.reltype for kind in ('/image', '/hyperlink', '/comments', '/customXml')):
                del self.doc.part.rels[key]
        p = self.doc.paragraphs
        replace_paragraph(p[17], 'по дисциплине «Машинное обучение»', alignment=WD_ALIGN_PARAGRAPH.CENTER)
        replace_paragraph(p[19], 'Многоязычная суммаризация новостей на XL-Sum', size=16,
                          bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)
        replace_paragraph(p[24], 'Выполнила\tгр. 5140904/50101')
        replace_paragraph(p[25], AUTHOR, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
        for index in (26, 27, 28, 29):
            replace_paragraph(p[index], '')
        replace_paragraph(p[31], 'Руководитель\tСелин Иван Андреевич')
        for paragraph in p[:34]:
            paragraph.paragraph_format.first_line_indent = Cm(0)
        for section in self.doc.sections:
            section.header.is_linked_to_previous = False
            for paragraph in section.header.paragraphs:
                paragraph.clear()
            for paragraph in section.first_page_footer.paragraphs:
                paragraph.paragraph_format.first_line_indent = Cm(0)
                for run in paragraph.runs:
                    if '2023' in run.text:
                        run.text = run.text.replace('2023', '2026')
            for paragraph in section.first_page_header.paragraphs:
                paragraph.paragraph_format.first_line_indent = Cm(0)
            section.footer.is_linked_to_previous = False
            for paragraph in section.footer.paragraphs:
                paragraph.clear()
        main = self.doc.sections[-1]
        main.different_first_page_header_footer = False
        num = main._sectPr.find(qn('w:pgNumType'))
        if num is None:
            num = element('w:pgNumType')
            main._sectPr.append(num)
        num.set(qn('w:start'), '1')
        fp = main.footer.paragraphs[0]
        fp.paragraph_format.first_line_indent = Cm(0)
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        field(fp, 'PAGE')
        for run in fp.runs:
            font(run, 12)
        self.styles()
        props = self.doc.core_properties
        props.title = 'Многоязычная суммаризация новостей на XL-Sum'
        props.subject = 'Курсовая работа по дисциплине «Машинное обучение»'
        props.author = props.last_modified_by = AUTHOR
        props.keywords = 'XL-Sum, mT5, суммаризация, машинное обучение'
        props.comments = ''
        props.created = props.modified = datetime.now(timezone.utc)
        self.md = MarkdownIt('commonmark').enable('table')
        self.sources = {}
        self.headings = []
        self.figure_count = self.table_count = 0
        self.chapter = ''
        self.subchapter = ''

    def styles(self):
        styles = self.doc.styles
        normal = styles['Normal']
        normal.font.name = FONT
        normal.font.size = Pt(14)
        normal.paragraph_format.line_spacing = 1.15
        normal.paragraph_format.first_line_indent = Cm(1.27)
        normal.paragraph_format.space_before = Pt(0)
        normal.paragraph_format.space_after = Pt(4)
        normal.paragraph_format.widow_control = True
        # В образце основной текст выровнен по левому краю.
        normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for level in (1, 2, 3):
            style = styles[f'Heading {level}']
            style.font.name = FONT
            style.font.size = Pt(16 if level == 1 else 14)
            style.font.bold = True
            style.font.color.rgb = RGBColor(0, 0, 0)
            style.paragraph_format.first_line_indent = Cm(0)
            style.paragraph_format.left_indent = Cm(0 if level == 1 else .63)
            style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT
            style.paragraph_format.space_before = Pt(12)
            style.paragraph_format.space_after = Pt(8)
            style.paragraph_format.keep_with_next = True
            style.paragraph_format.keep_together = True
            style.paragraph_format.line_spacing = 1.0
            style.paragraph_format.page_break_before = level == 1
            style.element.get_or_add_pPr().append(element('w:outlineLvl', **{'w:val': level - 1}))
        for name, size in [('Report Code', 10), ('Report Caption', 12), ('Report TOC', 12)]:
            if name not in styles:
                styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            style = styles[name]
            style.base_style = normal
            style.font.name = FONT
            style.font.size = Pt(size)
            style.paragraph_format.first_line_indent = Cm(0)
            style.paragraph_format.line_spacing = 1.0
            style.paragraph_format.space_after = Pt(6)
        styles['Report Caption'].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        styles['Report Caption'].paragraph_format.space_before = Pt(6)
        styles['Report Caption'].paragraph_format.space_after = Pt(8)
        styles['Report Caption'].paragraph_format.keep_with_next = False
        styles['Report Caption'].paragraph_format.keep_together = True
        styles['Report Code'].paragraph_format.line_spacing = 1.15

    def add_heading(self, text, level, *, toc=True):
        p = self.doc.add_paragraph(style=f'Heading {level}')
        font(p.add_run(typography(text)), 16 if level == 1 else 14, True)
        if toc:
            name = f'report_h{len(self.headings) + 1}'
            bookmark(p, name, len(self.headings) + 100)
            self.headings.append((level, text, name))
        return p

    def inline(self, p, children, *, size=14, bold=False):
        strong, italic = bold, False
        href = None
        for node in children:
            kind = node.type
            if kind == 'strong':
                self.inline(p, node.children, size=size, bold=True)
            elif kind == 'em':
                before = len(p.runs)
                self.inline(p, node.children, size=size, bold=strong)
                for run in p.runs[before:]:
                    run.italic = True
            elif kind == 'link':
                self.inline(p, node.children, size=size, bold=strong)
                url = node.attrGet('href')
                if url.startswith('http'):
                    if url not in self.sources:
                        self.sources[url] = len(self.sources) + 1
                    font(p.add_run(f' [{self.sources[url]}]'), size)
            elif kind == 'strong_open':
                strong = True
            elif kind == 'strong_close':
                strong = bold
            elif kind == 'em_open':
                italic = True
            elif kind == 'em_close':
                italic = False
            elif kind == 'link_open':
                href = node.attrGet('href')
            elif kind == 'link_close':
                if href and href.startswith('http'):
                    if href not in self.sources:
                        self.sources[href] = len(self.sources) + 1
                    font(p.add_run(f' [{self.sources[href]}]'), size)
                href = None
            elif kind in ('text', 'code_inline'):
                content = node.content if kind == 'code_inline' else typography(node.content)
                run = font(p.add_run(content), min(size, 11) if kind == 'code_inline' else size, strong)
                run.italic = italic
                if kind == 'code_inline':
                    run._r.get_or_add_rPr().append(element('w:shd', **{'w:fill': 'F2F2F2'}))
            elif kind in ('softbreak', 'hardbreak'):
                p.add_run(' ' if kind == 'softbreak' else '\n')
            elif kind == 'image':
                self.picture(ROOT / node.attrGet('src'), node.content)

    def paragraph(self, node, *, prefix='', size=14, quote=False):
        content = node.children[0].content if node.children else ''
        if content.startswith('$$'):
            self.formulas()
            return
        if node.children and any(x.type == 'image' for x in node.children[0].children):
            for image in node.children[0].children:
                if image.type == 'image':
                    self.picture(ROOT / image.attrGet('src'), image.content)
            return
        p = self.doc.add_paragraph()
        if content.rstrip().endswith(':'):
            p.paragraph_format.keep_with_next = True
        if prefix:
            p.paragraph_format.first_line_indent = Cm(-.5)
            p.paragraph_format.left_indent = Cm(.5)
            font(p.add_run(prefix), size)
        if quote:
            p.paragraph_format.left_indent = Cm(1)
            p.paragraph_format.right_indent = Cm(.5)
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
        if node.children:
            self.inline(p, node.children[0].children, size=size)
        return p

    def picture(self, path, caption):
        p = self.doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Cm(17.0))
        self.figure_count += 1
        p = self.doc.add_paragraph(style='Report Caption')
        font(p.add_run(f'Рисунок {self.figure_count} — {caption}'), 12)

    def table(self, node):
        rows = []
        for part in node.children:
            for row in part.children:
                rows.append(row.children)
        self.table_count += 1
        columns = len(rows[0])
        table = self.doc.add_table(rows=len(rows), cols=columns)
        table.autofit = False
        # Сетка и поля ячеек задаются явно, без зависимости от стилей Word.
        widths = {2: [5.4, 11.6], 3: [7.6, 4.7, 4.7], 4: [6.8, 3.4, 3.4, 3.4],
                  5: [3.4, 3.4, 3.4, 3.4, 3.4], 6: [4.7, 2.0, 2.0, 2.0, 3.8, 2.5]}[columns]
        if columns == 6 and rows[0][1].children[0].content == 'Шаги':
            widths = [4.5, 2.7, 1.2, 2.8, 3.0, 2.8]
        props = table._tbl.tblPr
        borders = element('w:tblBorders')
        for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            borders.append(element('w:' + edge, **{'w:val': 'single', 'w:sz': '4', 'w:color': '555555'}))
        props.append(borders)
        for column, width in zip(table.columns, widths):
            column.width = Cm(width)
        for i, row in enumerate(rows):
            trpr = table.rows[i]._tr.get_or_add_trPr()
            trpr.append(element('w:cantSplit'))
            if i == 0:
                trpr.append(element('w:tblHeader'))
            for j, cell_node in enumerate(row):
                cell = table.cell(i, j)
                cell.width = Cm(widths[j])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr = cell._tc.get_or_add_tcPr()
                if i == 0:
                    tcpr.append(element('w:shd', **{'w:fill': 'E7E6E6'}))
                margins = element('w:tcMar')
                for edge in ('top', 'bottom', 'left', 'right'):
                    margins.append(element('w:' + edge, **{'w:w': '65', 'w:type': 'dxa'}))
                tcpr.append(margins)
                p = cell.paragraphs[0]
                p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1.15
                # Строки не разрываются, таблица остаётся вместе с нижней подписью.
                p.paragraph_format.keep_with_next = True
                self.inline(p, cell_node.children[0].children, size=11, bold=i == 0)
        caption = self.doc.add_paragraph(style='Report Caption')
        title = TABLE_TITLES[self.table_count - 1]
        font(caption.add_run(f'Таблица {self.table_count} — {title}'), 12)

    def formulas(self):
        def math_run(text):
            r = element('m:r')
            properties = element('m:rPr')
            properties.append(element('m:nor'))
            r.append(properties)
            t = element('m:t')
            t.text = text
            r.append(t)
            return r
        for name, numerator, denominator in [
            ('P = ', 'число совпадений n-грамм', 'число n-грамм прогноза'),
            ('R = ', 'число совпадений n-грамм', 'число n-грамм эталона'),
            ('F₁ = ', '2PR', 'P + R')]:
            p = self.doc.add_paragraph()
            p.paragraph_format.first_line_indent = Cm(0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.keep_together = True
            mp = element('m:oMathPara')
            m = element('m:oMath')
            m.append(math_run(name))
            fraction = element('m:f')
            num, den = element('m:num'), element('m:den')
            num.append(math_run(numerator))
            den.append(math_run(denominator))
            fraction.extend([num, den])
            m.append(fraction)
            mp.append(m)
            p._p.append(mp)

    def render(self, nodes, *, quote=False):
        for node in nodes:
            if node.type == 'heading':
                if node.tag == 'h1':
                    continue  # Название работы уже на титульном листе.
                text = node.children[0].content
                level = int(node.tag[1]) - 1
                if level == 1:
                    self.chapter = text
                    self.subchapter = ''
                else:
                    self.subchapter = text
                self.add_heading(text, level)
            elif node.type == 'paragraph':
                self.paragraph(node, quote=quote)
            elif node.type == 'table':
                self.table(node)
            elif node.type in ('bullet_list', 'ordered_list'):
                for i, item in enumerate(node.children, 1):
                    for paragraph in item.children:
                        if paragraph.type == 'paragraph':
                            self.paragraph(paragraph, prefix='• ' if node.type == 'bullet_list' else f'{i}. ')
                        else:
                            self.render([paragraph])
            elif node.type == 'blockquote':
                self.render(node.children, quote=True)
            elif node.type == 'fence':
                if node.info.strip() == 'mermaid':
                    path = BUILD / 'architecture.png'
                    draw_diagram(path)
                    self.picture(path, 'Архитектура экспериментального тракта')
                else:
                    p = self.doc.add_paragraph(style='Report Code')
                    p.paragraph_format.keep_together = True
                    font(p.add_run(node.content.rstrip()), 10)
                    p._p.get_or_add_pPr().append(element('w:shd', **{'w:fill': 'F2F2F2'}))
            elif node.type == 'html_block':
                continue
            else:
                raise ValueError(f'Необработанный блок Markdown: {node.type}')

    def add_toc(self, anchor):
        # Кэш оглавления виден сразу; Word/LibreOffice обновляют номера полями PAGEREF.
        p = self.doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(12)
        font(p.add_run('Содержание'), 16)
        anchor.addprevious(p._p)
        start = self.doc.add_paragraph()
        run = start.add_run()
        run._r.append(element('w:fldChar', **{'w:fldCharType': 'begin'}))
        instr = element('w:instrText', **{'xml:space': 'preserve'})
        instr.text = ' TOC \\o "1-2" \\h \\z \\u '
        run._r.append(instr)
        run._r.append(element('w:fldChar', **{'w:fldCharType': 'separate'}))
        start.paragraph_format.space_after = Pt(0)
        start.paragraph_format.line_spacing = Pt(1)
        anchor.addprevious(start._p)
        for level, text, name in self.headings:
            p = self.doc.add_paragraph(style='Report TOC')
            p.paragraph_format.left_indent = Cm(0 if level == 1 else .65)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.tab_stops.add_tab_stop(Cm(17), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            link = element('w:hyperlink', **{'w:anchor': name})
            run = self.doc.add_paragraph().add_run(text)
            font(run, 12, level == 1)
            empty = run._r.getparent()
            link.append(run._r)
            empty.getparent().remove(empty)
            p._p.append(link)
            p.add_run('\t')
            field(p, f'PAGEREF {name} \\h', '…')
            anchor.addprevious(p._p)
        end = self.doc.add_paragraph()
        end.add_run()._r.append(element('w:fldChar', **{'w:fldCharType': 'end'}))
        end.paragraph_format.line_spacing = Pt(1)
        end.paragraph_format.space_after = Pt(0)
        anchor.addprevious(end._p)
        # Введение перед первым разделом переносится на отдельную страницу от оглавления.
        anchor.get_or_add_pPr().append(element('w:pageBreakBefore'))
        settings = self.doc.settings.element
        settings.append(element('w:updateFields', **{'w:val': 'true'}))

    def build(self):
        intro = self.doc.add_paragraph()
        nodes = SyntaxTreeNode(self.md.parse((ROOT / 'README.md').read_text(encoding='utf-8'))).children
        # Вводные сведения README идут в начале первого раздела, без отдельной пустой главы.
        self.render([nodes[2], nodes[1], *nodes[3:]])
        # Все примеры доступны и при чтении DOCX без файлов репозитория.
        self.add_heading('Приложение А. Примеры резюме', 1)
        examples = SyntaxTreeNode(self.md.parse((ROOT / 'docs/results/examples.md').read_text(encoding='utf-8')))
        for node in examples.children:
            if node.type == 'heading' and node.tag == 'h1':
                continue
            if node.type == 'heading':
                self.add_heading(node.children[0].content.replace('`', ''), 2, toc=False)
            elif node.type == 'paragraph':
                self.paragraph(node, size=12)
            else:
                self.render([node])
        self.add_heading('Список использованных источников', 1)
        for url, index in self.sources.items():
            p = self.doc.add_paragraph()
            p.paragraph_format.first_line_indent = Cm(0)
            font(p.add_run(typography(f'[{index}] {SOURCE_TITLES.get(url, "Электронный ресурс.")} ')), 12)
            link = element('w:hyperlink')
            from docx.opc.constants import RELATIONSHIP_TYPE
            rel = self.doc.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
            link.set(qn('r:id'), rel)
            r = element('w:r')
            rpr = element('w:rPr')
            rpr.append(element('w:rFonts', **{'w:ascii': FONT, 'w:hAnsi': FONT}))
            rpr.append(element('w:sz', **{'w:val': '24'}))
            r.append(rpr)
            t = element('w:t')
            t.text = url
            r.append(t)
            link.append(r)
            p._p.append(link)
            font(p.add_run(' (дата обращения: 31.10.2026).'), 12)
        self.add_toc(intro._p)
        intro._p.getparent().remove(intro._p)
        path = OUT / 'Курсовая работа — Машинное обучение — Ролецкая.docx'
        self.doc.save(path)
        # ASCII-копия для стабильного экспорта в Linux-контейнере.
        (BUILD / 'report.docx').write_bytes(path.read_bytes())
        print(path)
        print(f'Разделов в оглавлении: {len(self.headings)}; таблиц: {self.table_count}; рисунков: {self.figure_count}')


if __name__ == '__main__':
    Exporter().build()
