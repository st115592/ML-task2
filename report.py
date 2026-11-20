"""Сборка отчёта A4 со сквозной нумерацией и оглавлением."""

import os
from collections import Counter
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.doctemplate import IndexingFlowable
from reportlab.platypus.tableofcontents import TableOfContents

from cnc_data import Recording
from experiment import (
    BOOST,
    FOREST,
    TRANSFER_REVIEW_FRACTION,
    Evaluation,
    MachineTransfer,
    Selection,
    candidates,
)
from figures import Figures

LEFT_MARGIN = 30 * mm
RIGHT_MARGIN = 15 * mm
WIDTH = A4[0] - LEFT_MARGIN - RIGHT_MARGIN


def configure_fonts() -> None:
    font_dir = Path(
        os.environ.get("CNC_FONT_DIR", Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")
    )
    for name, file in [("Times", "times.ttf"), ("Times-Bold", "timesbd.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    pdfmetrics.registerFontFamily("Times", normal="Times", bold="Times-Bold")


def styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle(
        "Body",
        fontName="Times",
        fontSize=12,
        leading=18,
        alignment=TA_JUSTIFY,
        firstLineIndent=12.5 * mm,
        spaceAfter=5,
    )
    return {
        "body": body,
        "section": ParagraphStyle(
            "Section",
            parent=body,
            fontName="Times-Bold",
            firstLineIndent=0,
            spaceAfter=12,
            keepWithNext=True,
        ),
        "appendix": ParagraphStyle(
            "Appendix",
            parent=body,
            fontName="Times-Bold",
            alignment=TA_CENTER,
            firstLineIndent=0,
            spaceAfter=12,
            keepWithNext=True,
        ),
        "center": ParagraphStyle("Center", parent=body, alignment=TA_CENTER, firstLineIndent=0),
        "title": ParagraphStyle(
            "Title",
            parent=body,
            fontName="Times-Bold",
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            firstLineIndent=0,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=body,
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            firstLineIndent=0,
            spaceAfter=6,
        ),
        "table_caption": ParagraphStyle(
            "TableCaption",
            fontName="Times",
            fontSize=11,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "keywords": ParagraphStyle(
            "Keywords",
            parent=body,
            alignment=TA_LEFT,
            firstLineIndent=0,
        ),
        "reference": ParagraphStyle("Reference", parent=body, keepWithNext=True),
        "cell": ParagraphStyle(
            "Cell", parent=body, fontSize=9.5, leading=12, firstLineIndent=0, spaceAfter=0
        ),
        "toc": ParagraphStyle("Toc", parent=body, firstLineIndent=0, leading=24),
    }


class ReportDocument(SimpleDocTemplate):
    def afterFlowable(self, flowable: Any) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name in {"Section", "Appendix"}:
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))


class ReportExtent(IndexingFlowable):
    """Подставляет число страниц после стабилизации вёрстки."""

    def __init__(self, style: ParagraphStyle) -> None:
        super().__init__()
        self.style = style
        self.pages = 0
        self.previous_pages = 0
        self.counts = ""

    def beforeBuild(self) -> None:
        self.previous_pages, self.pages = self.pages, 0

    def notify(self, kind: str, stuff: Any) -> None:
        if kind == "ReportPage":
            self.pages = stuff

    def isSatisfied(self) -> bool:
        return self.pages == self.previous_pages

    def wrap(self, availWidth: float, availHeight: float) -> tuple[float, float]:
        self.paragraph = Paragraph(f"Отчёт {self.previous_pages} с., {self.counts}.", self.style)
        return self.paragraph.wrap(availWidth, availHeight)

    def draw(self) -> None:
        self.paragraph.drawOn(self.canv, 0, 0)


def footer(canvas: Canvas, doc: ReportDocument) -> None:
    doc.notify("ReportPage", doc.page)
    if doc.page > 1:
        canvas.saveState()
        canvas.setFont("Times", 12)
        canvas.drawCentredString(A4[0] / 2, 10 * mm, str(doc.page))
        canvas.restoreState()


class Report:
    def __init__(self, figures: Figures) -> None:
        self.figures = figures
        self.style = styles()
        self.story: list[Any] = []
        self.image_count = 0
        self.table_count = 0
        self.appendix_count = 0
        self.appendix_letter = ""
        self.appendix_figures = 0
        self.appendix_tables = 0

    def p(self, value: str, style: str = "body") -> None:
        self.story.append(Paragraph(escape(value), self.style[style]))

    def section(self, title: str) -> None:
        self.story.append(PageBreak())
        self.p(title, "section")

    def appendix(self, letter: str, title: str) -> None:
        self.appendix_count += 1
        self.appendix_letter = letter
        self.appendix_figures = 0
        self.appendix_tables = 0
        self.story.append(PageBreak())
        self.p(f"ПРИЛОЖЕНИЕ {letter} — {title}", "appendix")

    def figure(self, key: str, reference: str, max_height: float = 125) -> None:
        self.image_count += 1
        number = str(self.image_count)
        if self.appendix_letter:
            self.appendix_figures += 1
            number = f"{self.appendix_letter}.{self.appendix_figures}"
        path, title, finding = self.figures.items[key]
        img = Image(str(path))
        scale = min(WIDTH / img.imageWidth, max_height * mm / img.imageHeight)
        img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
        caption = Paragraph(escape(f"Рисунок {number} — {title}"), self.style["caption"])
        self.story.append(
            KeepTogether(
                [
                    Paragraph(escape(reference.format(n=number)), self.style["body"]),
                    img,
                    caption,
                    Paragraph(escape(finding), self.style["body"]),
                ]
            )
        )

    def table(self, name: str, rows: list[list[Any]], widths: list[float], reference: str) -> None:
        self.table_count += 1
        number = str(self.table_count)
        if self.appendix_letter:
            self.appendix_tables += 1
            number = f"{self.appendix_letter}.{self.appendix_tables}"
        self.p(reference.format(n=number), "reference")
        caption = Paragraph(escape(f"Таблица {number} — {name}"), self.style["table_caption"])
        self.story.append(caption)
        cells = [[Paragraph(escape(str(v)), self.style["cell"]) for v in row] for row in rows]
        scale = min(mm, WIDTH / sum(widths))
        table = Table(cells, colWidths=[w * scale for w in widths], repeatRows=1, hAlign="CENTER")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eff3")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#8995a0")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        self.story.extend([table, Spacer(1, 8)])

    def save(self, path: Path) -> None:
        doc = ReportDocument(
            str(path),
            pagesize=A4,
            leftMargin=LEFT_MARGIN,
            rightMargin=RIGHT_MARGIN,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
            title="Нормальный и аномальный режим фрезерного станка",
            invariant=1,
        )
        doc.multiBuild(
            self.story,
            onFirstPage=footer,
            onLaterPages=footer,
        )


def parameters(params: dict[str, Any]) -> str:
    items = (f"{key.removeprefix('estimator__')}={value}" for key, value in params.items())
    return ", ".join(items) or "—"


def test_tables(report: Report, title: str, values: dict[str, Evaluation]) -> None:
    rows = [
        [
            "Модель",
            "Доля верных",
            "Сбал. доля",
            "Точность (precision)",
            "Полнота (recall)",
            "F1",
            "ROC-AUC",
            "PR-AUC",
            "AP",
        ]
    ]
    matrices = [["Модель", "good → good", "good → bad", "bad → good", "bad → bad"]]
    for name, value in values.items():
        rows.append([name, *[f"{metric:.3f}" for metric in value.metrics.values()]])
        matrices.append([name, *value.matrix.ravel()])
    report.table(
        f"{title}. Метрики",
        rows,
        [25, 16, 16, 20, 17, 13, 22, 22, 16],
        f"Метрики для выборки «{title}» приведены в таблице {{n}}.",
    )
    report.table(
        f"{title}. Матрицы ошибок, число файлов",
        matrices,
        [38, 32, 32, 32, 32],
        f"Число верных и ошибочных решений для выборки «{title}» приведено в таблице {{n}}.",
    )


def build_report(
    path: Path,
    figures: Figures,
    records: list[Recording],
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    choices: dict[str, Selection],
    winner: str,
    standard: dict[str, Evaluation],
    transfer: dict[str, Evaluation],
    scaling: dict[str, tuple[np.ndarray, np.ndarray]],
    time_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    time_choices: dict[str, Selection],
    early_test: dict[str, Evaluation],
    later: dict[str, Evaluation],
    machine_transfer: MachineTransfer,
    machine_test: Evaluation,
    machine_threshold: float,
    limited_review: dict[str, Evaluation],
    limited_thresholds: dict[str, float],
) -> None:
    configure_fonts()
    report = Report(figures)
    report.p("Санкт-Петербургский государственный университет", "center")
    report.story.append(Spacer(1, 64 * mm))
    report.p("Задание 2. Классификация", "center")
    report.story.append(Spacer(1, 8 * mm))
    report.p("Нормальный и аномальный режим фрезерного станка", "title")
    report.story.append(Spacer(1, 13 * mm))
    report.p("Клименко Георгий", "center")
    report.story.append(Spacer(1, 104 * mm))
    report.p("Санкт-Петербург", "center")
    report.p("2026", "center")

    report.section("РЕФЕРАТ")
    extent = ReportExtent(report.style["body"])
    report.story.append(extent)
    report.p(
        "ФРЕЗЕРОВАНИЕ, ВИБРАЦИЯ, УСКОРЕНИЕ, КЛАССИФИКАЦИЯ, " "ДИСБАЛАНС КЛАССОВ, ПЕРЕНОС МОДЕЛИ",
        "keywords",
    )
    report.p(
        f"На {len(records)} записях вибрации Bosch CNC Machining сопоставлены семь моделей "
        "для обнаружения аномалий фрезерования. Качество проверено на новых записях "
        "знакомых станков, на отложенном станке M03 и на данных августа 2021 года. "
        "Для ручного отбора записей оценён лимит просмотра 5 % партии. "
        "При переносе рабочие характеристики зависели от модели и правила отбора."
    )
    report.story.append(PageBreak())
    report.p("СОДЕРЖАНИЕ", "center")
    toc = TableOfContents(dotsMinLevel=0)
    toc.levelStyles = [report.style["toc"]]
    report.story.append(toc)

    train, test, moved = splits
    earlier, control, late = time_indices
    time_winner = max(time_choices, key=lambda name: time_choices[name].scores.mean())
    transfer_name = machine_transfer.winner
    _, transfer_fp, transfer_fn, transfer_tp = machine_test.matrix.ravel()

    report.section("ВВЕДЕНИЕ")
    report.p(
        "По вибрации станка можно обнаруживать отклонения при фрезеровании. "
        "Однако колебания зависят также от станка и операции. В работе проверяем, "
        "сохраняется ли качество распознавания при смене станка и со временем. "
        "Для ручной проверки считаем, сколько записей придётся просмотреть и "
        "сколько аномалий останется вне отбора."
    )

    report.section("1 Данные и схема проверки")
    bad_total = sum(record.label for record in records)
    report.p(
        f"Набор Bosch CNC Machining [1, 2] содержит {len(records)} записей с датчика "
        f"ускорения по трём осям, из них {bad_total} аномальных ({bad_total / len(records):.1%}). "
        "Одна запись соответствует одному файлу HDF5. Метки good и bad обозначают "
        "нормальный и аномальный процессы. Частота измерений 2000 Гц. Амплитуды показаны "
        "в единицах исходного файла."
    )
    report.figure(
        "signal",
        "Фрагменты ускорения для двух классов сопоставлены на рисунке {n}.",
        max_height=85,
    )
    rows = [["Часть", "Станки", "Файлы", "good", "bad", "Доля bad"]]
    for label, machines, indices in [
        ("Обучение", "M01, M02", train),
        ("Обычный тест", "M01, M02", test),
        ("Перенос", "M03", moved),
    ]:
        bad = sum(records[i].label for i in indices)
        rows.append(
            [label, machines, len(indices), len(indices) - bad, bad, f"{bad / len(indices):.1%}"]
        )
    report.table(
        "Разбиение для обычного теста и переноса на M03",
        rows,
        [40, 30, 23, 23, 23, 29],
        "Число файлов в каждой части приведено в таблице {n}.",
    )
    report.p(
        "Для обычного теста отложили 25 % записей M01 и M02 с сохранением доли классов. "
        "Те же модели и пороги проверили на всех записях M03. Для проверки переноса "
        "во времени отдельно выделили записи августа 2021 года и более ранний контрольный тест."
    )
    report.p(
        "Модели получают 24 признака ускорения из каждого файла. "
        "Расчёт признаков описан в приложении А."
    )

    report.section("2 Выбор модели и результаты тестов")
    report.p(
        "Сопоставили семь классификаторов. DummyClassifier всегда предсказывает "
        "нормальную работу. Для выбора параметров перебрали сетки на пяти стратифицированных "
        "разбиениях обучения. Критерий выбора — средняя AP, которая показывает, насколько "
        "высоко модель ставит аномалии в списке подозрительных записей. Порог выбирали "
        "по максимуму F1 на проверочных частях обучения. Настройки предобработки "
        "вычислялись внутри разбиений, включая калибровку SVM. Графики подбора "
        "приведены в приложении А, полные метрики и матрицы ошибок — в приложении Б."
    )
    summary = [
        [
            "Модель",
            "Выбранные параметры",
            "Средняя AP по 5 разбиениям",
            "AP тест",
            "F1 тест",
            "AP M03",
            "F1 M03",
        ]
    ]
    for name, choice in choices.items():
        src, dst = standard[name].metrics, transfer[name].metrics
        summary.append(
            [
                name,
                parameters(choice.params),
                f"{choice.scores.mean():.3f}",
                f"{src['AP']:.3f}",
                f"{src['F1']:.3f}",
                f"{dst['AP']:.3f}",
                f"{dst['F1']:.3f}",
            ]
        )
    report.table(
        "Параметры и результаты обычной проверки и переноса на M03",
        summary,
        [23, 52, 23, 18, 18, 18, 18],
        "Параметры моделей и результаты обоих тестов собраны в таблице {n}.",
    )
    report.figure(
        "transfer",
        "Сравнение обычного теста с переносом на M03 показано на рисунке {n}.",
        max_height=53,
    )

    report.section("3 Отбор записей на станке M03")
    max_reviews = int(len(moved) * TRANSFER_REVIEW_FRACTION)
    report.p(
        f"В эксперименте все {len(moved)} записей M03 образуют одну партию. "
        f"Лимит {TRANSFER_REVIEW_FRACTION:.0%} допускает {max_reviews} просмотров. "
        "В очередь попадают записи с наибольшими оценками. При совпадении оценок "
        "на границе отбора вся группа остаётся вне очереди. Для применения на станке "
        "размер партии предстоит определить, например по сменам."
    )
    rows = [["Модель и правило отбора", "Просмотры", "Найдено", "Пропущено", "Ложные тревоги"]]
    for name, evaluation in (
        (f"{winner}, порог из обучения", transfer[winner]),
        ("Лес, фиксированный порог", transfer[FOREST]),
        ("kNN, фиксированный порог", transfer["kNN"]),
        ("Лес, лимит 5 %", limited_review[FOREST]),
        ("Бустинг, лимит 5 %", limited_review[BOOST]),
    ):
        _, fp, fn, tp = evaluation.matrix.ravel()
        rows.append([name, tp + fp, tp, fn, fp])
    report.table(
        "Нагрузка ручной проверки и ошибки на M03",
        rows,
        [57, 25, 23, 27, 33],
        "Основные варианты отбора сопоставлены в таблице {n}.",
    )
    _, svm_fp, _, svm_tp = transfer[winner].matrix.ravel()
    normal_mask = np.array([records[i].label == 0 for i in moved])
    median = np.median(transfer[winner].scores[normal_mask])
    report.p(
        f"На M03 медианная оценка SVM для нормальных записей составила {median:.3f} "
        f"при пороге {choices[winner].threshold:.3f}. Сдвиг оценок проявился "
        "сразу в нескольких операциях и привёл к большому числу тревог. "
        "По данным этого эксперимента установить физическую причину сдвига нельзя."
    )
    report.p(
        "Полное сравнение при лимите, отдельный подбор по взаимному переносу M01 и M02 "
        "и пропуски по операциям приведены в приложении В."
    )

    report.section("4 Перенос во времени")
    late_by_machine = Counter(records[i].machine for i in late if records[i].label)
    late_bad = sum(late_by_machine.values())
    report.p(
        f"Из ранних записей M01 и M02 для обучения взяли {len(earlier)}, для контроля — "
        f"{len(control)}. Августовский тест содержит {len(late)} записей. "
        "Для обеих проверок использованы одни и те же обученные модели и пороги."
    )
    report.figure(
        "time_transfer",
        "AP на раннем и августовском тестах сопоставлены на рисунке {n}.",
        max_height=85,
    )
    _, time_fp, _, time_tp = later[time_winner].matrix.ravel()
    _, forest_time_fp, _, forest_time_tp = later[FOREST].matrix.ravel()
    report.p(
        f"По ранним разбиениям выбрали модель «{time_winner}». В августе она обнаружила "
        f"{time_tp} из {late_bad} аномалий, ложных тревог — {time_fp}. Лес обнаружил "
        f"{forest_time_tp} из {late_bad}, ложных тревог — {forest_time_fp}. "
        f"На M01 приходится {late_by_machine['M01']} аномалий, на M02 — "
        f"{late_by_machine['M02']}. Сравнение стоит повторить на следующем периоде "
        "с результатами отдельно по каждому станку. Полная сводка дана в приложении В."
    )

    report.section("ЗАКЛЮЧЕНИЕ")
    ordinary_ap = choices[winner].scores.mean()
    forest_ap = choices[FOREST].scores.mean()
    report.p(
        f"Для знакомых станков выбрана {winner}. Она получила наибольшую среднюю AP "
        f"на пяти разбиениях обучающих записей M01 и M02 ({ordinary_ap:.3f} против "
        f"{forest_ap:.3f} у случайного леса). На обычном тесте её AP составила "
        f"{standard[winner].metrics['AP']:.3f}. На отложенном M03 AP снизилась до "
        f"{transfer[winner].metrics['AP']:.3f}. Из {svm_tp + svm_fp} тревог "
        f"обоснованными оказались {svm_tp} ({transfer[winner].metrics['Precision']:.1%}). "
        "Поэтому для ручного отбора на новом станке прежний порог SVM непрактичен."
    )
    _, forest_fp, forest_fn, forest_tp = limited_review[FOREST].matrix.ravel()
    _, boost_fp, _, boost_tp = limited_review[BOOST].matrix.ravel()
    _, fixed_fp, _, fixed_tp = transfer[FOREST].matrix.ravel()
    report.p(
        "При жёстком лимите 5 % для следующей проверки предлагаем случайный лес "
        f"с порогом по оценкам каждой партии. На M03 он нашёл {forest_tp} из "
        f"{forest_tp + forest_fn} аномалий за {forest_tp + forest_fp} просмотра. "
        f"Бустинг нашёл {boost_tp} аномалий за {boost_tp + boost_fp} просмотров. "
        f"Фиксированный порог леса {choices[FOREST].threshold:.3f} позволил найти "
        f"{fixed_tp} аномалий, но потребовал {fixed_tp + fixed_fp} просмотров при "
        f"лимите {max_reviews}. Рекомендация основана на сравнении на M03 и требует "
        "проверки на другом станке."
    )

    report.section("СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ")
    report.p(
        "1 Тнани М.-А., Файль М., Дипольд К. Система сбора данных для фрезерных "
        "станков с ЧПУ. Procedia CIRP. 2022. Т. 107. С. 131–136. "
        "DOI 10.1016/j.procir.2022.04.022."
    )
    report.p(
        "2 Bosch CNC Machining Dataset. UCI Machine Learning Repository. 2022. "
        "URL https://archive.ics.uci.edu/dataset/752/bosch+cnc+machining+dataset. "
        "Файлы и описание: https://github.com/boschresearch/CNC_Machining. CC BY 4.0."
    )
    report.p(
        "3 SciPy. Документация метода Уэлча. "
        "URL https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html."
    )
    report.p(
        "4 Scikit-learn. Документация HistGradientBoostingClassifier и метрики AP. "
        "URL https://scikit-learn.org/stable/modules/ensemble.html."
    )
    report.p(
        "5 ГОСТ 7.32-2017. Отчёт о научно-исследовательской работе. Структура и правила "
        "оформления. Москва, Стандартинформ, 2017."
    )
    report.appendix("А", "Подбор и диагностика моделей")
    report.p(
        "На каждой оси рассчитаны среднеквадратическое значение, стандартное отклонение, "
        "размах, эксцесс и энергии в четырёх полосах. Эксцесс — четвёртый "
        "стандартизованный момент за вычетом трёх. "
        "Энергии вычислены по методу Уэлча [3] в полосах 0–100, 100–300, 300–600 "
        "и 600–1000 Гц. Использовано окно Ханна до 1024 отсчётов с перекрытием 50 % "
        "и удалением среднего в каждом сегменте. Пропуски заполнялись медианой "
        "обучающей части. Для логистической регрессии, kNN и SVM применялась стандартизация."
    )
    report.p(
        "Для анализа бустинга использован HistGradientBoostingClassifier из "
        "scikit-learn [4]. Он позволяет ограничить сложность дерева и оценить "
        "качество после каждой итерации. Для SVM применялась сигмоидальная калибровка."
    )
    rows = [["Модель", "Сетка параметров", "Выбранные параметры", "AP, среднее ± СКО", "Порог"]]
    for name, candidate in candidates().items():
        choice = choices[name]
        rows.append(
            [
                name,
                parameters(candidate.grid),
                parameters(choice.params),
                f"{choice.scores.mean():.3f} ± {choice.scores.std():.3f}",
                f"{choice.threshold:.4g}",
            ]
        )
    report.table(
        "Подбор параметров на обучающих файлах",
        rows,
        [25, 51, 50, 27, 17],
        "Сетки поиска и выбранные параметры приведены в таблице {n}.",
    )
    for key, reference in (
        ("cv", "Средняя AP и разброс по разбиениям показаны на рисунке {n}."),
        ("grid", "Результат сеточного поиска SVM показан на рисунке {n}."),
        ("threshold", "Зависимость метрик от порога показана на рисунке {n}."),
    ):
        report.figure(key, reference, max_height=67)
    report.p(
        "Для дерева по очереди менялись максимальная глубина и минимальное число "
        "записей в листе. Ошибка равна единице минус сбалансированная доля верных ответов. "
        "Для ансамблей использована ошибка 1 − AP."
    )
    for key, reference, height in (
        ("tree", "Ошибки обучения и проверки дерева сопоставлены на рисунке {n}.", 64),
        ("forest_count", "Влияние числа деревьев показано на рисунке {n}.", 64),
        ("forest_parameters", "Влияние параметров леса показано на рисунке {n}.", 64),
        ("boost_iterations", "Ошибка бустинга по итерациям показана на рисунке {n}.", 51),
        ("boost_parameters", "Влияние параметров бустинга показано на рисунке {n}.", 51),
    ):
        report.figure(key, reference, max_height=height)
    scale_notes = [
        f"{name} — {raw.mean():.4f} без неё и {scaled.mean():.4f} с ней"
        for name, (raw, scaled) in scaling.items()
    ]
    report.p("Средняя AP при проверке стандартизации (" + ", ".join(scale_notes) + ").")

    report.appendix("Б", "Метрики, матрицы ошибок и важность признаков")
    report.p(
        "Точность показывает долю обоснованных тревог, полнота — долю найденных "
        "аномалий. F1 учитывает обе величины. PR-AUC рассчитана методом трапеций. "
        "Для Dummy оценки постоянны, поэтому AP совпадает с долей аномалий, а "
        "трапецеидальная PR-AUC выше из-за отрезка до точки нулевой полноты."
    )
    report.figure(
        "roc_pr", "ROC и кривые точность–полнота приведены на рисунке {n}.", max_height=90
    )
    report.figure(
        "matrix", "Нормированные матрицы ошибок SVM приведены на рисунке {n}.", max_height=66
    )
    test_tables(report, "Обычный тест", standard)
    test_tables(report, "Перенос на M03", transfer)
    test_tables(report, "Август 2021 года", later)
    report.figure(
        "importance", "Перестановочная важность для SVM показана на рисунке {n}.", max_height=52
    )

    report.appendix("В", "Дополнительные результаты переноса")
    report.p(
        "При отдельном подборе модели для ручного просмотра использованы только "
        "обучающие записи M01 и M02. Каждый из станков по очереди откладывали. Параметры "
        "выбирали по средней AP, модели — по средней полноте при лимите просмотра 5 %. "
        "При равной полноте решение принимали по AP."
    )
    rows = [["Модель", "Средняя AP", "Полнота M01", "Полнота M02", "Средняя полнота"]]
    for name, checks in machine_transfer.validation.items():
        first, second = (checks[machine].metrics["Recall"] for machine in ("M01", "M02"))
        rows.append(
            [
                name,
                f"{machine_transfer.choices[name].scores.mean():.3f}",
                f"{first:.1%}",
                f"{second:.1%}",
                f"{(first + second) / 2:.1%}",
            ]
        )
    report.table(
        "Выбор модели для ручной проверки на M01 и M02",
        rows,
        [36, 30, 34, 34, 36],
        "Результаты взаимной проверки станков приведены в таблице {n}.",
    )
    report.p(
        f"В этой проверке выбрана модель «{transfer_name}» "
        f"({parameters(machine_transfer.choices[transfer_name].params)}). На M03 "
        f"она нашла {transfer_tp} из {transfer_tp + transfer_fn} аномалий и дала "
        f"{transfer_fp} ложных тревог."
    )
    rows = [
        [
            "Способ отбора",
            "Порог, окр.",
            "На просмотр",
            "Найдены",
            "Пропуски",
            "Ложные тревоги",
            "F1",
        ]
    ]
    for name, threshold, result, limited in (
        (f"{winner}, прежний порог", choices[winner].threshold, transfer[winner], False),
        *(
            (
                f"{name}, лимит 5 %",
                limited_thresholds[name],
                result,
                True,
            )
            for name, result in limited_review.items()
        ),
        (f"{transfer_name} по станкам, лимит 5 %", machine_threshold, machine_test, True),
        ("Случайный лес, фикс. порог", choices[FOREST].threshold, transfer[FOREST], False),
        ("Ближайшие соседи", choices["kNN"].threshold, transfer["kNN"], False),
    ):
        _, fp, fn, tp = result.matrix.ravel()
        cut = "—" if limited and tp + fp == 0 else f"{threshold:.4g}"
        if limited:
            cut += "*"
        rows.append([name, cut, tp + fp, tp, fn, fp, f"{result.metrics['F1']:.3f}"])
    report.table(
        "Нагрузка на ручную проверку и ошибки на M03",
        rows,
        [44, 20, 22, 19, 20, 24, 21],
        "Способы отбора на M03 сопоставлены в таблице {n}.",
    )
    report.p(
        "* Округлённые пороги для партии M03. При лимите 5 % порог рассчитывается заново "
        "по оценкам каждой партии. Для воспроизведения результатов M03 используйте "
        "точные значения из столбца threshold файла build/metrics.csv в строках "
        "transfer_review_models и transfer_review. Если оценки совпадают на границе, "
        "модель может использовать лишь часть лимита. Прочерк означает, что отбор пуст."
    )
    anomaly_indices = [i for i in moved if records[i].label == 1]
    misses = {
        name: {
            i
            for i, predicted in zip(moved, transfer[name].prediction)
            if records[i].label == 1 and predicted == 0
        }
        for name in (FOREST, "kNN")
    }
    operations = sorted({records[i].operation for i in misses[FOREST] | misses["kNN"]})
    if operations:
        rows = [["Операция", "Аномалии", "Пропуски леса", "Пропуски kNN"]]
        for operation in operations:
            files = {i for i in anomaly_indices if records[i].operation == operation}
            rows.append(
                [
                    operation,
                    len(files),
                    len(files & misses[FOREST]),
                    len(files & misses["kNN"]),
                ]
            )
        report.table(
            "Пропуски случайного леса и kNN по операциям на M03",
            rows,
            [38, 38, 47, 47],
            "Операции с пропущенными аномалиями приведены в таблице {n}.",
        )
    time_rows = [
        [
            "Модель",
            "Параметры",
            "Порог",
            "Средняя AP по 5 разбиениям",
            "AP ранняя",
            "F1 ранний",
            "AP август",
            "F1 август",
            "AP август − ранняя",
        ]
    ]
    for name, choice in time_choices.items():
        metrics = later[name].metrics
        early_metrics = early_test[name].metrics
        delta = metrics["AP"] - early_metrics["AP"]
        time_rows.append(
            [
                name,
                parameters(choice.params),
                f"{choice.threshold:.3f}",
                f"{choice.scores.mean():.3f}",
                f"{early_metrics['AP']:.3f}",
                f"{early_metrics['F1']:.3f}",
                f"{metrics['AP']:.3f}",
                f"{metrics['F1']:.3f}",
                f"{delta:+.3f}",
            ]
        )
    report.table(
        "Сравнение раннего и августовского тестов при одном обучении",
        time_rows,
        [18, 36, 14, 21, 14, 16, 14, 14, 18],
        "Последний столбец таблицы {n} показывает разность AP двух тестов.",
    )
    extent.counts = (
        f"1 кн., {report.image_count} рис., {report.table_count} табл., 5 источн., "
        f"{report.appendix_count} прил"
    )
    report.save(path)
