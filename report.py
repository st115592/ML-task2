"""Сборка отчёта A4 со сквозной нумерацией и оглавлением."""

import os
from collections import Counter
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
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
from reportlab.platypus.tableofcontents import TableOfContents

from cnc_data import Recording
from experiment import FOREST, TREE, Evaluation, Selection, candidates
from figures import Figures

WIDTH = 170 * mm


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
        "center": ParagraphStyle("Center", parent=body, alignment=TA_CENTER, firstLineIndent=0),
        "caption": ParagraphStyle(
            "Caption",
            parent=body,
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            firstLineIndent=0,
            spaceAfter=6,
        ),
        "cell": ParagraphStyle(
            "Cell", parent=body, fontSize=9.5, leading=12, firstLineIndent=0, spaceAfter=0
        ),
        "toc": ParagraphStyle("Toc", parent=body, firstLineIndent=0, leading=24),
    }


class ReportDocument(SimpleDocTemplate):
    def afterFlowable(self, flowable: Any) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name == "Section":
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))


def footer(canvas: Canvas, doc: ReportDocument) -> None:
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

    def p(self, value: str, style: str = "body") -> None:
        self.story.append(Paragraph(escape(value), self.style[style]))

    def section(self, title: str) -> None:
        self.story.append(PageBreak())
        self.p(title, "section")

    def figure(self, key: str, max_height: float = 125) -> None:
        self.image_count += 1
        path, title, finding = self.figures.items[key]
        img = Image(str(path))
        scale = min(WIDTH / img.imageWidth, max_height * mm / img.imageHeight)
        img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
        caption = Paragraph(escape(f"Рисунок {self.image_count} — {title}"), self.style["caption"])
        self.story.append(
            KeepTogether([img, caption, Paragraph(escape(finding), self.style["body"])])
        )

    def table(self, name: str, rows: list[list[Any]], widths: list[float]) -> None:
        self.table_count += 1
        caption = Paragraph(escape(f"Таблица {self.table_count} — {name}"), self.style["caption"])
        caption.keepWithNext = True
        self.story.append(caption)
        cells = [[Paragraph(escape(str(v)), self.style["cell"]) for v in row] for row in rows]
        table = Table(cells, colWidths=[w * mm for w in widths], repeatRows=1, hAlign="CENTER")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eff3")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#8995a0")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
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
            leftMargin=30 * mm,
            rightMargin=10 * mm,
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
            "Точность",
            "Полнота",
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
    report.table(f"{title}. Метрики", rows, [27, 17, 17, 17, 16, 14, 22, 22, 16])
    report.table(f"{title}. Матрицы ошибок, число файлов", matrices, [38, 32, 32, 32, 32])


def errors_by_operation(
    records: list[Recording], indices: np.ndarray, result: Evaluation
) -> list[list[Any]]:
    counts: dict[str, Counter[str]] = {}
    for index, prediction in zip(indices, result.prediction):
        row = records[index]
        key = f"{row.machine} / {row.operation}"
        counts.setdefault(key, Counter())[f"{row.label}{prediction}"] += 1
    ranked = sorted(
        counts.items(),
        key=lambda item: (
            -item[1]["10"],
            -item[1]["01"],
            item[0],
        ),
    )
    return [
        [key, c["00"] + c["01"], c["01"], c["10"] + c["11"], c["10"]]
        for key, c in ranked
        if c["01"] or c["10"]
    ][:6]


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
) -> None:
    configure_fonts()
    report = Report(figures)
    report.p("ОТЧЁТ ПО УЧЕБНО-ИССЛЕДОВАТЕЛЬСКОЙ РАБОТЕ", "center")
    report.p("по курсу «ML in mechanics»", "center")
    report.story.append(Spacer(1, 60 * mm))
    report.p("НОРМАЛЬНЫЙ И АНОМАЛЬНЫЙ РЕЖИМ ФРЕЗЕРНОГО СТАНКА", "center")
    report.p("Задание 2. Классификация", "center")
    report.story.append(Spacer(1, 105 * mm))
    report.p("2026", "center")

    report.section("РЕФЕРАТ")
    train, test, moved = splits
    result = standard[winner].metrics
    target = transfer[winner].metrics
    report.p(
        f"Исследованы {len(records)} записей трёхкомпонентного ускорения из набора Bosch. "
        "Цель работы — распознать аномальный процесс фрезерования по вибрации и проверить "
        "сохранение качества на другом станке. Из каждой записи извлечены 24 признака. "
        "Сопоставлены семь классификаторов. Выбор по средней точности класса bad на "
        f"перекрёстной проверке дал модель «{winner}». На обычном тесте AP={result['AP']:.3f} "
        f"и F1={result['F1']:.3f}. При переносе AP={target['AP']:.3f} и F1={target['F1']:.3f}."
    )
    report.p(
        "КЛЮЧЕВЫЕ СЛОВА: ФРЕЗЕРОВАНИЕ, ВИБРАЦИЯ, УСКОРЕНИЕ, КЛАССИФИКАЦИЯ, "
        "ДИСБАЛАНС КЛАССОВ, ПЕРЕНОС МОДЕЛИ."
    )

    report.story.append(PageBreak())
    report.p("СОДЕРЖАНИЕ", "center")
    toc = TableOfContents()
    toc.levelStyles = [report.style["toc"]]
    report.story.append(toc)

    report.section("ВВЕДЕНИЕ")
    report.p(
        "При работе фрезерного станка датчик ускорения регистрирует вибрации инструмента "
        "и конструкции. Аномальное состояние может менять амплитуду и распределение "
        "энергии по частотам. Похожий эффект вызывают штатная смена операции, "
        "скорости резания и особенности станка. Поэтому полезность классификатора "
        "оценивается и на знакомых станках, и на отдельном станке."
    )
    report.p(
        "Входом модели служит одна законченная запись, выходом — оценка принадлежности "
        "к классу bad. Цель — определить, какие алгоритмы лучше выделяют редкие аномалии, "
        "и измерить изменение качества при переносе."
    )

    report.section("1 Данные и план эксперимента")
    report.p(
        "Набор Bosch CNC Machining [1, 2] содержит записи датчика Bosch CISS по осям X, Y "
        "и Z с частотой 2000 Гц. Каталог good обозначает нормальный процесс, каталог bad — "
        "аномальный. За объект взят целый HDF5-файл. Единица исходных отсчётов в описании "
        "формата не указана, поэтому амплитуда приведена в единицах файла."
    )
    report.figure("classes", max_height=55)
    report.figure("signal", max_height=99)
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
    report.table("Разбиение исходных файлов", rows, [40, 30, 23, 23, 23, 29])
    report.p(
        "Для обычного теста отложены 25 % файлов M01 и M02 со стратификацией по классу. "
        "M03 целиком отложен для переноса, как в проверке по станкам, предложенной "
        "авторами [1]. Настройки, включая пороги, одинаковы на обеих проверках. "
        "Обычная оценка допускает совпадение станков и операций между обучением и тестом. "
        "Оценка переноса отражает совокупную смену станка и состава режимов."
    )
    report.p(
        "Для каждой оси вычислены среднеквадратическое значение (СКЗ), стандартное "
        "отклонение, размах и эксцесс. Последний равен четвёртому стандартизованному "
        "моменту за вычетом трёх. Четыре широкополосные энергии на ось получены "
        "суммированием односторонней спектральной плотности мощности в полосах "
        "0–100, 100–300, 300–600 и 600–1000 Гц с умножением на шаг частоты. "
        "Использован метод Уэлча [3] с окном Ханна до 1024 отсчётов, перекрытием 50 % "
        "и удалением среднего в каждом сегменте. Энергия выражена в квадрате исходной "
        "единицы ускорения."
    )
    report.p(
        "Расчёт признаков выполняется внутри конвейера для каждого файла независимо. "
        "Пропуски заполняются медианой обучающей части текущего разбиения. Для "
        "логистической регрессии, kNN и SVM применяется стандартизация по средним "
        "и стандартным отклонениям этой же части. Идентификаторы служат для разбиения "
        "и анализа ошибок. В модель поступают только статистики ускорения."
    )

    report.section("2 Модели, параметры и рабочий порог")
    report.p(
        "Сравниваются предсказание наиболее частого класса (DummyClassifier), "
        "логистическая регрессия, метод ближайших соседей с весами по расстоянию, "
        "SVM с радиальным ядром и сигмоидальной калибровкой, "
        "решающее дерево, случайный лес и гистограммный "
        "градиентный бустинг HistGradientBoostingClassifier. Для бустинга выбрана "
        "реализация scikit-learn [4], поскольку она позволяет ограничить листья, "
        "проверить каждую итерацию по вероятности и обойтись единым набором зависимостей."
    )
    report.p(
        "Основная метрика — средняя точность AP для класса bad. Она усредняет точность "
        "по приращениям полноты и показывает качество ранжирования редких аномалий. "
        "PR-AUC в таблицах рассчитана отдельно трапециями. GridSearchCV сравнивает "
        "полные сочетания параметров на пяти стратифицированных разбиениях обучающей "
        "выборки. Начальное состояние генератора равно 42. Окончательная модель "
        "выбрана по максимуму среднего AP."
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
    report.table("Подбор параметров на обучающих файлах", rows, [25, 51, 50, 27, 17])
    report.figure("cv")
    report.figure("grid")
    report.p(
        "Рабочий порог выбирается отдельно для каждой обучаемой модели по максимуму F1 "
        "на вневыборочных вероятностях обучающей части. Такой критерий учитывает "
        "пропущенные аномалии и ложные тревоги через полноту и точность. Dummy всегда "
        "предсказывает наиболее частый класс. Подбор параметров и порога использует "
        "одни и те же разбиения, поэтому оценку обобщения дают отложенные выборки."
    )
    report.figure("threshold")

    report.section("3 Эксперименты с деревьями")
    report.p(
        "Для дерева по очереди изменены максимальная глубина и минимальное число "
        "объектов в листе. Для первой кривой разрешён лист из одной записи. "
        "Для второй глубина не ограничена. Ошибка равна единице "
        "минус сбалансированная доля верных ответов. Для ансамблей использована "
        "ранговая ошибка 1 − AP."
    )
    report.figure("tree")
    for name in (TREE, FOREST):
        raw, scaled = scaling[name]
        report.p(
            f"Для модели «{name}» средняя AP без стандартизации {raw.mean():.4f}, "
            f"со стандартизацией {scaled.mean():.4f}. "
            f"Изменение {scaled.mean() - raw.mean():+.4f}. "
            "В итоговой модели использованы исходные масштабы признаков."
        )
    max_change = max(abs(scaled.mean() - raw.mean()) for raw, scaled in scaling.values())
    if max_change < 0.00005:
        report.p(
            "В этом эксперименте стандартизация дерева и леса не изменила среднюю AP "
            "с точностью до четырёх знаков после запятой."
        )
    report.figure("forest_count")
    report.figure("forest_parameters")
    report.figure("boost_iterations")
    report.figure("boost_parameters")

    report.section("4 Итоговая проверка и перенос")
    report.p(
        "После фиксации признаков, моделей, гиперпараметров и порогов рассчитаны метрики "
        "на обычном тесте и на M03. ROC и кривые точности–полноты построены по "
        "вероятностям класса bad. Точность, полнота и F1 относятся к классу bad."
    )
    report.p(
        "У Dummy одна оценка для всех файлов. Его AP равна доле bad. "
        "PR-AUC по трапециям включает линейный отрезок до условной точки "
        "с нулевой полнотой, поэтому для постоянной оценки она выше AP."
    )
    test_tables(report, "Обычный тест", standard)
    test_tables(report, "Перенос на M03", transfer)
    report.figure("roc_pr")
    report.figure("matrix")
    report.figure("transfer")
    deltas = [
        ["Модель", "AP проверка", "AP тест", "AP M03", "Изм. AP", "F1 тест", "F1 M03", "Изм. F1"]
    ]
    for name in choices:
        src, dst = standard[name].metrics, transfer[name].metrics
        deltas.append(
            [
                name,
                f"{choices[name].scores.mean():.3f}",
                f"{src['AP']:.3f}",
                f"{dst['AP']:.3f}",
                f"{dst['AP'] - src['AP']:+.3f}",
                f"{src['F1']:.3f}",
                f"{dst['F1']:.3f}",
                f"{dst['F1'] - src['F1']:+.3f}",
            ]
        )
    report.table(
        "Сводка перекрёстной проверки, теста и переноса", deltas, [32, 24, 20, 19, 19, 19, 18, 19]
    )
    report.p(
        f"Доли аномалий составляют {sum(records[i].label for i in test) / len(test):.1%} "
        f"в обычном тесте и {sum(records[i].label for i in moved) / len(moved):.1%} "
        "на M03. Эти доли задают разные уровни AP случайного ранжирования. При "
        "сопоставлении AP и F1 также следует учитывать малое число аномальных файлов."
    )

    report.section("5 Интерпретация и инженерный разбор")
    report.figure("importance")
    report.p(
        "Перестановочная важность рассчитана по падению AP после перемешивания одного "
        "признака в проверочной части каждого обучающего разбиения. Отрезки показывают "
        "стандартное отклонение по пяти разбиениям и 12 повторениям. Связанные "
        "спектральные признаки могут частично заменять друг друга."
    )
    for title, indices, evaluation in [
        ("Обычный тест", test, standard[winner]),
        ("Перенос на M03", moved, transfer[winner]),
    ]:
        tn, fp, fn, tp = evaluation.matrix.ravel()
        report.p(
            f"{title}. Модель «{winner}» пропустила {fn} из {fn + tp} аномалий, а "
            f"{fp} из {tn + fp} нормальных записей вызвали ложную тревогу."
        )
        labels = np.array([records[i].label for i in indices])
        good = np.median(evaluation.scores[labels == 0])
        bad = np.median(evaluation.scores[labels == 1])
        report.p(
            f"Медиана оценки bad среди нормальных файлов равна {good:.3f}, "
            f"среди аномальных — {bad:.3f}. Рабочий порог равен "
            f"{choices[winner].threshold:.3f}."
        )
        rows = errors_by_operation(records, indices, evaluation)
        if rows:
            report.table(
                f"{title}. Операции с ошибками модели «{winner}»",
                [["Станок / операция", "good", "Ложные тревоги", "bad", "Пропуски"], *rows],
                [48, 24, 36, 24, 36],
            )
            key, _, false_pos, _, false_neg = rows[0]
            report.p(
                f"Для {key} число пропусков равно {false_neg}, "
                f"число ложных тревог равно {false_pos}. "
                "Пропуск означает, что статистики аномальной записи оказались похожи "
                "на нормальный режим по оценке модели. Ложная тревога означает, что "
                "штатная запись пересекла выбранный порог."
            )
    report.p(
        "Смена операции или станка меняет фон вибрации и отклик конструкции. Это "
        "возможное объяснение путаницы классов, поскольку модель обучена "
        "на записях других станков. Метки набора не указывают конкретный механизм "
        "неисправности. Сравнение характеризует данный отложенный станок."
    )

    report.section("ЗАКЛЮЧЕНИЕ")
    report.p(
        f"По средней AP на обучающих разбиениях выбрана модель «{winner}» с рабочим "
        f"порогом {choices[winner].threshold:.4g}. На файлах знакомых станков получены "
        f"AP={result['AP']:.3f} и F1={result['F1']:.3f}. На M03 AP={target['AP']:.3f}, "
        f"F1={target['F1']:.3f}. Изменение составляет {target['AP'] - result['AP']:+.3f} "
        f"по AP и {target['F1'] - result['F1']:+.3f} по F1."
    )
    winner_moved = max(transfer, key=lambda name: transfer[name].metrics["AP"])
    best_f1 = max(transfer, key=lambda name: transfer[name].metrics["F1"])
    report.p(
        f"При фиксированных порогах на M03 наибольший F1 у модели «{best_f1}» "
        f"({transfer[best_f1].metrics['F1']:.3f})."
    )
    report.p(
        f"Лучшая AP при переносе у модели «{winner_moved}» "
        f"({transfer[winner_moved].metrics['AP']:.3f}). "
        "Для оценки работы системы в новом производственном режиме полезно "
        "учитывать одновременно изменение ранжирования и ошибки при фиксированном пороге."
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
    report.save(path)
