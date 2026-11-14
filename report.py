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
        if isinstance(flowable, Paragraph) and flowable.style.name == "Section":
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

    def p(self, value: str, style: str = "body") -> None:
        self.story.append(Paragraph(escape(value), self.style[style]))

    def section(self, title: str) -> None:
        self.story.append(PageBreak())
        self.p(title, "section")

    def figure(self, key: str, reference: str, max_height: float = 125) -> None:
        self.image_count += 1
        path, title, finding = self.figures.items[key]
        img = Image(str(path))
        scale = min(WIDTH / img.imageWidth, max_height * mm / img.imageHeight)
        img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
        caption = Paragraph(escape(f"Рисунок {self.image_count} — {title}"), self.style["caption"])
        self.story.append(
            KeepTogether(
                [
                    Paragraph(escape(reference.format(n=self.image_count)), self.style["body"]),
                    img,
                    caption,
                    Paragraph(escape(finding), self.style["body"]),
                ]
            )
        )

    def table(self, name: str, rows: list[list[Any]], widths: list[float], reference: str) -> None:
        self.table_count += 1
        self.p(reference.format(n=self.table_count), "reference")
        caption = Paragraph(
            escape(f"Таблица {self.table_count} — {name}"), self.style["table_caption"]
        )
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
    time_indices: tuple[np.ndarray, np.ndarray],
    time_choices: dict[str, Selection],
    early_test: dict[str, Evaluation],
    later: dict[str, Evaluation],
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
    report.p("2026", "center")

    report.section("РЕФЕРАТ")
    extent = ReportExtent(report.style["body"])
    report.story.append(extent)
    report.p(
        "ФРЕЗЕРОВАНИЕ, ВИБРАЦИЯ, УСКОРЕНИЕ, КЛАССИФИКАЦИЯ, " "ДИСБАЛАНС КЛАССОВ, ПЕРЕНОС МОДЕЛИ",
        "keywords",
    )
    train, test, moved = splits
    result = standard[winner].metrics
    target = transfer[winner].metrics
    earlier, control, late = time_indices
    time_winner = max(time_choices, key=lambda name: time_choices[name].scores.mean())
    report.p(
        f"Исследованы {len(records)} записей трёхкомпонентного ускорения из набора Bosch. "
        "Цель работы — распознать аномальный процесс фрезерования по вибрации и проверить "
        "сохранение качества на другом станке и в более позднем периоде. "
        "Из каждой записи извлечены 24 признака. "
        "Сопоставлены семь классификаторов. Выбор по средней точности класса bad на "
        f"перекрёстной проверке дал модель «{winner}». На обычном тесте AP={result['AP']:.3f} "
        f"и F1={result['F1']:.3f}. На M03 AP={target['AP']:.3f} и F1={target['F1']:.3f}. "
        f"Для проверки переноса во времени модель «{time_winner}» обучили на более "
        "ранних записях. На данных августа 2021 года модель получила "
        f"AP {later[time_winner].metrics['AP']:.3f}."
    )
    report.story.append(PageBreak())
    report.p("СОДЕРЖАНИЕ", "center")
    toc = TableOfContents(dotsMinLevel=0)
    toc.levelStyles = [report.style["toc"]]
    report.story.append(toc)

    report.section("ВВЕДЕНИЕ")
    report.p(
        "Во время обработки детали фреза, шпиндель и корпус станка вибрируют. "
        "Установленный на станке датчик записывает, как меняется ускорение по трём осям. "
        "Если обработка идёт аномально, могут появиться более сильные колебания, "
        "отдельные всплески или новые частоты. При этом и исправный станок может "
        "вибрировать по-разному при смене инструмента или операции. По одному "
        "высокому пику поэтому трудно решить, действительно ли произошла аномалия."
    )
    report.p(
        "Задача состоит в том, чтобы по целой записи вибрации отнести один цикл обработки "
        "к нормальному (good) или аномальному (bad). Модель получает примеры обоих режимов "
        "и учится выставлять каждой записи оценку аномальности. При превышении рабочего "
        "порога запись помечается для проверки. Пропуск оставляет аномальную обработку "
        "без внимания, а ложная тревога требует проверки исправного процесса."
    )
    report.p(
        "В работе проверяются два вопроса. Во-первых, какие методы лучше распознают "
        "редкие аномалии в новых записях со знакомых станков. Во-вторых, сохраняется ли "
        "это качество на новом станке и в более позднем периоде. "
        "Для этого сравниваются семь методов и измеряются как способность ставить "
        "аномалии выше в списке подозрительных записей, так и ошибки при выбранном пороге."
    )

    report.section("1 Данные и план эксперимента")
    report.p(
        "Набор Bosch CNC Machining [1, 2] содержит записи датчика Bosch CISS по осям X, Y "
        "и Z с частотой 2000 Гц. Каталог good обозначает нормальный процесс, каталог bad — "
        "аномальный. За объект взят целый HDF5-файл. Единица исходных отсчётов в описании "
        "формата не указана, поэтому амплитуда приведена в единицах файла."
    )
    report.figure(
        "classes", "Распределение записей по классам показано на рисунке {n}.", max_height=55
    )
    report.figure(
        "signal",
        "Фрагменты ускорения для двух классов сопоставлены на рисунке {n}.",
        max_height=99,
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
        "Обычный тест отвечает на вопрос, как модель работает на новых записях "
        "с уже знакомых станков. Для него отложены 25 % файлов M01 и M02 "
        "с сохранением пропорции классов. Проверка переноса отвечает на другой вопрос: "
        "можно ли применить ту же модель на M03. Все его записи отложены целиком "
        "по схеме проверки по станкам, предложенной авторами набора [1]. В обоих "
        "тестах использованы одни и те же модели и пороги. Результат на M03 отражает "
        "смену и станка, и состава операций. Дополнительно проверен перенос во времени "
        "на тех же двух станках M01 и M02. Для него модели, параметры и пороги "
        "заново подбирались только по записям до августа 2021 года."
    )
    known_operations = {records[i].operation for i in train}
    known_periods = {records[i].period for i in train}
    moved_operations = {records[i].operation for i in moved}
    moved_periods = {records[i].period for i in moved}
    report.p(
        f"Из {len(moved_operations)} операций M03 в обучении встречаются "
        f"{len(moved_operations & known_operations)}, из {len(moved_periods)} периодов — "
        f"{len(moved_periods & known_periods)}. Поэтому перенос на M03 и проверка "
        "на целиком отложенном августе отвечают на разные вопросы."
    )
    report.p(
        "Одна запись может содержать десятки тысяч отсчётов. Для сравнения файлов "
        "одинаковым способом каждую ось описывают восемь чисел. Среднеквадратическое "
        "значение (СКЗ) характеризует общий уровень ускорения, стандартное отклонение — "
        "колебания вокруг среднего, размах — разницу между наибольшим и наименьшим "
        "отсчётами. Эксцесс помогает заметить редкие сильные отклонения. Ещё четыре "
        "числа показывают, сколько энергии колебаний приходится на разные частоты."
    )
    report.p(
        "Для воспроизведения расчёта эксцесс определён как четвёртый стандартизованный "
        "момент за вычетом трёх. Энергии рассчитаны по односторонней спектральной "
        "плотности в полосах 0–100, 100–300, 300–600 и 600–1000 Гц. Использован метод "
        "Уэлча [3] с окном Ханна до 1024 отсчётов, перекрытием 50 % и удалением среднего "
        "в каждом сегменте. Сумма спектральной плотности в полосе умножается на шаг "
        "частоты. Энергия выражена в квадрате исходной единицы ускорения."
    )
    report.p(
        "Все 24 признака рассчитываются отдельно для каждого файла. Номер станка, "
        "операция и период нужны для разбиения и разбора ошибок. Вход модели состоит "
        "из характеристик ускорения. Для трёх методов, чувствительных к масштабу "
        "признаков, значения стандартизируются. Параметры стандартизации и медианы "
        "для заполнения пропусков рассчитываются только на обучающей части "
        "соответствующего разбиения."
    )

    report.section("2 Модели, параметры и рабочий порог")
    report.p(
        "За точку отсчёта взят DummyClassifier, который объявляет все записи "
        "нормальными. Логистическая регрессия складывает взвешенные признаки, "
        "метод ближайших соседей (kNN) опирается на похожие примеры, а метод "
        "опорных векторов (SVM) строит нелинейную границу между классами. Для SVM "
        "использованы радиальное ядро и сигмоидальная калибровка оценок. "
        "На каждом из трёх внутренних разбиений калибровки медианы и масштабы "
        "оцениваются только по файлам, на которых обучается SVM."
    )
    report.p(
        "Решающее дерево задаёт последовательность вопросов о признаках. Случайный "
        "лес объединяет ответы деревьев, обученных на разных подвыборках. "
        "Градиентный бустинг строит деревья последовательно, чтобы каждое уточняло "
        "оценки предыдущих. В работе использован HistGradientBoostingClassifier из "
        "scikit-learn [4]. Он позволяет ограничить сложность каждого дерева "
        "и проследить, как меняется качество после каждой итерации."
    )
    report.p(
        "Аномалий мало, поэтому даже постоянный ответ good даёт высокую долю верных "
        "ответов. Для выбора модели используется средняя точность (AP). Она "
        "характеризует, насколько хорошо модель поднимает реальные аномалии в начало "
        "списка подозрительных записей. В итоговых таблицах также приведена площадь "
        "под кривой точность–полнота (PR-AUC), вычисленная методом трапеций."
    )
    report.p(
        "Каждый набор параметров проверяется на пяти разбиениях обучающих файлов "
        "с сохранением доли аномалий. На каждом разбиении модель учится на четырёх "
        "частях и оценивается на пятой. GridSearchCV перебирает сочетания из таблицы. "
        "Выбирается вариант с наибольшей средней AP. Начальное состояние генератора равно 42."
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
    report.figure("cv", "Средние значения AP и разброс по разбиениям показаны на рисунке {n}.")
    report.figure("grid", "Оценки SVM для разных сочетаний параметров приведены на рисунке {n}.")
    report.p(
        "При снижении порога модель обычно обнаруживает больше аномалий и чаще "
        "подаёт ложные тревоги. Полнота (recall) показывает, какую долю аномалий "
        "обнаружили, а точность (precision) — какая доля тревог была обоснованной. "
        "Для выбора порога использован максимум F1. Эта метрика учитывает обе "
        "величины и падает, если одна из них мала."
    )
    report.p(
        f"Для модели «{winner}» этот максимум получен при пороге "
        f"{choices[winner].threshold:.3f} на оценках проверочных частей обучения. "
        "При равном F1 выбран больший порог, при котором тревог будет столько же "
        "или меньше. Для M03 применён тот же порог."
    )
    report.figure("threshold", "Зависимость метрик от порога показана на рисунке {n}.")

    report.section("3 Эксперименты с деревьями")
    report.p(
        "Для дерева по очереди изменены максимальная глубина и минимальное число "
        "объектов в листе. Для первой кривой разрешён лист из одной записи. "
        "Для второй глубина не ограничена. Ошибка равна единице "
        "минус сбалансированная доля верных ответов. Для ансамблей использована "
        "ранговая ошибка 1 − AP."
    )
    report.figure("tree", "Ошибки обучения и проверки дерева сопоставлены на рисунке {n}.")
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
    report.p(
        "У леса проверено, насколько усреднение большого числа деревьев снижает "
        "ошибку и зависимость результата от случайного состояния. Для бустинга "
        "сравниваются две скорости обучения. Меньший шаг меняет оценку медленнее, "
        "поэтому важно смотреть, какой результат достигается при разном числе деревьев."
    )
    report.figure("forest_count", "Зависимость ошибки от числа деревьев показана на рисунке {n}.")
    report.figure("forest_parameters", "Влияние параметров леса на AP показано на рисунке {n}.")
    report.figure(
        "boost_iterations", "Изменение ошибки бустинга по итерациям показано на рисунке {n}."
    )
    report.figure("boost_parameters", "Влияние параметров бустинга на AP показано на рисунке {n}.")

    report.section("4 Итоговая проверка и перенос")
    report.p(
        "После выбора моделей и порогов проведены итоговые проверки. В таблицах "
        "доля верных ответов учитывает все файлы, а сбалансированная доля даёт "
        "одинаковый вес нормальному и аномальному классам. Матрицы ошибок показывают, "
        "сколько записей каждого класса распознано верно, сколько аномалий пропущено "
        "и сколько нормальных записей вызвало тревогу."
    )
    report.p(
        "ROC показывает, как растёт доля обнаруженных аномалий при увеличении доли "
        "ложных тревог среди нормальных файлов. Кривая точность–полнота показывает, "
        "какая доля тревог остаётся обоснованной при том же охвате аномалий. Обе "
        "кривые построены по вероятностям класса bad при изменении порога. Для Dummy "
        "оценка всех файлов одинакова, поэтому AP равна доле аномалий. Трапецеидальная "
        "PR-AUC для этой модели выше AP из-за отрезка до условной точки нулевой полноты."
    )
    test_tables(report, "Обычный тест", standard)
    test_tables(report, "Перенос на M03", transfer)
    report.figure("roc_pr", "Кривые ROC и точность–полнота приведены на рисунке {n}.")
    report.figure(
        "matrix", "Нормированные матрицы ошибок выбранной модели показаны на рисунке {n}."
    )
    report.figure(
        "transfer", "Значения AP обычного теста и переноса на M03 сопоставлены на рисунке {n}."
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
    report.p(
        f"Доли аномалий составляют {sum(records[i].label for i in test) / len(test):.1%} "
        f"в обычном тесте и {sum(records[i].label for i in moved) / len(moved):.1%} "
        "на M03. Эти доли задают разные уровни AP случайного ранжирования. При "
        "сопоставлении AP и F1 также следует учитывать малое число аномальных файлов."
    )

    report.section("5 Перенос на август 2021 года")
    late_bad = sum(records[i].label for i in late)
    report.p(
        f"Из записей M01 и M02 до августа 2021 года выделили ранний тест "
        f"({len(control)} файлов, из них {sum(records[i].label for i in control)} "
        f"аномальных). Остальные {len(earlier)} ранних файлов использовали для обучения "
        "и подбора параметров и порогов. В августовский тест вошли все "
        f"{len(late)} записей этих двух станков, из них {late_bad} аномальных."
    )
    report.p(
        f"По средней AP на обучении выбрали модель «{time_winner}». В обоих тестах "
        "для каждого метода использовали один и тот же обученный классификатор и порог. "
        "Разность AP рассчитана как AP августа за вычетом AP раннего теста."
    )
    report.figure(
        "time_transfer", "Значения AP раннего и августовского тестов сопоставлены на рисунке {n}."
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
        [19, 38, 13, 23, 15, 14, 15, 14, 19],
        "Последний столбец таблицы {n} показывает разность AP двух тестов.",
    )
    for name in dict.fromkeys((time_winner, "SVM", FOREST)):
        _, false_alarms, missed, detected = later[name].matrix.ravel()
        report.p(
            f"В августе модель «{name}» обнаружила {detected} из {detected + missed} "
            f"аномалий. Число ложных тревог — {false_alarms}."
        )

    test_tables(report, "Август 2021 года", later)

    report.section("6 Интерпретация и инженерный разбор")
    report.p(
        "Чтобы понять, на какие измерения опирается выбранная модель, один признак "
        "перемешивали между файлами, сохраняя остальные. Если AP падает, модель "
        "пользовалась информацией этого признака. Связанные признаки могут частично "
        "заменять друг друга. Отрезки на графике показывают стандартное отклонение "
        "по пяти проверочным частям и 12 повторениям."
    )
    report.figure("importance", "Результаты перестановки признаков показаны на рисунке {n}.")
    for title, indices, evaluation in [
        ("Обычный тест", test, standard[winner]),
        ("Перенос на M03", moved, transfer[winner]),
    ]:
        tn, fp, fn, tp = evaluation.matrix.ravel()
        report.p(
            f"{title}. Модель «{winner}» пропустила {fn} из {fn + tp} аномалий, а "
            f"{fp} из {tn + fp} нормальных записей вызвали ложную тревогу."
        )
        rows = errors_by_operation(records, indices, evaluation)
        if rows:
            report.table(
                f"{title}. Операции с ошибками модели «{winner}»",
                [["Станок / операция", "good", "Ложные тревоги", "bad", "Пропуски"], *rows],
                [48, 24, 36, 24, 36],
                f"Ошибки по операциям для выборки «{title}» приведены в таблице {{n}}.",
            )
        if title == "Перенос на M03" and fp:
            examples = sorted((row for row in rows if row[2]), key=lambda row: -row[2])[:2]
            details = ", ".join(
                f"{key.split(' / ')[-1]} — {fp} из {good}" for key, good, fp, _, _ in examples
            )
            labels = np.array([records[i].label for i in indices])
            median = np.median(evaluation.scores[labels == 0])
            report.p(
                f"Ложные тревоги возникали в нескольких операциях ({details} нормальных "
                f"записей). Медианная оценка нормальных файлов M03 составила {median:.3f} "
                f"при рабочем пороге {choices[winner].threshold:.3f}."
            )
    ordinary_tn, ordinary_fp, _, _ = standard[winner].matrix.ravel()
    moved_tn, moved_fp, _, _ = transfer[winner].matrix.ravel()
    report.p(
        f"Среди нормальных записей доля ложных тревог изменилась с "
        f"{ordinary_fp / (ordinary_tn + ordinary_fp):.1%} в обычном тесте "
        f"до {moved_fp / (moved_tn + moved_fp):.1%} на M03. На новом станке "
        "штатные записи часто получают высокую оценку аномальности. "
        "Различия в фоне вибрации и составе операций могут объяснять этот результат. "
        "Метки показывают состояние процесса и не позволяют установить конкретную "
        "причину отклонения. Проверка переноса характеризует один отложенный станок."
    )

    report.section("ЗАКЛЮЧЕНИЕ")
    winner_moved = max(transfer, key=lambda name: transfer[name].metrics["AP"])
    best_f1 = max(transfer, key=lambda name: transfer[name].metrics["F1"])
    tn, fp, fn, tp = transfer[winner].matrix.ravel()
    report.p(
        f"Модель «{winner}» выбрана по средней AP на обучающих разбиениях. "
        f"При переносе на M03 с прежним порогом она обнаружила {tp} из {tp + fn} аномалий "
        f"и подала ложную тревогу для {fp} из {tn + fp} нормальных записей. "
        f"На этом станке модель «{winner_moved}» лучше ранжировала аномалии "
        f"(AP {transfer[winner_moved].metrics['AP']:.3f}), а модель «{best_f1}» "
        f"получила лучший F1 при заранее выбранном пороге "
        f"({transfer[best_f1].metrics['F1']:.3f}). Сравнение относится к одному "
        "отложенному станку."
    )
    early_ap = early_test[time_winner].metrics["AP"]
    late_ap = later[time_winner].metrics["AP"]
    _, alarms, _, detected = later[time_winner].matrix.ravel()
    svm_early = early_test["SVM"].metrics["AP"]
    svm_late = later["SVM"].metrics["AP"]
    forest_early = early_test[FOREST].metrics["AP"]
    forest_late = later[FOREST].metrics["AP"]
    report.p(
        f"При переносе во времени AP модели «{time_winner}», выбранной по ранним записям, "
        f"изменилась с {early_ap:.3f} до {late_ap:.3f}. В августе она обнаружила "
        f"{detected} из {late_bad} аномалий, число ложных тревог — {alarms}. "
        f"У SVM AP изменилась с {svm_early:.3f} до {svm_late:.3f}. "
        f"У случайного леса AP изменилась с {forest_early:.3f} до {forest_late:.3f}. "
        f"Августовский тест включает только {late_bad} аномалий, поэтому одна ошибка "
        "может заметно изменить результат."
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
    extent.counts = (
        f"1 кн., {report.image_count} рис., {report.table_count} табл., 5 источн., 0 прил"
    )
    report.save(path)
