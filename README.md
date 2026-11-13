# Нормальный и аномальный режим фрезерного станка

Задание 2 по курсу ML in mechanics. Бинарная классификация записей вибрации Bosch CNC Machining. Один объект соответствует одному файлу HDF5. Положительный класс — `bad`.

## Подготовка на Windows

Требуются Python 3.14 и шрифты Times New Roman (`times.ttf`, `timesbd.ttf`). Если шрифты находятся вне стандартного каталога Windows, задайте путь в переменной `CNC_FONT_DIR`.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

В работе использована ревизия Bosch `d60581d6a3ab6015dcc5488c3d76112bb8e1bcb1`. Архив этой ревизии доступен [на GitHub](https://github.com/boschresearch/CNC_Machining/archive/d60581d6a3ab6015dcc5488c3d76112bb8e1bcb1.zip). Архив можно использовать для сборки без доступа к сети. Укажите путь к распакованному каталогу `data` через `--data`.

## Сборка одной командой

```powershell
.\.venv\Scripts\python build_report.py
```

При первом запуске команда скачивает 1702 файла Bosch (около 950 МБ) в `.bosch/data` и сверяет их контрольные суммы. При каждом запуске все файлы сверяются с манифестом зафиксированной ревизии. В стандартном кэше недостающие или повреждённые файлы скачиваются заново. Для каталога, переданного через `--data`, выполняется проверка без скачивания.

Команда читает исходные записи, строит признаки внутри конвейера, подбирает параметры и пороги, выполняет три отложенные проверки и создаёт `report.pdf`, рисунки и `build/metrics.csv`. Производные файлы можно удалить и собрать снова. Исходный набор хранится локально.

Если зависимости уже установлены в локальную папку .deps, в PowerShell можно запустить `$env:PYTHONPATH = (Resolve-Path .deps).Path; python build_report.py`.

Для обычного теста выделено 25 % файлов M01 и M02. M03 целиком используется для проверки переноса между станками. Для проверки переноса во времени на M01 и M02 обучение и подбор выполняются на записях до августа 2021 года, а все файлы августа 2021 года используются для теста. Основная метрика выбора — AP класса bad по пяти разбиениям обучающей выборки. Рабочий порог выбирается по F1 на вневыборочных оценках обучения.

## Проверки кода

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m black --check --line-length 99 cnc_data.py download_data.py experiment.py figures.py report.py build_report.py test_calculations.py test_regressions.py
.\.venv\Scripts\python -m flake8 .
```

## Источник данных

Tnani M.-A., Feil M., Diepold K. Smart Data Collection System for Brownfield CNC Milling Machines: A New Benchmark Dataset for Data-Driven Machine Monitoring. Procedia CIRP, 107, 131–136, 2022. [DOI](https://doi.org/10.1016/j.procir.2022.04.022). Данные опубликованы по лицензии CC BY 4.0.
