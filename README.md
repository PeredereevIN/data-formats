# Modern Data Formats Study

Сравнительное исследование Apache Parquet, Vortex и Lance на больших табличных данных.
Проект выполняет генерацию исходного набора в C++, конвертацию в Vortex/Lance, независимую проверку идентичности и native benchmark. Python используется для конвертации, запуска измерений и последующего анализа, а генерация Parquet выполняется нативно.

## 1. Конфигурация эксперимента

По умолчанию используется 250 batch-наборов по 1 000 000 строк, то есть 250 000 000 строк. Размер Arrow read batch — 65 536 строк, Parquet row group — 128 000 строк.

Основной профиль `mixed_v2` содержит идентификаторы разной кардинальности, временные поля, категории, страны, статусы, строковые идентификаторы устройств, хеши, числовые значения, пропуски и текстовые поля. Это позволяет проверять не только простой последовательный scan, но и проекции и фильтрацию по полям с разными свойствами.

Доступен также `DATA_PROFILE=legacy` для воспроизведения старой семиколоночной схемы.

Версии:

```text
PyArrow   25.0.1
Vortex    0.86.1
Lance     12.0.0
```

## 2. Структура проекта

```text
parquet-format-study/
├── src/main.cpp
├── scripts/
│   ├── vortex_converter.py
│   ├── lance_converter.py
│   ├── integrity_validator.py
│   └── benchmark_runner.py
├── analysis/
│   ├── analyze_benchmark.py
│   └── requirements.txt
├── docker/
│   ├── Dockerfile.parquet
│   ├── Dockerfile.vortex
│   ├── Dockerfile.lance
│   ├── Dockerfile.integrity
│   ├── Dockerfile.benchmark
│   └── Dockerfile.analytics
├── data/
│   ├── parquet/
│   ├── vortex/
│   ├── lance/
│   ├── validation/
│   ├── benchmark/
│   └── analysis/
├── compose.yaml
├── .env.example
├── CMakeLists.txt
└── README.md
```

`data/` содержит только результаты выполнения и не является исходным кодом проекта.

## 3. Запуск

Сначала при необходимости скопировать `.env.example` в `.env` и изменить параметры.

### Генерация Parquet

```bash
docker compose run --rm parquet-generator
```

По умолчанию используется `DATA_PROFILE=mixed_v2`.

### Конвертация

```bash
docker compose run --rm vortex-converter
docker compose run --rm lance-converter
```

Для конвертации используется потоковая передача Arrow RecordBatch без материализации полного batch в Python. Независимые batch-наборы могут конвертироваться параллельно через `CONVERTER_WORKERS`.

Например:

```bash
CONVERTER_WORKERS=4 docker compose run --rm vortex-converter
CONVERTER_WORKERS=4 docker compose run --rm lance-converter
```

Количество рабочих потоков следует подбирать по CPU, памяти и скорости диска: чрезмерное увеличение параллелизма может замедлить запись из-за конкуренции за ресурсы.

### Проверка идентичности

```bash
docker compose run --rm integrity-validator
```

Проверяются схемы, количество строк и значения всех столбцов между Parquet, Vortex и Lance.

### Native benchmark

```bash
docker compose run --rm benchmark
```

Benchmark запускается только после успешного integrity report.

### Анализ

```bash
docker compose run --rm analytics
```

Результаты анализа записываются в `data/analysis/`.

## 4. Методика benchmark

Измеряются последовательное чтение, проекции, фильтрация по полям с различной кардинальностью, операции метаданных и произвольный доступ. Для каждого формата используются его штатные API чтения. Результаты сохраняются в JSON/CSV и затем обрабатываются отдельно от измерительного кода.

Benchmark не сбрасывает page cache операционной системы. Поэтому текущие результаты относятся к локальной файловой системе с фактически сложившимся состоянием кэша и не должны интерпретироваться как cold-storage measurements.

Для каждого запуска фиксируются wall time, CPU user/system time, пиковое RSS, процессный I/O, throughput и число обрабатываемых строк. Для коротких операций используются повторные измерения и медиана.

## 5. Pipeline

```text
                 C++ data generator
                         │
                         ▼
                 Parquet / ZSTD
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Vortex converter       Lance converter
        Python + native       Python + native
              │                     │
              ▼                     ▼
           Vortex                 Lance
              │                     │
              └──────────┬──────────┘
                         ▼
                Integrity validator
                         │
                         ▼
                  Native benchmark
                         │
                         ▼
                   JSON / CSV
                         │
                         ▼
                  Python analysis
```

Основной принцип проекта: генерация и измеряемые операции должны выполняться максимально близко к нативному уровню библиотек, а Python должен использоваться прежде всего для orchestration и анализа результатов.
