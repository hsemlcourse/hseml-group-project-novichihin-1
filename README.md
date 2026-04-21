# ML Project — Predicting Digital Ad Campaign Profit


**Студент:** Новичихин Степан Алексеевич

**Группа:** БИВ232

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Быстрый старт](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)

## Описание задачи

Цель — построить ML-модель для прогноза прибыли рекламной кампании (`profit`) по характеристикам канала, формата и возраста креатива, параметров аудитории, аукционных метрик (quality score), поведенческих показателей пользователей и сезонных факторов. Результат используется для предварительной оценки эффективности кампании и оптимизации рекламного бюджета.

**Задача:** регрессия.

**Датасет:** [Digital Advertising Campaign Performance Dataset](https://www.kaggle.com/datasets/juniornsa/digital-advertising-campaign-performance-dataset) — 10 000 строк, 41 колонка.

**Целевая метрика:** **RMSE** (primary). Доп.: **MAE** и **R²**. Логарифмические метрики (RMSLE/MAPE) не применимы: `profit` принимает отрицательные значения.

**Контроль data leakage:** колонки `revenue`, `ad_spend`, `ROAS`, `CPA`, `CPC`, `actual_cpc` напрямую реконструируют таргет (`profit = revenue - ad_spend`) и удаляются из фич в `src/preprocessing._drop_leakage`.

## Структура репозитория

```
.
├── data/
│   ├── processed/              # Очищенные и обработанные данные
│   └── raw/                    # Сырые данные (CSV в .gitignore)
├── models/                     # Сохранённые модели
├── notebooks/
│   ├── 01_eda.ipynb            # Разведочный анализ
│   ├── 02_baseline.ipynb       # Baseline (Dummy, LinearRegression)
│   └── 03_experiments_cp1.ipynb  # Эксперименты CP1: Ridge, Lasso, KNN, RandomForest
├── presentation/               # Презентация (CP3)
├── report/
│   ├── images/                 # Графики и таблицы для отчёта
│   └── report.md               # Финальный отчёт
├── src/
│   ├── config.py               # SEED, пути, схема колонок, LEAKAGE_COLS
│   ├── data_loader.py          # Kaggle API + fallback публичной загрузки
│   ├── preprocessing.py        # Pipeline препроцессинга + сплит
│   ├── modeling.py             # Обёртки fit/evaluate, CV
│   └── utils.py                # set_seed, метрики
├── tests/
│   └── test.py                 # Smoke-тесты пайплайна
├── .github/workflows/ci.yml    # CI: ruff по src/
├── pyproject.toml              # Конфиг pytest / ruff
├── requirements.txt
└── README.md
```

## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone <url>
cd hseml-group-project-novichihin

# 2. Создать и активировать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Загрузить датасет
#    Вариант A: через Kaggle API — положить kaggle.json в ~/.kaggle/ (chmod 600)
#    Вариант B: вручную — скачать CSV с Kaggle и положить в data/raw/
python -m src.data_loader

# 5. Запустить ноутбуки
jupyter lab
# или воспроизводимо через papermill:
papermill notebooks/01_eda.ipynb notebooks/01_eda.ipynb --kernel python3
papermill notebooks/02_baseline.ipynb notebooks/02_baseline.ipynb --kernel python3
papermill notebooks/03_experiments_cp1.ipynb notebooks/03_experiments_cp1.ipynb --kernel python3

# 6. Тесты и линтер
pytest -q
ruff check src/ --line-length 120
```

Все эксперименты используют фиксированный `SEED=42` из `src/config.py`.

## Данные

- `data/raw/tech_advertising_campaigns_dataset.csv` — 10 000 строк, 41 колонка.
  Загружается автоматически через `src.data_loader.ensure_dataset()`; CSV не коммитится в репозиторий.
- `data/processed/` — артефакты предобработки (при необходимости).

**Ключевые группы фичей** (см. `src/config.py`):

- Таргетинг: `target_audience_age`, `target_audience_gender`, `audience_interest_category`, `income_bracket`, `purchase_intent_score`, `retargeting_flag`.
- Креатив: `creative_format`, `creative_size`, `ad_copy_length`, `has_call_to_action`, `creative_emotion`, `creative_age_days`.
- Канал/аукцион: `platform`, `ad_placement`, `device_type`, `operating_system`, `quality_score`.
- Поведение: `impressions`, `clicks`, `conversions`, `bounce_rate`, `avg_session_duration_seconds`, `pages_per_session`, `CTR`, `conversion_rate`.
- Сезонность: производные от `start_date` — `date_month`, `date_dayofweek`, `date_is_weekend`, `date_weekofyear`, `date_dayofyear`.

## Результаты

Итоги CP1 на отложенной выборке (time-based split 70/15/15):

| Модель              | RMSE (val) | RMSE (test) | MAE (test) | R² (test) |
| ------------------- | ---------: | ----------: | ---------: | --------: |
| Dummy(median)       |     73 946 |     127 486 |     32 116 |    −0.057 |
| LinearRegression    |     31 538 |      72 708 |     21 966 |     0.656 |
| Ridge (α=100)       |     30 739 |      72 869 |     21 378 |     0.655 |
| Lasso (α=10)        |     31 517 |      72 702 |     21 882 |     0.656 |
| KNN (k=5)           |     40 106 |      79 389 |     21 424 |     0.590 |
| **RandomForest(300)** |   41 051 |  **69 462** | **14 869** | **0.686** |

RandomForest уверенно лидирует по RMSE/MAE/R² на test. На CP2 добавим XGBoost/LightGBM, полноценный hyperparameter search и sampling-стратегии.

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md).
