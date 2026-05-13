# ML Project — Predicting Digital Ad Campaign Profit


**Студент:** Новичихин Степан Алексеевич

**Группа:** БИВ232

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Быстрый старт](#быстрый-старт)
4. [Docker (CP2)](#docker-cp2)
5. [Данные](#данные)
6. [Результаты](#результаты)
7. [Отчёт](#отчёт)

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
│   ├── 01_eda.ipynb             # Разведочный анализ
│   ├── 02_baseline.ipynb      # Baseline без календарного FE (Dummy, LinearRegression)
│   ├── 03_experiments_cp1.ipynb  # Ridge, Lasso, KNN, RandomForest
│   └── 04_experiments_cp2.ipynb  # CP2: бустинг, RandomizedSearchCV, Stacking, PCA
├── Dockerfile
├── docker-compose.yml
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
├── .github/workflows/ci.yml    # CI: ruff (src/) + pytest
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
papermill notebooks/04_experiments_cp2.ipynb notebooks/04_experiments_cp2.ipynb --kernel python3

# 6. Тесты и линтер
pytest -q
ruff check src/ --line-length 120
```

Все эксперименты используют фиксированный `SEED=42` из `src/config.py`.

Рекомендуется перед сдачей полностью перезапустить ноутбуки с записью вывода (один раз, при наличии `data/raw/*.csv`):

```bash
for nb in 01_eda 02_baseline 03_experiments_cp1 04_experiments_cp2; do \
  python -m jupyter nbconvert --to notebook --execute "notebooks/${nb}.ipynb" --inplace --ExecutePreprocessor.timeout=1800; \
done
```

После прогона [`notebooks/04_experiments_cp2.ipynb`](notebooks/04_experiments_cp2.ipynb) обновляются, в частности: [`report/images/cp2_experiments.csv`](report/images/cp2_experiments.csv), [`permutation_importance_cp2.csv`](report/images/permutation_importance_cp2.csv), [`permutation_importance_cp2.png`](report/images/permutation_importance_cp2.png), [`train_test_shift_cp2.csv`](report/images/train_test_shift_cp2.csv), [`train_test_shift_cp2.png`](report/images/train_test_shift_cp2.png), [`split_ablation_cp2.csv`](report/images/split_ablation_cp2.csv), [`pca_sweep_cp2.csv`](report/images/pca_sweep_cp2.csv), графики PCA (`report/images/pca_*_cp2.png`) и [`models/final_cp2.joblib`](models/final_cp2.joblib) (при необходимости коммитится для сдачи).

## Docker (CP2) и эталонный прогон

Образ **Linux** (`python:3.10-slim` + `libgomp1` для LightGBM). В контейнере обычно успешно импортируются **LightGBM** и **XGBoost** из `requirements.txt`. На **macOS** локально LightGBM часто недоступен (OpenMP); ноутбук [`04_experiments_cp2.ipynb`](notebooks/04_experiments_cp2.ipynb) тогда использует **HistGradientBoostingRegressor** / **GradientBoostingRegressor** из sklearn — это **другой численный прогон**, но тот же протокол сплита и таблицы метрик.

| Окружение | Что ожидать в первой ячейке ноутбука | Таблица `cp2_experiments.csv` |
|-----------|--------------------------------------|-------------------------------|
| `docker compose run ... papermill` / Linux venv с LGBM+XGB | `HAS_LGBM=True`, `HAS_XGB=True` | Модели LightGBM / XGBoost в exp02/exp03 |
| Локально macOS / без библиотек | фоллбек sklearn (как в закоммиченном CSV сейчас) | HistGradient / GradBoost в названиях строк |

Чтобы получить **эталонный прогон с LightGBM и XGBoost**, воспроизводите CP2 в Docker:

```bash
docker compose build
docker compose run --rm app pytest -q
docker compose run --rm app ruff check src/ --line-length 120
# при смонтированном каталоге и наличии data/raw CSV:
docker compose run --rm app papermill notebooks/04_experiments_cp2.ipynb notebooks/04_experiments_cp2.ipynb --kernel python3
```

Локально (без Docker) команды те же, что в разделе «Быстрый старт»; сравнивайте числа только при **одинаковом** наборе библиотек.

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
| LinearRegression¹   |     31 564 |      72 691 |     21 931 |     0.656 |
| Ridge (α=100)       |     30 739 |      72 869 |     21 378 |     0.655 |
| Lasso (α=10)        |     31 517 |      72 702 |     21 882 |     0.656 |
| KNN (k=5)           |     40 106 |      79 389 |     21 424 |     0.590 |
| **RandomForest(300)** |   41 051 |  **69 462** | **14 869** | **0.686** |

¹`LinearRegression` — из [`02_baseline.ipynb`](notebooks/02_baseline.ipynb) (`prepare_features_baseline`, без календарного FE).

RandomForest уверенно лидирует среди моделей CP1.

### CP2 (ветка `cp2`)

Эксперименты: [`notebooks/04_experiments_cp2.ipynb`](notebooks/04_experiments_cp2.ipynb). Подбор гиперпараметров — только на train (CV), выбор финальной модели — по **val RMSE**. Таблица: [`report/images/cp2_experiments.csv`](report/images/cp2_experiments.csv).

| Эксперимент | Модель | RMSE (val) | RMSE (test) | R² (test) |
|-------------|--------|------------|-------------|-----------|
| exp01_rf | RandomForest | 40 084 | 69 441 | 0.686 |
| exp02_boost_tuned | HistGradientBoosting (sklearn)* | 30 876 | 79 368 | 0.590 |
| exp03_tree_tuned | GradientBoosting (sklearn)* | **26 778** | **64 019** | **0.733** |
| exp04_stack | Stacking (RF+Boost→Ridge) | 34 734 | 72 138 | 0.662 |
| exp05_pca_boost | HistGradientBoosting + PCA(0.95)* | 39 829 | 79 027 | 0.594 |

\*В Docker с рабочими `lightgbm`/`xgboost` в строках вместо sklearn могут быть **LightGBM** и **XGBoost** — перезапустите ноутбук в Linux-окружении для эталона.

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md).
