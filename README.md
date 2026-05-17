# ML Project — Predicting Digital Ad Campaign Profit

**Студент:** Новичихин Степан Алексеевич

**Группа:** БИВ232

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Быстрый старт](#быстрый-старт)
4. [Docker](#docker)
5. [Деплой (CP3)](#деплой-cp3)
6. [Публичный деплой](#публичный-деплой)
7. [Данные](#данные)
8. [Результаты](#результаты)
9. [Отчёт](#отчёт)

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
│   ├── processed/                # Очищенные и обработанные данные
│   └── raw/                      # Сырые данные (CSV в .gitignore)
├── models/
│   └── final_cp2.joblib          # Финальная модель (GradientBoostingRegressor)
├── notebooks/
│   ├── 01_eda.ipynb              # Разведочный анализ
│   ├── 02_baseline.ipynb         # Baseline без календарного FE
│   ├── 03_experiments_cp1.ipynb  # Ridge, Lasso, KNN, RandomForest
│   └── 04_experiments_cp2.ipynb  # Бустинг, RandomizedSearchCV, Stacking, PCA
├── src/
│   ├── api/                      # FastAPI-сервис (CP3)
│   │   ├── main.py               # Приложение, эндпоинты
│   │   ├── schemas.py            # Pydantic-модели запросов/ответов
│   │   └── service.py            # Загрузка модели и предсказания
│   ├── ui/                       # Streamlit UI (CP3)
│   │   └── app.py                # Веб-интерфейс
│   ├── config.py                 # SEED, пути, схема колонок, LEAKAGE_COLS
│   ├── data_loader.py            # Kaggle API + fallback публичной загрузки
│   ├── preprocessing.py          # Pipeline препроцессинга + сплит
│   ├── modeling.py               # Обёртки fit/evaluate, CV
│   └── utils.py                  # set_seed, метрики
├── tests/
│   ├── test.py                   # Smoke-тесты пайплайна
│   └── test_api.py               # Тесты API (CP3)
├── report/
│   ├── images/                   # Графики и таблицы для отчёта
│   └── report.md                 # Финальный отчёт
├── presentation/                 # Презентация
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml      # CI: ruff (src/) + pytest
├── pyproject.toml                # Конфиг pytest / ruff
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

## Docker

Образ **Linux** (`python:3.10-slim` + `libgomp1` для LightGBM). В CP3 `docker-compose.yml` определяет два сервиса: `api` (FastAPI) и `ui` (Streamlit).

```bash
# Собрать и запустить всё одной командой
docker compose up --build

# Тесты и линтер в контейнере
docker compose run --rm api pytest -q
docker compose run --rm api ruff check src/ --line-length 120
```

## Деплой (CP3)

### Запуск

```bash
docker compose up --build
```

- **FastAPI (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit UI:** [http://localhost:8501](http://localhost:8501)

Без Docker:

```bash
# Терминал 1 — API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Терминал 2 — UI
API_URL=http://localhost:8000 streamlit run src/ui/app.py --server.port 8501
```

### API-эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Проверка живости |
| GET | `/model` | Метаданные модели (версия, RMSE, R²) |
| POST | `/predict` | Предсказание для одной кампании |
| POST | `/predict_batch` | Предсказание для нескольких кампаний |

### Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2024-06-15",
    "campaign_objective": "Lead Generation",
    "platform": "Facebook",
    "ad_placement": "Feed",
    "device_type": "Mobile",
    "operating_system": "Android",
    "creative_format": "Image",
    "creative_size": "300x250",
    "ad_copy_length": "Medium",
    "creative_emotion": "Curiosity",
    "target_audience_age": "25-34",
    "target_audience_gender": "Male",
    "audience_interest_category": "Shoppers",
    "income_bracket": "$50K-$100K",
    "purchase_intent_score": "Medium",
    "day_of_week": "Monday",
    "industry_vertical": "E-commerce",
    "budget_tier": "Medium",
    "has_call_to_action": true,
    "retargeting_flag": false,
    "creative_age_days": 30,
    "quarter": 2,
    "hour_of_day": 14,
    "campaign_day": 10,
    "quality_score": 7,
    "impressions": 50000,
    "clicks": 500,
    "conversions": 20,
    "bounce_rate": 45.0,
    "avg_session_duration_seconds": 120.0,
    "pages_per_session": 3.5,
    "CTR": 1.0,
    "conversion_rate": 4.0
  }'
```

**Ответ:**

```json
{"profit_prediction": 2771.54}
```

### Python-клиент

```python
import requests

payload = {
    "start_date": "2024-06-15",
    "platform": "Facebook",
    # ... (остальные поля)
}
resp = requests.post("http://localhost:8000/predict", json=payload)
print(resp.json()["profit_prediction"])
```

## Публичный деплой

Помимо локального `docker compose`, проект задеплоен на двух бесплатных платформах. Они независимы — Streamlit (HF) обращается к FastAPI (Render) через публичный HTTPS, между ними настроен CORS.

| Платформа | Что хостит | Файл-конфиг | Публичный URL |
|-----------|-----------|-------------|---------------|
| **Render** | FastAPI (`/health`, `/model`, `/predict`, `/predict_batch`) | [render.yaml](render.yaml) | `https://<service>.onrender.com` |
| **Hugging Face Spaces** | Streamlit UI | [deploy/hf-space/](deploy/hf-space/) | `https://huggingface.co/spaces/<owner>/ad-profit-predictor` |

### Render (FastAPI)

1. Запушить ветку `cp3` в GitHub.
2. <https://dashboard.render.com> → **New → Blueprint** → подключить репозиторий.
3. Render найдёт [render.yaml](render.yaml) и развернёт сервис `ad-profit-api`.
4. Healthcheck по пути `/health`. Free plan: 512 MB RAM, засыпает после 15 мин неактивности, cold start ~30 с.
5. Проверка:
   ```bash
   curl https://<service>.onrender.com/health
   curl -X POST https://<service>.onrender.com/predict -H "Content-Type: application/json" -d '{...}'
   ```

### Hugging Face Spaces (Streamlit)

Полная инструкция: [deploy/hf-space/DEPLOY.md](deploy/hf-space/DEPLOY.md). Кратко:

1. <https://huggingface.co/new-space> → SDK **Streamlit** → создать.
2. Загрузить `deploy/hf-space/{app.py, requirements.txt, README.md}` в репозиторий Space.
3. В **Settings → Variables** задать `API_URL = https://<service>.onrender.com`.
4. Открыть `https://huggingface.co/spaces/<owner>/ad-profit-predictor`.

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
