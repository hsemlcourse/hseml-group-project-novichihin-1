# Отчёт по проекту

**Студент:** Новичихин Степан Алексеевич

**Группа:** БИВ232

**Тема:** Предсказание прибыльности digital-рекламной кампании по параметрам таргетинга, креатива и аукциона.

---

## 1. Введение и постановка задачи

**Цель проекта.** Построить модель регрессии, которая по характеристикам рекламной кампании (канал, формат и возраст креатива, таргетинг аудитории, аукционные метрики, сезонность, поведенческие показатели пользователей) предсказывает прибыль кампании `profit`. Модель нужна для предварительной оценки эффективности кампаний и оптимизации бюджета до/во время запуска.

**Формулировка.** Регрессия с непрерывным таргетом `profit` (разница между выручкой и затратами, может быть отрицательной).

**Обоснование метрики.**

- **Primary — RMSE.** Квадратичный штраф сильно наказывает крупные ошибки. В задаче оптимизации рекламного бюджета большая ошибка на «жирной» кампании стоит дороже, чем пара маленьких ошибок, поэтому именно RMSE ближе всего к бизнес-стоимости ошибки.
- **Secondary — MAE.** Устойчива к выбросам и интерпретируется в единицах таргета (USD). Нужна как «второе мнение», чтобы модель не подгонялась под хвосты распределения.
- **Secondary — R².** Сопоставимость моделей между собой, доля объяснённой дисперсии.
- **MAPE / RMSLE не используются:** `profit` принимает отрицательные значения (~40% наблюдений), логарифмические/процентные метрики на них не определены.

---

## 2. Поиск и описание данных

**Источник.** Kaggle: [juniornsa/digital-advertising-campaign-performance-dataset](https://www.kaggle.com/datasets/juniornsa/digital-advertising-campaign-performance-dataset). Датасет отвечает формальным требованиям (monetary, не Getting Started, ≥10 000 строк, ≥10 колонок).

**Объём.** 10 000 строк, 41 колонка, таргет `profit`.

**Колонки по смысловым группам.**

| Группа | Колонки |
|---|---|
| Идентификатор | `campaign_id` |
| Кампания / канал | `campaign_objective`, `platform`, `ad_placement`, `device_type`, `operating_system`, `industry_vertical`, `budget_tier` |
| Креатив | `creative_format`, `creative_size`, `ad_copy_length`, `has_call_to_action`, `creative_emotion`, `creative_age_days` |
| Таргетинг аудитории | `target_audience_age`, `target_audience_gender`, `audience_interest_category`, `income_bracket`, `purchase_intent_score`, `retargeting_flag` |
| Сезонность / время | `start_date`, `quarter`, `day_of_week`, `hour_of_day`, `campaign_day` |
| Аукцион | `quality_score`, `actual_cpc` |
| Поведение / трафик | `impressions`, `clicks`, `conversions`, `bounce_rate`, `avg_session_duration_seconds`, `pages_per_session`, `CTR`, `conversion_rate` |
| Стоимость / результат | `ad_spend`, `revenue`, `CPC`, `CPA`, `ROAS`, **`profit` (таргет)** |

Почему этот датасет подходит: соответствует теме (рекламные кампании, таргетинг, креатив, аукцион, поведение, сезонность), размер выше порогового, есть явный числовой таргет, хорошо покрыто доменной логикой e-commerce/медиа-закупок.

---

## 3. Обработка и подготовка данных

### 3.1 Очистка

- **Пропуски.** В датасете пропусков нет (`df.isna().mean() == 0` по всем колонкам). Тем не менее в пайплайн заложены импутеры (`SimpleImputer(median)` для числовых, `constant='missing'` для категориальных), чтобы пайплайн был устойчив к новым данным при переобучении.
- **Дубликаты.** Полных дубликатов и дубликатов по `campaign_id` не обнаружено.
- **Выбросы.** По ключевым числовым колонкам (`impressions`, `clicks`, `conversions`, `bounce_rate`, `pages_per_session`) — умеренные естественные хвосты, см. `report/images/numeric_outliers.png`. Строки не удаляются; влияние выбросов снижается за счёт `StandardScaler` + устойчивых моделей (Ridge/Lasso/RandomForest).
- **Типы данных.** `start_date` приводится к `datetime64` в `src/data_loader.load_raw`. Булевы колонки (`has_call_to_action`, `retargeting_flag`) приводятся к float в `src/preprocessing._cast_booleans`.

### 3.2 Контроль утечек (data leakage)

Профит определяется как `profit = revenue - ad_spend`, что легко проверить на данных:

```
(round(revenue - ad_spend, 2) == profit).mean() ≈ 1.0
```

Аналогично `ROAS = revenue/ad_spend`, `CPA = ad_spend/conversions`, `CPC = actual_cpc = ad_spend/clicks` — все они прямо несут информацию о стоимости/выручке. Поэтому в `src/config.LEAKAGE_COLS` заявлены:

```
revenue, ad_spend, ROAS, CPA, CPC, actual_cpc
```

Эти колонки **удаляются из фич** в `src/preprocessing._drop_leakage` до любых преобразований. Все прочие производные метрики (`CTR`, `conversion_rate`) зависят только от `clicks/impressions/conversions` и остаются в фичах — они характеризуют трафик, не стоимость.

### 3.3 Feature engineering

Выполняется внутри `Pipeline`, чтобы статистики учились только на train:

- **Календарные фичи** из `start_date`: `date_month`, `date_dayofweek`, `date_weekofyear`, `date_is_weekend`, `date_dayofyear` — фиксируют сезонность запуска кампании.
- **Категории** (17 low-card колонок) → `OneHotEncoder(handle_unknown='ignore')` с предварительной импутацией `'missing'`.
- **Числовые** (13 колонок) → `SimpleImputer(median)` + `StandardScaler` — стандартизация критична для линейных моделей и KNN.
- **Булевы** (2 колонки) → `SimpleImputer(most_frequent)`, подаются как 0/1.

Производные бизнес-метрики (`CTR`, `conversion_rate`) уже присутствуют в сыром датасете и включены в пайплайн как есть (деление «построчное», без оконных статистик → leakage не возникает).

### 3.4 Визуализации (в `report/images/`)

- `target_distribution.png` — гистограмма и QQ-plot распределения `profit` (тяжёлые хвосты, отрицательные значения).
- `numeric_outliers.png` — boxplots ключевых числовых фичей.
- `corr_heatmap.png` — тепловая карта корреляций числовых фичей.
- `profit_by_category.png` — boxplots `profit` по ключевым категориям.
- `profit_over_time.png` — усреднённый по неделям `profit` по датам старта (для выявления временного дрейфа).

### 3.5 Сплит и защита от data leakage

В датасете есть колонка `start_date`, поэтому применяется **time-based split**: датасет сортируется по дате, первые 70% идут в train, следующие 15% — val, последние 15% — test (`src/preprocessing.time_or_random_split`). Это исключает ситуации, когда модель обучается на «будущем» и угадывает «прошлое».

Дополнительные правила анти-leakage:

- Все параметры (медиана, OHE-категории, коэффициенты скейлера) учатся на train внутри sklearn Pipeline → val/test трансформируются уже обученным пайплайном.
- `GridSearchCV` работает **только на train** (с 5-fold KFold), гиперпараметры выбираются без участия val/test.
- Прямые компоненты таргета из фич удалены (см. 3.2).

Фиксируется `SEED=42` (`src/config.SEED`) и вызывается `src.utils.set_seed()` в начале каждого ноутбука и теста.

### 3.6 Сдвиг распределений train vs test (ковариатный сдвиг)

Помимо графика `profit_over_time.png`, для CP2 построена сводка по **числовым признакам после `prepare_features`**: средние, **медианы**, стандартные отклонения в train и test, разрывы по средним и медианам и нормировка \|Δmean\|/std(train) (для медиан — та же нормировка на std(train) для сопоставимости). Артефакты: [`train_test_shift_cp2.csv`](images/train_test_shift_cp2.csv), [`train_test_shift_cp2.png`](images/train_test_shift_cp2.png) (генерируются в [`notebooks/04_experiments_cp2.ipynb`](../notebooks/04_experiments_cp2.ipynb)).

По текущему прогону наибольший относительный разрыв средних у календарных признаков (`quarter`, `date_month`, `date_dayofyear`, `date_weekofyear`): тестовый период попадает в «другой» сезон относительно обучающей выборки. Это согласуется с типичной картиной time-based split: на валидации (ближе по времени к train) RMSE ниже, а на отложенном test — выше (см. `cp2_experiments.csv`, эксперимент `exp03_tree_tuned`: val ≈ 26.8k, test ≈ 64.0k). Модель не «ломается», но **сдвиг ковариат усиливает ошибку на «будущих» данных**, что подчёркивает выбор временного сплита как более честной оценки, чем случайное перемешивание строк.

### 3.7 Ablation: временной сплит vs случайный

Один и тот же финальный пайплайн (после `clone`) дообучается на **случайном** разбиении 70/15/15 с тем же составом колонок, но с `time_order=False` в `time_or_random_split` (строки перемешаны; `start_date` сохраняется для календарных фич). Результаты: [`split_ablation_cp2.csv`](images/split_ablation_cp2.csv).

На данных прогона случайный сплит даёт **заниженный RMSE на test** относительно строгого временного порядка (часть «сложных» наблюдений оказывается и в train, и в test), поэтому для сравнения моделей и отчёта об обобщении приоритет остаётся за **time split**. Случайное разбиение полезно как контрольный эксперимент, а не как целевая метрика под production.

---

## 4. Baseline-модель

**Модели.**

- `DummyRegressor(strategy='median')` — предсказывает медиану таргета, задаёт нижнюю границу качества.
- `LinearRegression` — линейный baseline на **минимальном** препроцессинге (`prepare_features_baseline`): без календарных признаков из `start_date`, OHE + масштабирование как в основном пайплайне (см. `02_baseline.ipynb`).

**Результаты (time-based split 70/15/15):**

| Модель           | Split | RMSE | MAE | R² |
| ---------------- | ----- | ---: | --: | -: |
| Dummy(median)    | val   |  73 946 | 18 846 | −0.044 |
| Dummy(median)    | test  | 127 486 | 32 116 | −0.057 |
| LinearRegression | val   |  31 564 | 14 750 |  0.810 |
| LinearRegression | test  |  72 691 | 21 931 |  0.656 |

Файл: [`report/images/baseline_metrics.csv`](images/baseline_metrics.csv).

**Вывод.** Линейная регрессия заметно обыгрывает Dummy по RMSE/MAE — в признаках есть сигнал. Разрыв val vs test характерен для time-based split (дрейф во времени).

---

## 5. Эксперименты CP1

Эксперименты — в [`notebooks/03_experiments_cp1.ipynb`](../notebooks/03_experiments_cp1.ipynb). Формат: **Гипотеза → Как проверялось → Результат**. Полноценный цикл из 4–5 моделей + бустингов + hyperparameter search выполняется в CP2; в CP1 закрываем базовый набор.

### Эксперимент 1. Ridge

- **Гипотеза.** L2-регуляризация улучшит/стабилизирует LinearRegression на большом числе OHE-фич.
- **Проверка.** `GridSearchCV(alpha ∈ {0.1, 1, 10, 100})`, 5-fold KFold на train (scoring=`neg_root_mean_squared_error`).
- **Результат.** Best `alpha=100`, RMSE на val `30 739` / test `72 869`, R² test `0.655`. Регуляризация почти не помогает — признаков не настолько много, чтобы модель переобучалась. Ridge близка к LinearRegression.

### Эксперимент 2. Lasso

- **Гипотеза.** L1 обнулит коэффициенты неинформативных OHE-уровней и даст более компактное решение.
- **Проверка.** `GridSearchCV(alpha ∈ {0.01, 0.1, 1, 10})` по 5-fold CV.
- **Результат.** Best `alpha=10`, RMSE test `72 702`, R² test `0.656`. Практически неотличимо от Ridge/LinearRegression — неинформативных фичей немного.

### Эксперимент 3. KNN

- **Гипотеза.** Схожие по таргетингу/креативу кампании дают схожий `profit`; KNN на стандартизованных фичах уловит «похожесть».
- **Проверка.** `GridSearchCV(n_neighbors ∈ {5, 15, 25})`.
- **Результат.** Best `k=5`, RMSE test `79 389`, R² test `0.590`. KNN проигрывает линейным моделям — большое число OHE-измерений размывает расстояние.

### Эксперимент 4. RandomForest

- **Гипотеза.** Нелинейная ансамблевая модель уловит взаимодействия между таргетингом, креативом и аукционными параметрами.
- **Проверка.** Фиксированные параметры `n_estimators=300, min_samples_leaf=1, random_state=SEED`; полный tuning — в CP2.
- **Результат.** RMSE test **`69 462`**, MAE test **`14 869`**, R² test **`0.686`** — лучший результат CP1.

### Сводная таблица CP1

| Модель | Split | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| RandomForest(300) | val  | 41 051 |  8 375 | 0.678 |
| RandomForest(300) | test | **69 462** | **14 869** | **0.686** |
| Ridge(α=100)      | val  | 30 739 | 14 060 | 0.820 |
| Ridge(α=100)      | test | 72 869 | 21 378 | 0.655 |
| Lasso(α=10)       | val  | 31 517 | 14 661 | 0.810 |
| Lasso(α=10)       | test | 72 702 | 21 882 | 0.656 |
| KNN(k=5)          | val  | 40 106 | 13 298 | 0.693 |
| KNN(k=5)          | test | 79 389 | 21 424 | 0.590 |
| LinearRegression* | val  | 31 564 | 14 750 | 0.810 |
| LinearRegression* | test | 72 691 | 21 931 | 0.656 |
| Dummy(median)*    | val  | 73 946 | 18 846 | −0.044 |
| Dummy(median)*    | test | 127 486| 32 116 | −0.057 |

*\*Строки LinearRegression / Dummy — те же, что в baseline (`prepare_features_baseline`), для сравнения в одной таблице.*

Файл: [`report/images/cp1_experiments.csv`](images/cp1_experiments.csv).

Промежуточный лидер CP1 — **RandomForest** по RMSE на test среди моделей того этапа.

### 5.2 Эксперименты CP2

Полный цикл: [`notebooks/04_experiments_cp2.ipynb`](../notebooks/04_experiments_cp2.ipynb). Подбор гиперпараметров — `RandomizedSearchCV` (5-fold) **только на train**; единый источник метрик — [`report/images/cp2_experiments.csv`](images/cp2_experiments.csv). В Docker/Linux при установленных `lightgbm`/`xgboost` ноутбук использует их; при ошибке импорта (типично macOS без OpenMP для LightGBM) подставляются `HistGradientBoostingRegressor` и `GradientBoostingRegressor` из sklearn — см. первую ячейку ноутбука. Ниже — формат **гипотеза → проверка → результат**; цифры взяты из текущего закоммиченного `cp2_experiments.csv` (прогон со sklearn-фоллбеком).

**Random Forest (`exp01_rf`).** Гипотеза: лес из CP1 остаётся сильной отсечкой качества. Проверка: те же 300 деревьев, полный препроцессинг, без дополнительного тюнинга. Результат: RMSE val **40 084** / test **69 441**, R² test **0.686** — база для сравнения бустингов и ансамбля.

**Бустинг «широких» деревьев (`exp02_boost_tuned`, HistGradientBoosting).** Гипотеза: бустинг сильнее улавливает нелинейности, чем лес. Проверка: 20 итераций `RandomizedSearchCV` по сетке гиперпараметров на train. Результат: на val RMSE **30 876** (лучше леса), но на test **79 368** и R² test **0.590** — сильный разрыв val/test, несовпадение с лидером по обобщению.

**Бустинг «мелких» деревьев (`exp03_tree_tuned`, GradientBoosting/sklearn).** Гипотеза: более консервативная глубина и subsample дадут лучший баланс смещения/дисперсии под временной сплит. Проверка: 15 итераций `RandomizedSearchCV`. Результат: лучший RMSE на val (**26 778**) и лучший на test среди ряда (**64 019**, R² **0.733**); эта модель выбрана финальной по протоколу «минимум val».

**Stacking (`exp04_stack`, RF + лучший бустинг → Ridge).** Гипотеза: мета-регрессор объединит сильные стороны баз. Проверка: `StackingRegressor` с 5-fold на мета-уровне. Результат: RMSE val **34 734** / test **72 138** — мета-модель не улучшила ни val, ни test относительно `exp03_tree_tuned`.

**PCA + бустинг (`exp05_pca_boost`).** Гипотеза: сжатие OHE-после-препроцессора до 95% дисперсии ускорит обобщение. Проверка: `PCA(0.95)` + клон лучшего бустинга из exp02, визуализации [`pca_scree_cp2.png`](images/pca_scree_cp2.png), [`pca2_scatter_cp2.png`](images/pca2_scatter_cp2.png). Результат: RMSE val **39 829** / test **79 027** — хуже финального дерева без PCA.

**Доп. сетка степени сжатия PCA (фиксированный GBR).** Для явной кривой «качество vs число компонент» на val: [`pca_sweep_cp2.csv`](images/pca_sweep_cp2.csv). При **n_components = 60** достигается RMSE val ≈ **29 764**, при **0.95** дисперсии — ≈ **29 805**; оба значения **выше** лучшего RMSE val полного пайплайна без PCA (**26 778**), то есть в этом датасете сжатие после OHE не даёт выигрыша по согласованной метрике.

Сводная таблица (источник — `cp2_experiments.csv`):

| Эксперимент | Модель | RMSE (val) | RMSE (test) | R² (test) |
|-------------|--------|-----------:|------------:|----------:|
| exp01_rf | RandomForest | 40 084 | 69 441 | 0.686 |
| exp02_boost_tuned | HistGradientBoosting (sklearn) | 30 876 | 79 368 | 0.590 |
| exp03_tree_tuned | GradientBoosting (sklearn) | **26 778** | **64 019** | **0.733** |
| exp04_stack | Stacking (RF+Boost→Ridge) | 34 734 | 72 138 | 0.662 |
| exp05_pca_boost | HistGradientBoosting + PCA(0.95) | 39 829 | 79 027 | 0.594 |

Финальный артефакт обучения: [`models/final_cp2.joblib`](../models/final_cp2.joblib) (ноутбук 04).

---

## 6. Финальная модель и интерпретируемость

**Выбор модели (CP2).** По минимальному RMSE на **val** выбирается `GradientBoostingRegressor` (sklearn) из эксперимента `exp03_tree_tuned`. На test: RMSE **64 019**, R² **0.733** — улучшение относительно лучшего CP1 RandomForest на test (**69 462**).

**Permutation importance (проверяемый артефакт).** На отложенной **val** для итогового полного `Pipeline` посчитана перестановочная важность (`sklearn.inspection.permutation_importance`, `n_repeats=5`, `scoring='neg_root_mean_squared_error'`). Файлы: [`permutation_importance_cp2.csv`](images/permutation_importance_cp2.csv), [`permutation_importance_cp2.png`](images/permutation_importance_cp2.png). Топ-5 по среднему росту RMSE при перемешивании признака: **`conversions`**, **`industry_vertical`**, **`income_bracket`**, **`creative_age_days`**, **`clicks`**. Это согласуется с бизнес-логикой: объём конверсий и отраслевой контекст сильнее всего влияют на точечный прогноз после остальных (не утечных) фич; далее идут богатые по уровням категориальные и поведенческие сигналы. Для сравнения с внутримодельными `feature_importances_` у бустинга полезно смотреть именно перестановочную важность поверх **уже обученного** пайплайна.

**Baseline без FE (рубрика).** В [`notebooks/02_baseline.ipynb`](../notebooks/02_baseline.ipynb) используется `prepare_features_baseline`: без календарных производных от `start_date` (см. `src/preprocessing.py`).

---

## 7. Деплой

### 7.1 Архитектура

Деплой состоит из двух сервисов, запускаемых через `docker compose`:

```
┌────────────────┐       POST /predict       ┌──────────────────┐
│  Streamlit UI  │ ────────────────────────── │   FastAPI :8000   │
│   :8501        │       JSON response        │                  │
└────────────────┘                            │  joblib.load()   │
                                              │  final_cp2.joblib│
      curl / Swagger /docs  ──────────────── │  Pipeline:       │
                                              │   prep → OHE →   │
                                              │   GBR            │
                                              └──────────────────┘
```

Препроцессинг (удаление leakage-колонок, календарные фичи из `start_date`, OHE, масштабирование) **не дублируется** — всё уже внутри сохранённого `Pipeline` (`FunctionTransformer(prepare_features) → ColumnTransformer → GradientBoostingRegressor`). Streamlit обращается к FastAPI **по HTTP** и не загружает модель сам.

**Leakage-колонки (`revenue`, `ad_spend`, `ROAS`, `CPA`, `CPC`, `actual_cpc`) в API не запрашиваются** — они исключены из Pydantic-схемы, поскольку реконструируют таргет.

### 7.2 Запуск

```bash
docker compose up --build
```

После запуска:

- FastAPI (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)
- Streamlit UI: [http://localhost:8501](http://localhost:8501)

Без Docker:

```bash
# Терминал 1 — API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Терминал 2 — UI
API_URL=http://localhost:8000 streamlit run src/ui/app.py --server.port 8501
```

### 7.3 Эндпоинты API

| Метод | Путь | Тело запроса | Ответ |
|-------|------|-------------|-------|
| GET | `/health` | — | `{"status": "ok"}` |
| GET | `/model` | — | Метаданные: модель, sklearn версия, SEED, val/test RMSE |
| POST | `/predict` | `CampaignFeatures` (JSON) | `{"profit_prediction": float}` |
| POST | `/predict_batch` | `list[CampaignFeatures]` (JSON) | `{"predictions": [float, ...]}` |

### 7.4 Пример запроса

**curl:**

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

**Python:**

```python
import requests

resp = requests.post("http://localhost:8000/predict", json={...})
print(resp.json())  # {"profit_prediction": 2771.54}
```

### 7.5 Публичный деплой

В дополнение к локальному `docker compose` оба сервиса задеплоены на бесплатных PaaS-платформах. Это позволяет проверить `/health` и `/predict` снаружи без локальной среды.

| Платформа | Что хостит | Конфиг | Публичный URL |
|-----------|-----------|--------|---------------|
| **Render** | FastAPI | [render.yaml](../render.yaml) | `https://<service>.onrender.com` |
| **Hugging Face Spaces** | Streamlit UI | [deploy/hf-space/](../deploy/hf-space/) | `https://huggingface.co/spaces/<owner>/ad-profit-predictor` |

**Render — FastAPI backend.** Используется механизм **Blueprint**: в корне репозитория лежит [`render.yaml`](../render.yaml), описывающий веб-сервис (Python 3.10, plan `free`, регион Frankfurt, `buildCommand = pip install -r requirements.txt`, `startCommand = uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`, healthcheck по `/health`). Артефакт модели `models/final_cp2.joblib` (272 КБ) закоммичен в git и подтягивается вместе с кодом. На free-плане сервис засыпает после 15 минут неактивности и стартует за ~30 секунд при первом запросе.

**Hugging Face Spaces — Streamlit UI.** Space создан как отдельный git-репозиторий с тремя файлами из `deploy/hf-space/` (`app.py`, `requirements.txt`, `README.md` с обязательным YAML frontmatter `sdk: streamlit`). Адрес API передаётся через переменную окружения `API_URL`, которая задаётся в **Settings → Variables**. При нажатии **Predict profit** Streamlit делает `POST {API_URL}/predict` и отображает результат.

**CORS.** В FastAPI добавлен `CORSMiddleware` (`allow_origins=["*"]`), чтобы любой публичный фронт (HF Spaces, браузерные fetch, Postman) мог обращаться к API.

**Проверка снаружи:**

```bash
# health
curl https://<service>.onrender.com/health
# → {"status":"ok"}

# predict
curl -X POST https://<service>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
# → {"profit_prediction": 2771.54}
```

### 7.6 Скриншоты

*(Скриншоты будут добавлены после запуска сервисов: `report/images/api_swagger.png`, `report/images/api_predict_response.png`, `report/images/streamlit_ui.png`, `report/images/hf_space.png`, `report/images/render_dashboard.png`)*

### 7.7 Видео-демо

*(Ссылка на видео будет добавлена после записи. Сценарий: запуск `docker compose up` → Swagger /docs → curl /predict → Streamlit UI → публичный URL HF Space → curl на Render-сервис.)*

### 7.8 Известные ограничения

- **Covariate shift.** Модель обучена на time-based split: test-период попадает в «другой» сезон, из-за чего val RMSE (26 778) заметно лучше test RMSE (64 019). На новых данных из далёкого будущего ошибка может быть выше.
- **Неизвестные категории.** `OneHotEncoder(handle_unknown="ignore")` корректно обработает новые значения категориальных признаков — они будут закодированы нулями. Предсказание не упадёт, но может быть менее точным.
- **Модель не переобучается на новых данных** в текущей версии API — используется фиксированный артефакт `models/final_cp2.joblib`.

---

## 8. Заключение и выводы

- Поставлена корректная ML-задача: регрессия `profit` по ~35 фичам после удаления прямых компонентов таргета.
- Обоснованно выбрана метрика RMSE (MAE/R² — вспомогательные). Логарифмические метрики не применимы из-за отрицательных значений `profit`.
- Построен воспроизводимый пайплайн обработки данных (sklearn Pipeline + time-based split), закрывающий ключевые каналы утечек; добавлен отдельный baseline-пайплайн без календарного feature engineering.
- CP1: RandomForest — сильнейшая модель среди Ridge/Lasso/KNN/Linear baseline.
- CP2: полный цикл бустинга (LightGBM/XGBoost или sklearn-аналоги), `RandomizedSearchCV`, Stacking, PCA; финальная модель по val — **GradientBoostingRegressor** (sklearn), test RMSE **64 019**, R² **0.733**.
- Интерпретация: permutation importance показала, что `conversions`, `industry_vertical` и `income_bracket` — ключевые драйверы предсказания. Это согласуется с бизнес-логикой: объём конверсий напрямую связан с выручкой, а отрасль и уровень дохода аудитории определяют маржинальность.
- CP3: реализован деплой — FastAPI-сервис (`/predict`, `/predict_batch`) + Streamlit UI + Docker Compose; модель загружается один раз при старте, leakage-колонки исключены из API-схемы. Дополнительно: публичный деплой FastAPI на Render (`render.yaml`) и Streamlit UI на Hugging Face Spaces (`deploy/hf-space/`) — оба сервиса доступны снаружи по HTTPS.
- **Что можно улучшить:** добавить квантильную регрессию для оценки неопределённости предсказания; использовать монотонные ограничения в бустинге (`conversions ↑ → profit ↑`); реализовать переобучение модели на новых данных через отдельный эндпоинт; добавить мониторинг data drift в production.
