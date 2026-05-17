# Деплой Streamlit UI на Hugging Face Spaces

## 1. Создать Space

1. Открой <https://huggingface.co/new-space>.
2. Поля:
   - **Owner** — твой профиль.
   - **Space name** — `ad-profit-predictor` (или другое).
   - **License** — MIT.
   - **Select SDK** — **Streamlit**.
   - **Hardware** — CPU basic (free).
   - **Visibility** — Public.
3. Нажми **Create Space**. Получишь пустой репо вида `https://huggingface.co/spaces/<owner>/ad-profit-predictor`.

## 2. Загрузить файлы

Самый простой путь — через UI: на странице Space нажми **Files → Add file → Upload files** и загрузи три файла из этой папки:

- `app.py`
- `requirements.txt`
- `README.md`

После загрузки Space автоматически начнёт сборку (статус **Building**), затем перейдёт в **Running**.

Альтернатива — через git:

```bash
git clone https://huggingface.co/spaces/<owner>/ad-profit-predictor
cd ad-profit-predictor
cp /path/to/repo/deploy/hf-space/{app.py,requirements.txt,README.md} .
git add -A
git commit -m "feat: initial deploy of streamlit ui"
git push
```

## 3. Задать переменную окружения `API_URL`

После того как FastAPI развёрнут на Render (см. `render.yaml`):

1. Открой Space → **Settings → Variables and secrets** → **New variable**.
2. **Name** — `API_URL`.
3. **Value** — `https://<твой-render-сервис>.onrender.com` (без слэша на конце).
4. Сохрани. Space автоматически перезапустится с новым окружением.

Если не задать — UI обратится по дефолту к `https://ad-profit-api.onrender.com` (плейсхолдер, заменишь на свой).

## 4. Проверить

Откройте `https://huggingface.co/spaces/<owner>/ad-profit-predictor` — должна появиться форма. Заполнить и нажать **Predict profit**. Первый запрос займёт ~30 секунд (cold start Render). Дальше — мгновенно.
