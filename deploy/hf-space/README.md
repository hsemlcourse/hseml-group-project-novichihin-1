---
title: Ad Campaign Profit Predictor
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.44.1
app_file: app.py
pinned: false
license: mit
---

# Ad Campaign Profit Predictor — Streamlit UI

Frontend для модели предсказания прибыли digital-рекламной кампании.

- **Бэкенд (FastAPI):** Render, public URL задаётся через переменную окружения `API_URL` в Space → Settings → Variables.
- **Backend by default:** `https://ad-profit-api.onrender.com`
- **Source repo:** <https://github.com/hsemlcourse/hseml-group-project-novichihin>

Если API в сонном состоянии (Render free tier), первый запрос занимает ~30 секунд (cold start).
