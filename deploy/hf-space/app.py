"""Streamlit UI for the Ad Campaign Profit Predictor (Hugging Face Space).

Entry point for the public Hugging Face Space.  Mirrors ``src/ui/app.py`` from the
parent project; the only difference is that ``API_URL`` defaults to the public
Render-hosted FastAPI instead of localhost.

Configure via Space → Settings → Variables:
    API_URL = https://ad-profit-api-<random>.onrender.com
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "https://ad-profit-api.onrender.com")

st.set_page_config(page_title="Ad Campaign Profit Predictor", layout="wide")
st.title("Ad Campaign Profit Predictor")
st.markdown(
    "Предскажите прибыль рекламной кампании по параметрам таргетинга, креатива и аукциона. "
    f"Backend: `{API_URL}`"
)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Параметры кампании")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Канал и размещение**")
        platform = st.selectbox("Platform", ["Facebook", "Google", "TikTok", "Instagram", "LinkedIn", "Twitter"])
        ad_placement = st.selectbox("Ad placement", ["Feed", "Search", "Stories", "Sidebar", "Banner"])
        device_type = st.selectbox("Device type", ["Mobile", "Desktop", "Tablet"])
        operating_system = st.selectbox("OS", ["Android", "iOS", "Windows", "macOS"])
        campaign_objective = st.selectbox(
            "Objective", ["Lead Generation", "Engagement", "Conversion", "Brand Awareness"]
        )
        industry_vertical = st.selectbox(
            "Industry", ["E-commerce", "Finance", "Gaming", "Healthcare", "Technology", "Education"]
        )
        budget_tier = st.selectbox("Budget tier", ["Low", "Medium", "High"])

    with c2:
        st.markdown("**Креатив**")
        creative_format = st.selectbox("Format", ["Text", "Image", "Video", "Carousel"])
        creative_size = st.selectbox("Size", ["728x90", "320x50", "300x250", "160x600", "970x250"])
        ad_copy_length = st.selectbox("Copy length", ["Short", "Medium", "Long"])
        creative_emotion = st.selectbox("Emotion", ["Curiosity", "Neutral", "Excitement", "Urgency", "Trust"])
        has_call_to_action = st.checkbox("Has CTA", value=True)
        creative_age_days = st.number_input("Creative age (days)", min_value=0, max_value=1000, value=30)

    with c3:
        st.markdown("**Аудитория**")
        target_audience_age = st.selectbox("Age group", ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"])
        target_audience_gender = st.selectbox("Gender", ["Male", "Female", "All"])
        audience_interest_category = st.selectbox(
            "Interest", ["Shoppers", "Business Professionals", "Tech Enthusiasts", "Gamers"]
        )
        income_bracket = st.selectbox("Income", ["<$50K", "$50K-$100K", ">$100K"])
        purchase_intent_score = st.selectbox("Purchase intent", ["Low", "Medium", "High"])
        retargeting_flag = st.checkbox("Retargeting", value=False)

    st.markdown("---")
    c4, c5 = st.columns(2)

    with c4:
        st.markdown("**Время и сезонность**")
        start_date = st.date_input("Start date")
        day_of_week = st.selectbox(
            "Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        quarter = st.number_input("Quarter", min_value=1, max_value=4, value=2)
        hour_of_day = st.number_input("Hour of day", min_value=0, max_value=23, value=14)
        campaign_day = st.number_input("Campaign day", min_value=1, max_value=365, value=10)

    with c5:
        st.markdown("**Метрики трафика**")
        quality_score = st.number_input("Quality score", min_value=1, max_value=10, value=7)
        impressions = st.number_input("Impressions", min_value=0, value=50000)
        clicks = st.number_input("Clicks", min_value=0, value=500)
        conversions = st.number_input("Conversions", min_value=0, value=20)
        bounce_rate = st.number_input("Bounce rate (%)", min_value=0.0, max_value=100.0, value=45.0)
        avg_session_duration_seconds = st.number_input("Avg session duration (s)", min_value=0.0, value=120.0)
        pages_per_session = st.number_input("Pages per session", min_value=0.0, value=3.5)
        ctr = st.number_input("CTR (%)", min_value=0.0, value=1.0)
        conversion_rate = st.number_input("Conversion rate (%)", min_value=0.0, value=4.0)

    predict_clicked = st.button("Predict profit", type="primary", use_container_width=True)

with col_right:
    st.subheader("Результат")

    if predict_clicked:
        payload = {
            "start_date": str(start_date),
            "campaign_objective": campaign_objective,
            "platform": platform,
            "ad_placement": ad_placement,
            "device_type": device_type,
            "operating_system": operating_system,
            "creative_format": creative_format,
            "creative_size": creative_size,
            "ad_copy_length": ad_copy_length,
            "creative_emotion": creative_emotion,
            "target_audience_age": target_audience_age,
            "target_audience_gender": target_audience_gender,
            "audience_interest_category": audience_interest_category,
            "income_bracket": income_bracket,
            "purchase_intent_score": purchase_intent_score,
            "day_of_week": day_of_week,
            "industry_vertical": industry_vertical,
            "budget_tier": budget_tier,
            "has_call_to_action": has_call_to_action,
            "retargeting_flag": retargeting_flag,
            "creative_age_days": creative_age_days,
            "quarter": float(quarter),
            "hour_of_day": float(hour_of_day),
            "campaign_day": float(campaign_day),
            "quality_score": float(quality_score),
            "impressions": float(impressions),
            "clicks": float(clicks),
            "conversions": float(conversions),
            "bounce_rate": bounce_rate,
            "avg_session_duration_seconds": avg_session_duration_seconds,
            "pages_per_session": pages_per_session,
            "CTR": ctr,
            "conversion_rate": conversion_rate,
        }

        try:
            with st.spinner("Wake up the API (Render free tier cold start ~30 s)…"):
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            profit = result["profit_prediction"]

            if profit >= 0:
                st.success(f"Predicted profit: **${profit:,.2f}**")
            else:
                st.error(f"Predicted profit: **${profit:,.2f}** (убыточная кампания)")
        except requests.ConnectionError:
            st.error(
                f"Не удалось подключиться к API. Backend: {API_URL}. "
                "На бесплатном плане Render сервис засыпает после 15 минут — "
                "первое обращение может занять ~30 секунд."
            )
        except requests.Timeout:
            st.error("API не ответил за 60 секунд (вероятно, cold start). Попробуйте ещё раз.")
        except requests.HTTPError as e:
            st.error(f"Ошибка API: {e.response.status_code} — {e.response.text}")
        except Exception as e:
            st.error(f"Ошибка: {e}")

    st.markdown("---")
    st.subheader("Важность признаков (Permutation Importance)")
    st.caption("Топ-10 признаков по росту RMSE при перемешивании (на val-выборке)")

    importance_data = {
        "Признак": [
            "conversions", "industry_vertical", "income_bracket",
            "creative_age_days", "clicks", "avg_session_duration",
            "campaign_objective", "hour_of_day", "CTR", "creative_emotion",
        ],
        "Importance (mean RMSE increase)": [
            76500.0, 7514.0, 4785.9,
            661.4, 542.6, 296.2,
            289.2, 277.1, 231.5, 226.0,
        ],
    }
    df_imp = pd.DataFrame(importance_data)
    st.bar_chart(df_imp.set_index("Признак"))

    st.markdown("---")
    st.subheader("О модели")
    st.markdown(
        """
- **Модель:** GradientBoostingRegressor (sklearn)
- **Эксперимент:** exp03_tree_tuned (CP2)
- **Val RMSE:** 26 778
- **Test RMSE:** 64 019 | **R²:** 0.733
- **SEED:** 42
"""
    )
