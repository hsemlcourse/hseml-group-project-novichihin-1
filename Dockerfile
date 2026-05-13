# Воспроизводимая среда (CP2): Python 3.10 как в CI, OpenMP для LightGBM
FROM python:3.10-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# По умолчанию прогон тестов; для ноутбука: docker compose run --rm app papermill ...
CMD ["pytest", "-q"]
