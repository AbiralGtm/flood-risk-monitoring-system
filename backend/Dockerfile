FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY alembic.ini .
COPY migrations ./migrations

# Model weights (model_random_forest.joblib alone is ~824MB) are NOT baked into
# this image -- that would make every build/push/pull move nearly a gigabyte of
# data unnecessarily. Mount the models/ directory into the container at runtime
# instead (see docker-compose.yml's volumes entry). If you're running this image
# standalone without compose, pass the equivalent yourself:
#   docker run -v $(pwd)/models:/app/models:ro -p 8000:8000 <image>
RUN mkdir -p ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
