FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY _wsgi.py .
COPY model.py .
COPY config.yaml .

EXPOSE 9090

CMD ["gunicorn", "_wsgi:app", "--bind", "0.0.0.0:9090", "--workers", "1", "--timeout", "120"]