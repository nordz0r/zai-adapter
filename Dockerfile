FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

RUN useradd --system --uid 10001 --no-create-home zai \
    && chown -R zai:zai /app
USER zai

ENV PORT=8100
EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
    CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8100')+'/healthz',timeout=3)"

CMD ["python", "-c", "import os,uvicorn; uvicorn.run('zai_adapter.app:app', host='0.0.0.0', port=int(os.environ.get('PORT','8100')))"]
