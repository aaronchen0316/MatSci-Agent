FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704
COPY --from=ghcr.io/astral-sh/uv:0.8.17@sha256:e4644cb5bd56fdc2c5ea3ee0525d9d21eed1603bccd6a21f887a938be7e85be1 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    MATSCI_REQUIRE_API_TOKEN=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY models/pretrained ./models/pretrained

ARG WITH_MATGL=0
RUN if [ "$WITH_MATGL" = "1" ]; then \
      uv sync --locked --no-dev --extra mp --extra matgl; \
    else \
      uv sync --locked --no-dev --extra mp; \
    fi \
    && useradd --create-home --uid 10001 matsci \
    && chown -R matsci:matsci /app

USER matsci
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"
CMD ["uvicorn", "matsci_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
