FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Build-time switch between a lean production image (runtime deps only)
# and a development image that also has the test toolchain installed, so
# `docker compose exec backend pytest -q` works without rebuilding.
# Defaults to false so a normal/production build never pulls in pytest,
# reportlab, etc.
ARG INSTALL_DEV=false

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./

RUN pip install --no-cache-dir --upgrade -r requirements.txt \
    && if [ "$INSTALL_DEV" = "true" ]; then \
        pip install --no-cache-dir --upgrade -r requirements-dev.txt; \
    fi

COPY app ./app
COPY tests ./tests
COPY pytest.ini .
COPY alembic.ini .
COPY alembic ./alembic

EXPOSE 8000

CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
