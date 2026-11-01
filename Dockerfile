FROM python:3.12-slim

WORKDIR /app

# Install uv.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Make uv and the project virtual environment available.
ENV PATH="/app/.venv/bin:/root/.local/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8050

# Copy project files needed to install and start the dashboard.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY conf ./conf
COPY entrypoints ./entrypoints

# Install the locked project dependencies.
RUN uv sync --no-dev --frozen

EXPOSE 8050

# Start the dashboard by default.
CMD ["python", "entrypoints/app_ui.py"]