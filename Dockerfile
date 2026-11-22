FROM python:3.12-slim

# Set environment variables for Python and Streamlit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Create non-root application user
RUN useradd -m -u 1000 appuser

# Install dependencies and project package
# (hatchling build backend requires pyproject.toml, README.md, and text2sql/ package sources)
COPY --chown=appuser:appuser pyproject.toml README.md ./
COPY --chown=appuser:appuser text2sql/ text2sql/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy application modules, verified demo cache, database, and eval benchmark results
COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser scripts/ scripts/
COPY --chown=appuser:appuser clinic.db clinic.db
COPY --chown=appuser:appuser data_checksum.txt data_checksum.txt
COPY --chown=appuser:appuser eval/results/latest.json eval/results/latest.json

# Create logs directory and ensure non-root ownership across /app
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose default Streamlit port
EXPOSE 8501

# Healthcheck testing Streamlit core health endpoint via Python standard library
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Launch Streamlit app on 0.0.0.0:8501
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
