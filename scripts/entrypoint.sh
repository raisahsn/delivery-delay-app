#!/bin/sh
# entrypoint.sh
# ---------------
# 1) If model/model.pkl is missing but MODEL_URL is set (e.g. on Railway,
#    which does NOT pull Git LFS files during build), download it before
#    starting the app.
# 2) Bind Streamlit to $PORT if the platform provides one (Railway, Render,
#    etc. assign a dynamic port), falling back to 8501 for local/Docker use.
set -e

MODEL_DIR="/app/model"
MODEL_PATH="${MODEL_DIR}/model.pkl"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ]; then
  if [ -n "$MODEL_URL" ]; then
    echo "model.pkl not found locally — downloading from MODEL_URL..."
    curl -fsSL "$MODEL_URL" -o "$MODEL_PATH"
  else
    echo "WARNING: model/model.pkl not found and MODEL_URL is not set."
    echo "The app will start but show a 'model not found' message until"
    echo "you provide the model (see README.md - Railway deployment)."
  fi
fi

if [ -n "$CONFIG_URL" ] && [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "Downloading config.json from CONFIG_URL..."
  curl -fsSL "$CONFIG_URL" -o "${MODEL_DIR}/config.json" || \
    echo "WARNING: failed to download config.json (non-fatal, continuing)."
fi

exec streamlit run app/streamlit_app.py \
  --server.port="${PORT:-8501}" \
  --server.address=0.0.0.0 \
  --server.headless=true
