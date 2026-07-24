#!/bin/sh
# entrypoint.sh
# ---------------
# 1) If MODEL_URL is set, ALWAYS download it and overwrite model/model.pkl.
#    This is deliberate: on platforms that don't pull Git LFS during
#    checkout (e.g. Railway), the model.pkl baked into the image is just
#    a small LFS *pointer* text file (not the real binary) — even though
#    it "exists". Checking existence alone would wrongly skip the
#    download and leave a broken pointer file in place. Downloading
#    unconditionally when MODEL_URL is set avoids this trap entirely.
# 2) Bind Streamlit to $PORT if the platform provides one (Railway, Render,
#    etc. assign a dynamic port), falling back to 8501 for local/Docker use.
set -e

MODEL_DIR="/app/model"
MODEL_PATH="${MODEL_DIR}/model.pkl"

mkdir -p "$MODEL_DIR"

if [ -n "$MODEL_URL" ]; then
  echo "MODEL_URL is set — downloading model.pkl (overwriting any existing file)..."
  curl -fsSL "$MODEL_URL" -o "$MODEL_PATH"
  echo "Downloaded model.pkl ($(wc -c < "$MODEL_PATH") bytes)."
elif [ ! -f "$MODEL_PATH" ]; then
  echo "WARNING: model/model.pkl not found and MODEL_URL is not set."
  echo "The app will start but show a 'model not found' message until"
  echo "you provide the model (see README.md - Railway deployment)."
fi

if [ -n "$CONFIG_URL" ]; then
  echo "Downloading config.json from CONFIG_URL..."
  curl -fsSL "$CONFIG_URL" -o "${MODEL_DIR}/config.json" || \
    echo "WARNING: failed to download config.json (non-fatal, continuing)."
fi

exec streamlit run app/streamlit_app.py \
  --server.port="${PORT:-8501}" \
  --server.address=0.0.0.0 \
  --server.headless=true