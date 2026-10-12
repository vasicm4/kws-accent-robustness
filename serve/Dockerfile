# Inference image: ONNX Runtime + FastAPI only, no torch.
#
# Export a model first, then build from the repo root:
#   uv run --extra serve python scripts/export_onnx.py runs/<run>
#   docker build -f serve/Dockerfile --build-arg MODEL=runs/<run>/model.onnx -t kws-serve .
#   docker run --rm -p 8000:8000 kws-serve
#   curl --data-binary @clip.wav -H "Content-Type: audio/wav" localhost:8000/predict
FROM python:3.12-slim

# The soundfile wheel bundles libsndfile, so no apt packages are needed.
RUN pip install --no-cache-dir \
    "onnxruntime==1.31.0" "numpy>=2" "soundfile>=0.12" "fastapi==0.141.1" "uvicorn==0.53.0"

WORKDIR /app
ARG MODEL
COPY serve/app.py /app/serve/app.py
COPY ${MODEL} /app/model.onnx

ENV MODEL_PATH=/app/model.onnx
RUN useradd --create-home kws
USER kws
EXPOSE 8000
CMD ["uvicorn", "serve.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
