FROM python:3.11-slim

# Baked into the image so /health and logs report exactly which build is
# running. Set at build time: `docker build --build-arg MODEL_VERSION=1.2.0`.
# Individual model versions still come from config.yaml / MLflow / Supabase
# Storage (see registry.py) -- this tag is the *image's* version, i.e. which
# revision of the inference service (routes, registry logic, base configs)
# is deployed, not any one model's version.
ARG MODEL_VERSION=dev
ENV MODEL_VERSION=${MODEL_VERSION}

WORKDIR /app

COPY inference/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY inference/ /app/
COPY models/ /app/models/

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
