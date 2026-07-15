FROM mambaorg/micromamba:2.3.3

USER root
WORKDIR /app
COPY service/environment.yml /tmp/environment.yml
RUN micromamba create -y -n redox-rfb-service -f /tmp/environment.yml \
    && micromamba clean --all --yes

COPY . /app
RUN micromamba run -n redox-rfb-service pip install --no-deps . \
    && chown -R $MAMBA_USER:$MAMBA_USER /app

ENV PATH=/opt/conda/envs/redox-rfb-service/bin:$PATH \
    REDOX_MODEL_DIR=/app/models \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER $MAMBA_USER

EXPOSE 7860
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
