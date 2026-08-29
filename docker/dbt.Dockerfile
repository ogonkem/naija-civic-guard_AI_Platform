# One-shot dbt runner for the analytics models (dbt/).
#   docker compose --profile analytics run --rm dbt build
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 DBT_PROFILES_DIR=/dbt

RUN pip install "dbt-postgres>=1.8,<1.10"

WORKDIR /dbt
COPY dbt/ .

ENTRYPOINT ["dbt"]
CMD ["build"]
