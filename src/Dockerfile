#Base image
FROM python:3.9-slim-buster

# Maintainer information
LABEL maintainer="Sharika Mahadevan <sharikamahadevan2023@u.northwestern.edu>"

#Set working directory
WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src /app/src

COPY tests /app/tests

COPY config /app/config

COPY pipeline.py .

CMD ["bash", "-c", "pytest tests/testing_generate_features.py && python pipeline.py"]