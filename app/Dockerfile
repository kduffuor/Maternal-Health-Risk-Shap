FROM python:3.9-slim-buster
RUN pip install -U pip
RUN pip install pipenv
WORKDIR /app
COPY ["app/Pipfile", "app/Pipfile.lock", "./"]
RUN pipenv install --system --skip-lock
COPY ["app/*.py", "./"]
ADD ["app/static", "./static"]
ADD ["app/templates", "./templates"]
ADD ["app/tests", "./tests"]
COPY ["data/data.csv", "./data/data.csv"]
RUN python retrain.py