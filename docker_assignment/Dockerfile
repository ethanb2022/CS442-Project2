FROM openjdk:11

COPY --from=python:2.7.17 / /

LABEL Maintainer="Ethan"

WORKDIR /home

COPY  requirements.txt .

RUN pip install -r requirements.txt

COPY project2.py .

COPY winequality-white.csv .

CMD ["python", "project2.py"]
