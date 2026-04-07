FROM python:3.11-slim

WORKDIR /env

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

ENTRYPOINT ["python", "docker/entrypoint.sh"]
