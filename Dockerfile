
FROM python:3.11-slim


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0


WORKDIR /app


COPY requirements.txt .
RUN pip install -r requirements.txt


COPY . .


EXPOSE 5000


CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "200", "app:app"]
