FROM python:3.11.9-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_BUILD_ISOLATION=1

WORKDIR /app

# Copy requirements (root includes web/requirements.txt)
COPY requirements.txt ./
COPY web/requirements.txt web/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Render will inject PORT; bind to it
EXPOSE 8080
CMD ["gunicorn", "-w", "4", "-k", "gthread", "-b", "0.0.0.0:${PORT}", "web.app:app"]


