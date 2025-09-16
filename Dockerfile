FROM python:3.11.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLBACKEND=Agg
ENV TZ=UTC

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy requirements and install Python dependencies
COPY web/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Change to web directory
WORKDIR /app/web

# Expose port
EXPOSE 8080

# Run the application with minimal resources for Render free tier
CMD ["sh", "-c", "python start_render.py && gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --worker-class gthread --threads 1 --timeout 120 app:app"]


