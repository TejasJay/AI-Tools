FROM python:3.13-slim

WORKDIR /app

# ✅ Add git so pip can install from GitHub
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
