# Helix Collective v14.5 - Quantum Handshake
# Dockerfile - Railway Deployment Configuration

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p Helix/state Helix/commands Helix/ethics Shadow/manus_archive

# Create default state files
RUN echo '{"zoom":1.0228,"harmony":0.355,"resilience":1.1191,"prana":0.5175,"drishti":0.5023,"klesha":0.010}' > Helix/state/ucf_state.json
RUN echo '{"timestamp":"init","status":"starting"}' > Helix/state/heartbeat.json
RUN echo '{}' > Helix/commands/manus_directives.json

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port (Railway will override with $PORT)
EXPOSE 8000

# Run the application
CMD ["python", "backend/main.py"]

