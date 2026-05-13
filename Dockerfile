# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --prefer-offline

COPY index.html index.tsx index.css tsconfig.json vite.config.ts ./
COPY App.tsx ./
COPY src/ ./src/

RUN npm run build

# ── Stage 2: Python backend + serve built frontend ────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY setup.py README.md ./
COPY physicore/ ./physicore/
RUN pip install --no-cache-dir -e ".[api]"

# Copy built frontend into the server's static directory
COPY --from=frontend-builder /app/dist ./dist/

# Copy deployment assets
COPY firestore.rules vercel.json ./
COPY *.yaml ./

# Registry volume mount point — learned params persist across restarts
RUN mkdir -p /root/.physicore/registry

EXPOSE 8000

ENV PHYSICORE_HOST=0.0.0.0
ENV PHYSICORE_PORT=8000

# Serve the FastAPI backend; frontend is served as static files from /dist
CMD ["uvicorn", "physicore.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
