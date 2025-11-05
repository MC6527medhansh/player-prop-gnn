# Deployment Guide

## Overview
Complete deployment strategy from local development to production-ready container deployment. Designed for single-developer deployment without cloud infrastructure.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     Docker Host                          │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │  Streamlit   │    │   FastAPI    │    │  Redis   │  │
│  │  Dashboard   │───▶│     API      │───▶│  Cache   │  │
│  │   :8501      │    │    :8000     │    │  :6379   │  │
│  └──────────────┘    └───────┬──────┘    └──────────┘  │
│                              │                           │
│                              ▼                           │
│                      ┌──────────────┐                    │
│                      │  PostgreSQL  │                    │
│                      │   Database   │                    │
│                      │    :5432     │                    │
│                      └──────────────┘                    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │            Model Files (volume mount)            │   │
│  │  /models/tier1_v1.0.pkl, /models/tier2_v1.0.pt  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Container Strategy

### Development (docker-compose.yml)

**Services:**
1. **postgres**: PostgreSQL 14 database
2. **redis**: Redis 7 cache
3. **api**: FastAPI application
4. **dashboard**: Streamlit UI (optional)

**Benefits:**
- One-command setup: `docker-compose up`
- Isolated environment
- Easy to tear down and rebuild
- Same setup across machines

---

### Production (same structure, different config)

**Changes from development:**
- Environment variables for secrets
- Resource limits (CPU, memory)
- Persistent volumes for data
- Restart policies
- Health checks

---

## Docker Compose Configuration

### docker-compose.yml

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:14-alpine
    container_name: player_props_db
    environment:
      POSTGRES_USER: ${DATABASE_USER:-postgres}
      POSTGRES_PASSWORD: ${DATABASE_PASSWORD:-password}
      POSTGRES_DB: ${DATABASE_NAME:-football_props}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./data/schemas:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - prop_network
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: player_props_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    networks:
      - prop_network
    restart: unless-stopped
    command: redis-server --appendonly yes

  # FastAPI Application
  api:
    build:
      context: .
      dockerfile: deployment/docker/Dockerfile.api
    container_name: player_props_api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://${DATABASE_USER:-postgres}:${DATABASE_PASSWORD:-password}@postgres:5432/${DATABASE_NAME:-football_props}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - MODEL_DIR=/app/models
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./src:/app/src:ro
      - ./models:/app/models:ro
      - ./logs:/app/logs
      - ./data:/app/data:ro
    networks:
      - prop_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Streamlit Dashboard (Optional)
  dashboard:
    build:
      context: .
      dockerfile: deployment/docker/Dockerfile.api
    container_name: player_props_dashboard
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
      - DATABASE_URL=postgresql://${DATABASE_USER:-postgres}:${DATABASE_PASSWORD:-password}@postgres:5432/${DATABASE_NAME:-football_props}
    depends_on:
      - api
    volumes:
      - ./src:/app/src:ro
      - ./models:/app/models:ro
    command: streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
    networks:
      - prop_network
    restart: unless-stopped

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local

networks:
  prop_network:
    driver: bridge
```

---

## Dockerfile

### deployment/docker/Dockerfile.api

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p logs data/raw data/processed

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Deployment Steps

### Local Development

**1. Prerequisites:**
```bash
# Check installations
docker --version          # Should be >= 20.10
docker-compose --version  # Should be >= 1.29
```

**2. Environment Setup:**
```bash
# Copy environment template
cp .env.example .env

# Edit .env (optional, defaults work for local)
vim .env
```

**3. Start Services:**
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

**4. Initialize Database:**
```bash
# Run migrations
docker-compose exec api python src/data/init_db.py

# Verify tables created
docker-compose exec postgres psql -U postgres -d football_props -c "\dt"
```

**5. Load Initial Data:**
```bash
# Run data pipeline
docker-compose exec api python src/data/pipeline.py --collect --matches 100

# Verify data loaded
docker-compose exec postgres psql -U postgres -d football_props -c "SELECT COUNT(*) FROM player_match_stats;"
```

**6. Test API:**
```bash
# Health check
curl http://localhost:8000/health

# Test prediction (after model trained)
curl -X POST http://localhost:8000/predict/player \
  -H "Content-Type: application/json" \
  -d '{"player_id": 1, "match_id": 1}'
```

**7. Access Dashboard:**
```
Open browser: http://localhost:8501
```

**8. Shutdown:**
```bash
# Stop services
docker-compose down

# Remove volumes (data will be lost!)
docker-compose down -v
```

---

### Production Deployment

**1. Build Production Images:**
```bash
# Build with production tag
docker-compose -f docker-compose.prod.yml build

# Tag for registry (if using)
docker tag player-props-api:latest your-registry/player-props-api:v1.0
```

**2. Production Environment:**
```bash
# Create production .env
cp .env.example .env.prod

# Set production values
vim .env.prod
# Change: DATABASE_PASSWORD, REDIS_PASSWORD, LOG_LEVEL=WARNING
```

**3. Deploy:**
```bash
# Use production config
docker-compose -f docker-compose.prod.yml up -d

# Check health
curl http://your-server:8000/health
```

**4. Monitor:**
```bash
# View logs
docker-compose logs -f

# Check resource usage
docker stats

# Check disk usage
docker system df
```

---

## Model Versioning

### Filename Convention

```
models/
├── tier1_v1.0_2024-11-01.pkl      # Training date in filename
├── tier1_v1.1_2024-11-15.pkl
├── tier2_v1.0_2024-11-01.pt
└── tier2_v1.1_2024-11-15.pt
```

**Version Scheme:**
- **Major (X.0)**: Model architecture changes
- **Minor (1.X)**: Re-training on new data, hyperparameter tuning

---

### Model Loading Strategy

**Symlinks for current model:**
```bash
# Create symlinks to latest
ln -sf tier1_v1.1_2024-11-15.pkl models/tier1_current.pkl
ln -sf tier2_v1.1_2024-11-15.pt models/tier2_current.pt
```

**Application loads from symlink:**
```python
# src/api/main.py
model_path = Path(settings.MODEL_DIR) / "tier1_current.pkl"
model = load_model(model_path)
```

**Hot-swap (reload without downtime):**
```python
# Reload endpoint (for admin)
@app.post("/admin/reload-model")
async def reload_model(tier: str):
    global tier1_model, tier2_model
    
    if tier == "tier1":
        tier1_model = load_tier1_model()
    elif tier == "tier2":
        tier2_model = load_tier2_model()
    
    return {"status": "reloaded", "tier": tier}
```

**Rollback:**
```bash
# Rollback to previous version
ln -sf tier1_v1.0_2024-11-01.pkl models/tier1_current.pkl

# Restart API
docker-compose restart api
```

---

### Model Metadata Storage

**Store with model:**
```python
# When saving model
metadata = {
    'version': '1.1',
    'train_date': '2024-11-15',
    'n_players': 920,
    'n_matches': 450,
    'performance': {
        'ece_goals': 0.041,
        'ece_assists': 0.044,
        'brier_goals': 0.189,
        'brier_assists': 0.172
    },
    'convergence': 'converged',
    'runtime_minutes': 28,
    'features': ['position', 'form_5', 'opponent_strength', ...]
}

with open('models/tier1_v1.1_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

**Load metadata at startup:**
```python
def get_model_info():
    with open(model_path.with_suffix('.json')) as f:
        return json.load(f)
```

---

## Database Management

### Schema Migrations (Alembic)

**Initialize:**
```bash
# Inside API container
docker-compose exec api alembic init alembic
```

**Create Migration:**
```bash
# After changing schema
docker-compose exec api alembic revision --autogenerate -m "add_player_nationality"
```

**Apply Migration:**
```bash
# Upgrade to latest
docker-compose exec api alembic upgrade head

# Rollback one version
docker-compose exec api alembic downgrade -1
```

**Migration Example:**
```python
# alembic/versions/001_add_player_nationality.py
def upgrade():
    op.add_column('players',
        sa.Column('nationality', sa.String(50), nullable=True)
    )

def downgrade():
    op.drop_column('players', 'nationality')
```

---

### Backup & Restore

**Automated Backup (Daily):**
```bash
# deployment/scripts/backup_db.sh
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec -T postgres pg_dump -U postgres football_props | gzip > backups/db_${DATE}.sql.gz

# Keep last 7 days
find backups/ -name "db_*.sql.gz" -mtime +7 -delete
```

**Schedule with cron:**
```bash
# crontab -e
0 2 * * * /path/to/backup_db.sh
```

**Manual Backup:**
```bash
docker-compose exec postgres pg_dump -U postgres football_props > backup.sql
```

**Restore:**
```bash
# Stop API (prevent writes)
docker-compose stop api

# Restore
docker-compose exec -T postgres psql -U postgres football_props < backup.sql

# Restart API
docker-compose start api
```

---

## Monitoring & Alerting

### Logging

**Configuration (src/utils/logging_config.py):**
```python
import logging
from pythonjsonlogger import jsonlogger

def setup_logging(level=logging.INFO):
    logger = logging.getLogger()
    handler = logging.FileHandler('logs/app.log')
    
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    
    return logger
```

**Usage:**
```python
logger.info("Prediction made", extra={
    'player_id': 123,
    'match_id': 456,
    'prediction': 0.35,
    'inference_time_ms': 85
})
```

**Log Rotation:**
```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

---

### Metrics (Prometheus)

**Expose metrics:**
```python
# src/api/main.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Metrics
REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total prediction requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency',
    ['endpoint']
)

CACHE_HIT_RATE = Gauge(
    'cache_hit_rate',
    'Cache hit rate'
)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

**Dashboard (Grafana):**
```yaml
# monitoring/dashboards/grafana_config.json
{
  "dashboard": {
    "title": "Player Prop API",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          "rate(prediction_requests_total[5m])"
        ]
      },
      {
        "title": "Latency (p95)",
        "targets": [
          "histogram_quantile(0.95, prediction_latency_seconds)"
        ]
      }
    ]
  }
}
```

---

### Health Checks

**Application Health:**
```python
@app.get("/health")
async def health_check():
    checks = {
        'model_loaded': tier1_model is not None,
        'database_connected': await db.is_connected(),
        'redis_connected': redis_client.ping()
    }
    
    status = "healthy" if all(checks.values()) else "unhealthy"
    status_code = 200 if status == "healthy" else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            'status': status,
            'timestamp': datetime.utcnow().isoformat(),
            **checks
        }
    )
```

**External Monitoring:**
```bash
# Uptime monitoring (UptimeRobot, etc.)
curl -f http://your-server:8000/health || exit 1
```

---

### Alerting

**Alert Conditions:**
1. API health check fails (p95 latency > 200ms for 5 min)
2. Error rate > 5% for 5 min
3. Database connection failures
4. Disk usage > 80%
5. Model calibration drift (ECE > 0.10)

**Alert Channels:**
- Email
- Slack webhook (if team)
- SMS (for critical)

**Example (simple email alert):**
```python
# deployment/scripts/check_health.py
import requests
import smtplib

response = requests.get('http://localhost:8000/health')
if response.status_code != 200:
    send_alert_email('API Health Check Failed')
```

---

## Security Considerations

### Environment Variables

**Never commit secrets:**
```bash
# .env (in .gitignore)
DATABASE_PASSWORD=secure_password_here
REDIS_PASSWORD=another_secure_password
API_SECRET_KEY=random_secret_for_jwt
```

**Use in production:**
```bash
# Load from secure store
docker-compose --env-file .env.prod up -d
```

---

### API Security (Phase 11)

**Rate Limiting:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict/player")
@limiter.limit("100/minute")
async def predict_player(request: Request, ...):
    ...
```

**API Keys:**
```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
```

---

## Performance Optimization

### Resource Limits

**docker-compose.prod.yml:**
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

---

### Database Optimization

**Connection Pooling:**
```python
# src/config/database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
```

**Indexes (created in migrations):**
```sql
CREATE INDEX idx_stats_player_match ON player_match_stats(player_id, match_id);
CREATE INDEX idx_stats_player_date ON player_match_stats(player_id, (SELECT match_date FROM matches WHERE match_id = player_match_stats.match_id));
```

---

## Disaster Recovery

### Recovery Time Objective (RTO): 1 hour
### Recovery Point Objective (RPO): 24 hours

**Recovery Procedure:**

1. **Database Failure:**
```bash
# Restore from latest backup
docker-compose stop api
cat backups/db_latest.sql.gz | gunzip | docker-compose exec -T postgres psql -U postgres football_props
docker-compose start api
```

2. **Model Corruption:**
```bash
# Rollback to previous model
ln -sf tier1_v1.0_2024-11-01.pkl models/tier1_current.pkl
docker-compose restart api
```

3. **Complete System Failure:**
```bash
# Full rebuild
docker-compose down -v
docker-compose up -d
# Restore database
# Restore models
# Verify health
```

---

## Phase Completion Checklist

- [x] Docker Compose configuration complete
- [x] Dockerfile builds successfully
- [x] Can deploy with one command (`docker-compose up`)
- [x] All services have health checks
- [x] Model versioning strategy defined
- [x] Database backup/restore procedures documented
- [x] Monitoring metrics defined
- [x] Alerting thresholds set
- [x] Disaster recovery plan documented
- [x] Security considerations addressed

---

## Next Steps (Phase 10)

1. Test docker-compose on fresh machine
2. Run deployment walkthrough
3. Test backup/restore procedures
4. Configure monitoring
5. Document any deployment issues
6. Update this guide based on learnings