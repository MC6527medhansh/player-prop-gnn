#!/bin/bash
# Phase 8.3: Full Deployment Automation Script
# Usage: ./scripts/deploy.sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "========================================================"
echo "   🚀 PLAYER PROP ENSEMBLE - DEPLOYMENT PIPELINE"
echo "========================================================"

# 1. Environment Check
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found. Please create one from .env.example"
    exit 1
fi

# 2. Run Tests (Pre-flight Check)
echo "--------------------------------------------------------"
echo "🧪 Running Unit & Integration Tests..."
echo "--------------------------------------------------------"

# We use the API Dockerfile to run tests to ensure the environment matches production
# This requires a temporary build to run pytest inside the container
docker build -t player-prop-test -f deployment/docker/Dockerfile.api .
docker run --rm player-prop-test pytest tests/unit -v

if [ $? -eq 0 ]; then
    echo "✅ Tests Passed!"
else
    echo "❌ Tests Failed. Aborting deployment."
    exit 1
fi

# 3. Build Production Containers
echo "--------------------------------------------------------"
echo "🐳 Building Docker Containers..."
echo "--------------------------------------------------------"
docker-compose -f deployment/docker/docker-compose.yml build

# 4. Deploy
echo "--------------------------------------------------------"
echo "🚀 Deploying Services..."
echo "--------------------------------------------------------"
docker-compose -f deployment/docker/docker-compose.yml up -d

# 5. Wait for Health Check
echo "⏳ Waiting for API to initialize..."
sleep 10

API_URL="http://localhost:8000/health"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $API_URL)

if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "========================================================"
    echo "✅ DEPLOYMENT SUCCESSFUL"
    echo "   API is live at: http://localhost:8000"
    echo "   Docs available: http://localhost:8000/docs"
    echo "========================================================"
else
    echo "========================================================"
    echo "⚠️  DEPLOYMENT WARNING"
    echo "   API container is running but returned status $HTTP_STATUS"
    echo "   Check logs with: docker-compose logs api"
    echo "========================================================"
    exit 1
fi