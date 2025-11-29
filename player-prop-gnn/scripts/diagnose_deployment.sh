#!/bin/bash
# Phase 8.3 Diagnostic Tool
# Run this to find out WHY the deployment is failing.

echo "=== 🔍 DEPLOYMENT DIAGNOSTICS ==="

# 1. Check if containers are actually running
echo -e "\n[1] Checking Docker Container Status..."
RUNNING=$(docker ps --format "{{.Names}} - {{.Status}}" | grep player-prop)

if [ -z "$RUNNING" ]; then
    echo "❌ NO CONTAINERS RUNNING!"
    echo "Listing all containers (including stopped):"
    docker ps -a --filter "name=player-prop" --format "table {{.Names}}\t{{.Status}}\t{{.State}}"
else
    echo "✅ Running Containers:"
    echo "$RUNNING"
fi

# 2. Check API Logs for Startup Errors
echo -e "\n[2] Fetching API Logs (Last 50 lines)..."
# Tries to find the api container name dynamically, usually ends in -api-1 or similar
API_CONTAINER=$(docker ps -a -q -f name=api | head -n 1)

if [ -z "$API_CONTAINER" ]; then
    echo "❌ Could not find API container."
else
    docker logs --tail 50 $API_CONTAINER
fi

# 3. Check Network Accessibility
echo -e "\n[3] Testing Endpoint Connectivity..."
CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)

if [ "$CODE" == "200" ]; then
    echo "✅ API is responding (Status 200)"
    echo "Response Payload:"
    curl -s http://localhost:8000/health | python3 -m json.tool
else
    echo "❌ API failed to respond (Status: $CODE)"
fi

echo -e "\n=== DIAGNOSTICS COMPLETE ==="