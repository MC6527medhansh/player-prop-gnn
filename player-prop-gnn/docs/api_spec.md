# API Specification

## Endpoints

### GET /health
Health check endpoint.

### POST /predict/player
Predict props for a single player.

**Request:**
```json
{
  "player_id": "123",
  "match_id": "456",
  "props": ["goals", "assists"]
}
```

**Response:**
```json
{
  "predictions": {
    "goals": {"probability": 0.23, "credible_interval": [0.15, 0.32]},
    "assists": {"probability": 0.18, "credible_interval": [0.10, 0.28]}
  }
}
```

### POST /predict/match
Predict props for all players in a match.
