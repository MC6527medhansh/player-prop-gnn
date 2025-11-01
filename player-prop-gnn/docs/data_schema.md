# Data Schema Documentation

## Overview
Document your database schema here.

## Tables

### players
- player_id (PK)
- name
- position
- team_id (FK)

### matches
- match_id (PK)
- home_team_id (FK)
- away_team_id (FK)
- date
- competition

### player_stats
- stat_id (PK)
- player_id (FK)
- match_id (FK)
- goals
- assists
- shots_on_target
- cards

## Relationships
Document table relationships here.
