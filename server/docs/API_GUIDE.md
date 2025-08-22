# Zappy REST API Documentation

This document describes the REST API endpoints implemented for the Zappy server that are compatible with the GUI client.

## Base URL
```
http://localhost:5000/api
```

## Response Format
All endpoints return JSON responses in the format:
```json
{
  "success": true|false,
  "data": <response_data>,
  "error": "<error_message>" // only if success is false
}
```

## Endpoints

### Server Management

#### GET /health
Health check endpoint.
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "server_ticks": 0,
    "server_size": 10,
    "teams": 2,
    "players": 2,
    "timestamp": 1692034567.123
  }
}
```

#### GET /server
Get complete server state including players, teams, map, and statistics.

#### GET /server/stats
Get server statistics only.

#### POST /server/step
Advance the server by one step/tick.

#### POST /server/reset
Reset the server state.
```json
// Request body (optional)
{
  "nb_eggs": 5,
  "teams": ["team1", "team2"]
}
```

### Player Management

#### GET /players
Get all players with optional filtering.
Query parameters:
- `team`: Filter by team name
- `alive`: Filter by alive status (true/false)

#### GET /players/{id}
Get specific player by ID.

#### PUT /players/{id}
Update a player property.
```json
// Request body
{
  "property": "level",
  "value": 3
}
```

#### PUT /players/bulk
Bulk update multiple players.
```json
// Request body
{
  "operations": [
    {
      "player_id": 1,
      "property": "level",
      "value": 3
    },
    {
      "player_id": 2,
      "property": "is_alive",
      "value": false
    }
  ]
}
```

### Convenience Player Endpoints

#### POST /players/{id}/kill
Kill a player (sets is_alive to false).

#### POST /players/{id}/resurrect
Resurrect a player (sets is_alive to true).

#### PUT /players/{id}/level
Set player level.
```json
// Request body
{
  "level": 3
}
```

#### PUT /players/{id}/position
Set player position.
```json
// Request body
{
  "position": [x, y]
}
```

#### PUT /players/{id}/inventory
Set player inventory resource amount.
```json
// Request body
{
  "resource": "food",
  "amount": 10
}
```

### Team Management

#### GET /teams
Get all teams.

#### GET /teams/{name}
Get specific team data.

#### POST /teams
Add a new team.
```json
// Request body
{
  "name": "new_team"
}
```

#### DELETE /teams/{name}
Remove a team.

### Map Management

#### GET /map
Get map data including size and tile information.

#### PUT /map/tile/{x}/{y}
Update map tile resources.
```json
// Request body - can update multiple resources at once
{
  "food": 5,
  "linemate": 2,
  "deraumere": 1
}

// Or single resource (legacy format)
{
  "resource": "food",
  "amount": 5
}
```

### Search

#### POST /search/players
Search players with criteria.
```json
// Request body
{
  "team": "team1",
  "alive": true,
  "min_level": 2,
  "max_level": 5
}
```

### Real-time Events

#### GET /events
Server-Sent Events stream for real-time updates.

Event types:
- `player_update`: Player data changed
- `server_update`: Server statistics updated
- `map_update`: Map tile changed
- `keepalive`: Periodic keepalive message

### Documentation

#### GET /docs
Get API documentation with all available endpoints.

## Event Stream Format

The `/events` endpoint uses Server-Sent Events (SSE) with the following format:

```
event: player_update
data: {"id": 1, "level": 3, ...}

event: server_update
data: {"total_players": 4, "alive_players": 3, ...}

event: map_update
data: {"x": 0, "y": 0, "resources": {...}}

event: keepalive
data: {"timestamp": 1692034567.123}
```

## Error Responses

All endpoints may return error responses:
```json
{
  "success": false,
  "error": "Error description"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (resource doesn't exist)
- `500`: Internal Server Error

## CORS Support

The API includes CORS headers to allow cross-origin requests from web applications.

## Usage Example

```python
import requests

# Health check
response = requests.get('http://localhost:5000/api/health')
print(response.json())

# Get all players
players = requests.get('http://localhost:5000/api/players')
print(players.json())

# Update player level
requests.put(
    'http://localhost:5000/api/players/1',
    json={'property': 'level', 'value': 3}
)

# Kill a player
requests.post('http://localhost:5000/api/players/1/kill')
```

## Running the Server

```bash
# From the python-server directory
python -m api.rest.server --host localhost --port 5000

# Or run the test script
python test_api.py
```
