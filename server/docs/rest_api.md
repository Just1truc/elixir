# Zappy Server REST API Documentation

## Overview

The Zappy REST API provides HTTP endpoints for interacting with the Zappy game server. It allows external applications to monitor game state, retrieve data, modify server state, and receive real-time events.

## Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-cors requests
```

### 2. Start the REST API Server

```bash
python rest_api.py
```

### 3. Test the API

```bash
# Health check
curl http://localhost:5000/api/health

# Get server statistics
curl http://localhost:5000/api/server/stats

# Run comprehensive tests
python rest_api_test.py
```

## API Reference

### Base URL
```
http://localhost:5000/api
```

### Response Format

All responses follow this format:
```json
{
  "success": true,
  "data": { ... },
  "error": "error message (if success=false)"
}
```

---

## Server Endpoints

### GET /server
Get complete server state including all players, map data, and game information.

**Response:**
```json
{
  "success": true,
  "data": {
    "ticks": 42,
    "teams": ["team1", "team2"],
    "players": [...],
    "map_size": 10,
    "map_tiles": [...],
    "eggs_tiles": {...},
    "incanted_tiles": {...},
    "total_resources": {...}
  }
}
```

### GET /server/stats
Get server statistics and analytics.

**Response:**
```json
{
  "success": true,
  "data": {
    "total_ticks": 42,
    "total_players": 4,
    "alive_players": 3,
    "teams_count": 2,
    "incantations_in_progress": 0,
    "total_eggs": 6,
    "average_player_level": 1.5,
    "highest_level": 3,
    "team_distribution": {"team1": 2, "team2": 2},
    "player_resources": {...},
    "map_resources": {...}
  }
}
```

### POST /server/step
Advance the server by one tick.

**Response:**
```json
{
  "success": true,
  "data": {
    "tick": 43,
    "message": "Server stepped successfully"
  }
}
```

### POST /server/reset
Reset the server state.

**Request Body:**
```json
{
  "nb_eggs": 5,
  "teams": ["red", "blue"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Server reset successfully",
    "ticks": 0,
    "teams": ["red", "blue"],
    "nb_eggs": 5
  }
}
```

---

## Player Endpoints

### GET /players
Get all players with optional filtering.

**Query Parameters:**
- `team` (string): Filter by team name
- `alive` (boolean): Filter by alive status ("true"/"false")

**Examples:**
```bash
# Get all players
curl http://localhost:5000/api/players

# Get alive players from team1
curl "http://localhost:5000/api/players?team=team1&alive=true"
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": 0,
      "team": "team1",
      "position": [5, 3],
      "level": 2,
      "orientation": 1,
      "inventory": {"food": 10, "linemate": 2},
      "is_alive": true,
      "lifetime_tick": 120,
      "remain_tick": 0,
      "current_command": null,
      "command_queue_length": 0
    }
  ],
  "count": 1
}
```

### GET /players/{id}
Get specific player data.

**Response:**
```json
{
  "success": true,
  "data": {
    "id": 0,
    "team": "team1",
    "position": [5, 3],
    "level": 2,
    "orientation": 1,
    "inventory": {"food": 10, "linemate": 2},
    "is_alive": true,
    "lifetime_tick": 120,
    "remain_tick": 0,
    "current_command": null,
    "command_queue_length": 0
  }
}
```

### PUT /players/{id}
Update a player property.

**Request Body:**
```json
{
  "property": "level",
  "value": 5
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Updated player 0 level to 5"
  }
}
```

---

## Team Endpoints

### GET /teams
Get all teams with their data.

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "team": "team1",
      "players": [...],
      "eggs": [[2, 3], [7, 8]],
      "alive_count": 2,
      "total_players": 2,
      "total_level": 4,
      "average_level": 2.0
    }
  ],
  "count": 2
}
```

### GET /teams/{name}
Get specific team data.

**Response:**
```json
{
  "success": true,
  "data": {
    "team": "team1",
    "players": [...],
    "eggs": [[2, 3], [7, 8]],
    "alive_count": 2,
    "total_players": 2,
    "total_level": 4,
    "average_level": 2.0
  }
}
```

### POST /teams
Add a new team.

**Request Body:**
```json
{
  "name": "team3"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Team team3 added successfully"
  }
}
```

### DELETE /teams/{name}
Remove a team and all its players.

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Team team3 removed successfully"
  }
}
```

---

## Map Endpoints

### GET /map
Get map data and statistics.

**Response:**
```json
{
  "success": true,
  "data": {
    "size": 10,
    "tiles": [
      [{"food": 2, "linemate": 1}, {"food": 0, "linemate": 0}, ...],
      [...]
    ],
    "eggs_tiles": {"team1": [[2, 3]], "team2": [[7, 8]]},
    "incanted_tiles": {},
    "total_resources": {"food": 150, "linemate": 80, ...},
    "resource_density": {"food": 1.5, "linemate": 0.8, ...}
  }
}
```

### PUT /map/tile/{x}/{y}
Update resources on a specific map tile.

**Request Body:**
```json
{
  "resource": "food",
  "amount": 10
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Updated tile (5, 5) food to 10"
  }
}
```

---

## Search Endpoints

### POST /search/players
Search for players matching specific criteria.

**Request Body:**
```json
{
  "team": "team1",
  "is_alive": true,
  "level": "lambda x: x >= 3"
}
```

**Response:**
```json
{
  "success": true,
  "data": [...],
  "count": 2
}
```

**Search Criteria Examples:**
```json
// Simple equality
{"team": "team1", "is_alive": true}

// Lambda functions (as strings)
{"level": "lambda x: x >= 3"}
{"lifetime_tick": "lambda x: x > 100"}

// Position matching
{"position": [5, 5]}
```

---

## Real-time Events

### GET /events
Server-Sent Events stream for real-time monitoring.

**Usage:**
```javascript
// JavaScript
const eventSource = new EventSource('http://localhost:5000/api/events');
eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Event:', data.type, data.data);
};
```

```python
# Python
import sseclient
import requests

response = requests.get('http://localhost:5000/api/events', stream=True)
client = sseclient.SSEClient(response)

for event in client.events():
    if event.data:
        data = json.loads(event.data)
        print(f"Event: {data['type']}")
```

**Event Types:**
- `step_start`: Server step beginning
- `step_end`: Server step completion
- `player_died`: Player death
- `player_property_changed`: Player property modification
- `tile_changed`: Map tile modification
- `team_added`: Team addition
- `team_removed`: Team removal
- `keepalive`: Connection keepalive

**Event Format:**
```json
{
  "type": "player_died",
  "data": {
    "id": 0,
    "team": "team1",
    "position": [5, 3],
    "level": 2,
    ...
  },
  "timestamp": 1640995200.0
}
```

---

## Utility Endpoints

### GET /health
Health check and server status.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "server_ticks": 42,
    "server_size": 10,
    "teams": 2,
    "players": 4,
    "timestamp": 1640995200.0
  }
}
```

### GET /docs
API documentation and endpoint list.

**Response:**
```json
{
  "success": true,
  "data": {
    "title": "Zappy Server REST API",
    "version": "1.0.0",
    "endpoints": {
      "GET /api/server": "Get complete server data",
      "GET /api/server/stats": "Get server statistics",
      ...
    }
  }
}
```

---

## Python Client Library

### Installation
```python
# Save rest_api_test.py and import the client
from rest_api_test import ZappyAPIClient

client = ZappyAPIClient("http://localhost:5000")
```

### Usage Examples

```python
# Basic operations
health = client.health_check()
stats = client.get_server_stats()
players = client.get_players(team="team1", alive_only=True)

# Modifications
client.update_player(0, 'level', 5)
client.update_map_tile(5, 5, 'food', 20)
client.step_server()

# Search
high_level_players = client.search_players(level="lambda x: x >= 3")
team1_players = client.search_players(team="team1", is_alive=True)

# Team management
client.add_team("team3")
team_data = client.get_team("team1")
client.remove_team("team3")
```

---

## Error Handling

### HTTP Status Codes
- `200 OK`: Success
- `201 Created`: Resource created
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

### Error Response Format
```json
{
  "success": false,
  "error": "Player not found"
}
```

### Common Errors
- **Player not found**: Invalid player ID
- **Team not found**: Invalid team name
- **Invalid coordinates**: Map coordinates out of bounds
- **Missing required fields**: Request body missing required data
- **Server error**: Internal server error

---

## Command Line Usage

### Starting the API Server
```bash
# Basic usage
python rest_api.py

# Custom configuration
python rest_api.py --host 0.0.0.0 --port 8080 --size 15 --eggs 10 --teams red blue green

# Debug mode
python rest_api.py --debug
```

### Command Line Options
- `--host`: Host to bind to (default: localhost)
- `--port`: Port to bind to (default: 5000)
- `--debug`: Enable debug mode
- `--size`: Map size (default: 10)
- `--eggs`: Number of eggs (default: 5)
- `--teams`: Team names (default: team1 team2)

### Testing
```bash
# Run comprehensive tests
python rest_api_test.py

# Manual testing with curl
curl -X GET http://localhost:5000/api/health
curl -X POST http://localhost:5000/api/server/step
curl -X GET "http://localhost:5000/api/players?alive=true"
```

---

## Integration Examples

### Monitoring Dashboard
```python
import time
from rest_api_test import ZappyAPIClient

client = ZappyAPIClient()

while True:
    stats = client.get_server_stats()
    if stats['success']:
        data = stats['data']
        print(f"Tick {data['total_ticks']}: {data['alive_players']} alive")
    
    time.sleep(1)
```

### AI Training Data Collection
```python
training_data = []

def collect_data():
    server_data = client.get_server_data()
    if server_data['success']:
        training_data.append({
            'tick': server_data['data']['ticks'],
            'state': server_data['data'],
            'timestamp': time.time()
        })

# Collect data every step
while True:
    collect_data()
    client.step_server()
    time.sleep(0.1)
```

### Game Analysis
```python
def analyze_game():
    stats = client.get_server_stats()
    teams = client.get_teams()
    
    if stats['success'] and teams['success']:
        # Analyze team performance
        for team in teams['data']:
            efficiency = team['total_level'] / max(team['total_players'], 1)
            print(f"{team['team']}: efficiency {efficiency:.2f}")
        
        # Check win conditions
        if stats['data']['highest_level'] >= 8:
            print("Someone is close to winning!")

analyze_game()
```

This REST API provides a complete HTTP interface for external applications to interact with the Zappy server, enabling web dashboards, mobile apps, AI training systems, and any other HTTP-capable application to monitor and control the game server.
