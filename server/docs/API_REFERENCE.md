# API Reference

This document provides a comprehensive reference for all classes, methods, and constants in the Zappy AI Training Environment.

## Core Classes

### Player

The `Player` class represents a game participant with abilities to move, interact with resources, and communicate.

#### Constructor
```python
Player(id: int, map: Map, team: str)
```

**Parameters:**
- `id`: Unique player identifier
- `map`: Reference to the game map
- `team`: Team name the player belongs to

**Properties:**
- `id: int` - Player identifier
- `team: str` - Team name
- `level: int` - Current player level (1-8)
- `position: tuple[int, int]` - Current (x, y) coordinates
- `direction: Direction` - Current facing direction
- `inventory: dict[str, int]` - Resource inventory
- `is_alive: bool` - Player life status
- `lifetime_tick: int` - Total ticks lived
- `remain_tick: int` - Ticks remaining for current command

#### Movement Commands
```python
def forward(self) -> None
```
Move player forward in current direction with map wrapping.
**Response:** "ok"

```python
def left(self) -> None
```
Turn player counter-clockwise.
**Response:** "ok"

```python
def right(self) -> None
```
Turn player clockwise.
**Response:** "ok"

#### Information Commands
```python
def look(self) -> None
```
Examine current tile resources.
**Response:** Dictionary representation of tile resources

```python
def inventory_cmd(self) -> None
```
Get current inventory status.
**Response:** Dictionary of inventory contents

```python
def connect_nbr(self) -> None
```
Get number of available connection slots for team.
**Response:** String representation of available eggs

#### Resource Commands
```python
def take(self, resource: str) -> None
```
Take one unit of specified resource from current tile.
**Parameters:**
- `resource`: Resource name ("food", "linemate", etc.)
**Response:** "ok" on success, "ko" on failure

```python
def set(self, resource: str) -> None
```
Drop one unit of specified resource onto current tile.
**Parameters:**
- `resource`: Resource name to drop
**Response:** "ok" on success, "ko" on failure

#### Communication Commands
```python
def broadcast(self, text: str) -> None
```
Send message to all players with directional information.
**Parameters:**
- `text`: Message content
**Response:** "ok"

#### Special Commands
```python
def fork(self) -> None
```
Create new connection slot (egg) at current position.
**Response:** "ok"

```python
def eject(self) -> None
```
Push other players from current tile in facing direction.
**Response:** "ok" if players moved, "ko" if none moved

```python
def incantation(self) -> None
```
Attempt to start elevation ritual if requirements met.
**Response:** "Elevation underway" on success, "ko" on failure

#### Command Management
```python
def add_cmd(self, cmd: str) -> None
```
Add command to player's command queue.
**Parameters:**
- `cmd`: Command string with optional parameters

```python
def get_res(self) -> str | None
```
Retrieve next response from response queue.
**Returns:** Response string or None if empty

---

### Server

The `Server` class manages the overall game state and simulation.

#### Constructor
```python
Server(size: int, nb_eggs: int, teams: list[str])
```

**Parameters:**
- `size`: Map dimensions (size × size)
- `nb_eggs`: Initial eggs per team
- `teams`: List of team names

**Properties:**
- `teams: list[str]` - Team names
- `players: list[Player]` - All active players
- `map: Map` - Game map instance
- `ticks: int` - Total simulation ticks

#### Player Management
```python
def add_player(self, team: str) -> Player | None
```
Add new player to specified team.
**Returns:** Player instance or None if impossible

#### Simulation Control
```python
def step(self) -> None
```
Advance simulation by one tick, processing all players.

```python
def reset(self, nb_eggs: int, teams: list[str] = None) -> None
```
Reset server state for new game.

```python
def clone(self) -> Server
```
Create deep copy of current server state.

---

### Map

The `Map` class represents the game world with resources and tile information.

#### Constructor
```python
Map(teams: list[str], players: list[Player], nb_eggs: int, size: int = 42)
```

**Properties:**
- `size: int` - Map dimensions
- `tiles: list[list[dict]]` - 2D array of tile resources
- `players: list[Player]` - Reference to all players
- `incanted_tile: dict` - Tiles with ongoing incantations
- `eggs_tile: dict[str, list[tuple]]` - Egg locations per team

#### Methods
```python
def nb_eggs(self, team: str) -> int
```
Get number of available eggs for team.

```python
def clone(self) -> Map
```
Create deep copy of map state.

```python
def evolve(self, tile: IncantationTile, coords: tuple) -> None
```
Complete incantation and level up players.

---

### Environment

Base class for AI training environments with reward calculation.

#### Constructor
```python
Environment(size: int = 10, nb_teams: int = 4, nb_connect: int = 5, seed: int = None)
```

#### Methods
```python
def reset(self) -> dict
```
Reset environment and return initial state.
**Returns:** State dictionary with players, map, and resources

```python
def step(self) -> tuple[dict, dict, bool]
```
Advance simulation one step.
**Returns:** (state, rewards, terminated)

```python
def get_state(self) -> dict
```
Get current environment state.

---

## Constants and Enums

### Direction
```python
class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
```

### Resource
```python
class Resource(Enum):
    FOOD = "food"
    LINEMATE = "linemate"
    DERAUMERE = "deraumere"
    SIBUR = "sibur"
    MENDIANE = "mendiane"
    PHIRAS = "phiras"
    THYSTAME = "thystame"
```

### Resource Density
Default spawn rates for each resource type:
```python
RESOURCE_DENSITY = {
    Resource.FOOD: 0.5,
    Resource.LINEMATE: 0.3,
    Resource.DERAUMERE: 0.15,
    Resource.SIBUR: 0.1,
    Resource.MENDIANE: 0.1,
    Resource.PHIRAS: 0.08,
    Resource.THYSTAME: 0.05,
}
```

### Elevation Requirements
Player and resource requirements for each level advancement:
```python
# Format: level: (required_players, {resource: count})
ElevationRequirement.requirements = {
    1: (1, {Resource.LINEMATE: 1}),
    2: (2, {Resource.LINEMATE: 1, Resource.DERAUMERE: 1, Resource.SIBUR: 1}),
    3: (2, {Resource.LINEMATE: 2, Resource.SIBUR: 1, Resource.PHIRAS: 2}),
    4: (4, {Resource.LINEMATE: 1, Resource.DERAUMERE: 1, Resource.SIBUR: 2, Resource.PHIRAS: 1}),
    5: (4, {Resource.LINEMATE: 1, Resource.DERAUMERE: 2, Resource.SIBUR: 1, Resource.MENDIANE: 3}),
    6: (6, {Resource.LINEMATE: 1, Resource.DERAUMERE: 2, Resource.SIBUR: 3, Resource.PHIRAS: 1}),
    7: (6, {Resource.LINEMATE: 2, Resource.DERAUMERE: 2, Resource.SIBUR: 2,
             Resource.MENDIANE: 2, Resource.PHIRAS: 2, Resource.THYSTAME: 1})
}
```

### Command Timings
Tick duration for each player command:
```python
CommandTick = {
    Player.forward: 7,
    Player.right: 7,
    Player.left: 7,
    Player.look: 7,
    Player.inventory_cmd: 1,
    Player.broadcast: 7,
    Player.connect_nbr: 0,
    Player.fork: 42,
    Player.eject: 7,
    Player.take: 7,
    Player.set: 7,
    Player.incantation: 300
}
```

## Environment Categories

Pre-configured environment types available through `Category` dictionary:

- `"alone"`: Single player environment
- `"group"`: Multi-player same level
- `"random_group"`: Multi-player random levels
- `"survival"`: Limited food challenge
- `"incantation"`: Ready-to-elevate scenario
- `"close_incantation"`: Almost-ready elevation scenario

## Error Handling

### Common Response Codes
- `"ok"`: Command executed successfully
- `"ko"`: Command failed (insufficient resources, invalid action, etc.)
- `"message <dir>, <text>"`: Broadcast message received
- `"eject: <dir>"`: Player was ejected from tile
- `"Elevation underway"`: Incantation started successfully

### Exception Handling
- Invalid commands are ignored with console warning
- Out-of-bounds positions are wrapped using modulo arithmetic
- Resource operations fail gracefully with "ko" response
- Dead players cannot execute commands
