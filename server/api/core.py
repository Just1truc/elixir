"""
core.py

This module provides a comprehensive API system for the Zappy server simulation.
It offers event-driven callbacks, data serialization, and CRUD operations for all server entities.

The API supports:
- Real-time event monitoring through callbacks
- Complete server state serialization
- Player, team, and map data access and modification
- Search and statistics functionality
- Extensible event system for custom integrations

Usage Example:
    from server import Server
    from api.core import ServerAPI
    
    server = Server(10, 5, ["team1", "team2"])
    api = ServerAPI(server)
    
    # Register callbacks
    api.register_callback('step_end', lambda data: print(f"Tick {data['ticks']}"))
    
    # Get data
    all_data = api.get_server_data()
    player_data = api.get_player_data(0)
    
    # Modify state
    api.set_player_property(0, 'level', 5)
    api.set_map_tile(5, 5, 'food', 10)
"""

from typing import Dict, List, Any, Callable, Optional
from server import constant


class ServerAPI:
    """
    Comprehensive API for Zappy server data access and manipulation.
    
    This class provides a clean interface to:
    - Monitor server state changes through events
    - Access and modify all server data
    - Search for specific entities
    - Get statistics and analytics
    """
    
    def __init__(self, server):
        """
        Initialize the API with a server instance.
        
        Args:
            server: The Server instance to provide API access for
        """
        self.server = server
        self.callbacks: Dict[str, List[Callable]] = {}
    
    # Event System
    def register_callback(self, event: str, callback: Callable):
        """
        Register a callback function for specific API events.
        
        Available events:
        - 'step_start': Triggered at the beginning of each server step
        - 'step_end': Triggered at the end of each server step
        - 'player_property_changed': Triggered when a player property is modified
        - 'tile_changed': Triggered when a map tile is modified
        - 'team_added': Triggered when a team is added
        - 'team_removed': Triggered when a team is removed
        - 'player_died': Triggered when a player dies
        - 'incantation_started': Triggered when an incantation begins
        - 'incantation_completed': Triggered when an incantation completes
        
        Args:
            event (str): The event name to listen for
            callback (Callable): Function to call when event occurs
                                Callback receives event data as parameter
        
        Example:
            api.register_callback('step_end', lambda data: print(f"Tick: {data['ticks']}"))
        """
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def unregister_callback(self, event: str, callback: Callable):
        """
        Remove a specific callback for an event.
        
        Args:
            event (str): The event name
            callback (Callable): The callback function to remove
        """
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
    
    def trigger_event(self, event: str, data: Any = None):
        """
        Trigger all callbacks for a specific event.
        
        Args:
            event (str): The event name to trigger
            data (Any): Data to pass to the callbacks
        """
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Error in callback for event '{event}': {e}")
    
    # Data Getters
    def get_server_data(self) -> Dict[str, Any]:
        """
        Get complete server state as a dictionary.
        
        Returns:
            Dict containing:
            - ticks: Current server tick count
            - teams: List of all team names
            - players: List of serialized player data
            - map_size: Size of the game map
            - map_tiles: 2D array of map tile data
            - eggs_tiles: Dictionary of egg positions by team
            - incanted_tiles: Dictionary of ongoing incantations
            - total_resources: Total resources available on the map
        """
        return {
            'ticks': self.server.ticks,
            'teams': self.server.teams.copy(),
            'players': [self._serialize_player(p) for p in self.server.players],
            'map_size': self.server.map.size,
            'map_tiles': self._serialize_map_tiles(),
            'eggs_tiles': dict(self.server.map.eggs_tile),
            'incanted_tiles': dict(self.server.map.incanted_tile),
            'total_resources': self._serialize_resources(self.server.map.total_resources)
        }
    
    def get_player_data(self, player_id: int) -> Optional[Dict[str, Any]]:
        """
        Get data for a specific player.
        
        Args:
            player_id (int): The ID of the player to retrieve
            
        Returns:
            Dict containing player data or None if player not found
        """
        player = next((p for p in self.server.players if p.id == player_id), None)
        return self._serialize_player(player) if player else None
    
    def get_team_data(self, team: str) -> Dict[str, Any]:
        """
        Get comprehensive data for a specific team.
        
        Args:
            team (str): The team name
            
        Returns:
            Dict containing team data
        """
        team_players = [p for p in self.server.players if p.team == team]
        alive_players = [p for p in team_players if p.is_alive]
        total_level = sum(p.level for p in team_players)
        
        return {
            'team': team,
            'players': [self._serialize_player(p) for p in team_players],
            'eggs': self.server.map.eggs_tile.get(team, []).copy(),
            'alive_count': len(alive_players),
            'total_players': len(team_players),
            'total_level': total_level,
            'average_level': total_level / len(team_players) if team_players else 0
        }
    
    def get_map_data(self) -> Dict[str, Any]:
        """
        Get map-specific data and statistics.
        
        Returns:
            Dict containing map data
        """
        total_tiles = self.server.map.size ** 2
        return {
            'size': self.server.map.size,
            'tiles': self._serialize_map_tiles(),
            'eggs_tiles': dict(self.server.map.eggs_tile),
            'incanted_tiles': dict(self.server.map.incanted_tile),
            'total_resources': self._serialize_resources(self.server.map.total_resources),
            'resource_density': {
                self._resource_to_string(res): count / total_tiles 
                for res, count in self.server.map.total_resources.items()
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive server statistics and analytics.
        
        Returns:
            Dict containing various statistics
        """
        alive_players = [p for p in self.server.players if p.is_alive]
        team_distribution = {}
        for team in self.server.teams:
            team_distribution[team] = len([p for p in self.server.players if p.team == team])
        
        # Calculate resource utilization - fix Resource enum comparison
        player_resources = {}
        for player in self.server.players:
            for resource, amount in player.inventory.items():
                resource_key = self._resource_to_string(resource)
                player_resources[resource_key] = player_resources.get(resource_key, 0) + amount
        
        return {
            'total_ticks': self.server.ticks,
            'total_players': len(self.server.players),
            'alive_players': len(alive_players),
            'teams_count': len(self.server.teams),
            'incantations_in_progress': len(self.server.map.incanted_tile),
            'total_eggs': sum(len(eggs) for eggs in self.server.map.eggs_tile.values()),
            'average_player_level': sum(p.level for p in alive_players) / len(alive_players) if alive_players else 0,
            'highest_level': max((p.level for p in self.server.players), default=0),
            'team_distribution': team_distribution,
            'player_resources': player_resources,
            'map_resources': self._serialize_resources(self.server.map.total_resources)
        }
    
    # Lightweight Data Methods for Efficient Transfer
    def get_server_stats_light(self) -> Dict[str, Any]:
        """Get lightweight server statistics for efficient transfer."""
        alive_players = [p for p in self.server.players if p.is_alive]
        return {
            'ticks': self.server.ticks,
            'total_players': len(self.server.players),
            'alive_players': len(alive_players),
            'teams': len(self.server.teams),
            'incantations': len(self.server.map.incanted_tile),
            'total_eggs': sum(len(eggs) for eggs in self.server.map.eggs_tile.values())
        }
    
    def get_players_light(self) -> List[Dict[str, Any]]:
        """Get lightweight player data for efficient transfer."""
        return [self._serialize_player(p, lightweight=True) for p in self.server.players]
    
    def get_alive_players_light(self) -> List[Dict[str, Any]]:
        """Get lightweight data for alive players only."""
        return [self._serialize_player(p, lightweight=True) for p in self.server.players if p.is_alive]
    
    def get_teams_summary(self) -> List[Dict[str, Any]]:
        """Get lightweight team summary data."""
        teams_data = []
        for team in self.server.teams:
            team_players = [p for p in self.server.players if p.team == team]
            alive_count = sum(1 for p in team_players if p.is_alive)
            teams_data.append({
                'team': team,
                'total': len(team_players),
                'alive': alive_count,
                'eggs': len(self.server.map.eggs_tile.get(team, []))
            })
        return teams_data
    
    # Search and Query
    def search_players(self, **criteria) -> List[Dict[str, Any]]:
        """
        Search for players matching given criteria.
        
        Args:
            **criteria: Key-value pairs to match against player attributes
            
        Returns:
            List of serialized player data matching the criteria
        """
        result = []
        for player in self.server.players:
            match = True
            for key, value in criteria.items():
                if not hasattr(player, key):
                    match = False
                    break
                
                player_value = getattr(player, key)
                if callable(value):
                    if not value(player_value):
                        match = False
                        break
                elif player_value != value:
                    match = False
                    break
            
            if match:
                result.append(self._serialize_player(player))
        return result
    
    def get_players_by_team(self, team: str) -> List[Dict[str, Any]]:
        """Get all players in a specific team."""
        return [self._serialize_player(p) for p in self.server.players if p.team == team]
    
    def get_alive_players(self) -> List[Dict[str, Any]]:
        """Get alive players only."""
        return [self._serialize_player(p) for p in self.server.players if p.is_alive]
    
    def get_players_at_position(self, x: int, y: int) -> List[Dict[str, Any]]:
        """Get all players at a specific position."""
        return [
            self._serialize_player(p) for p in self.server.players 
            if p.position == (x, y)
        ]
    
    # Data Setters
    def set_player_property(self, player_id: int, property: str, value: Any) -> bool:
        """
        Set a property for a specific player.
        Supports nested inventory updates via dot-path (e.g., 'inventory.food')
        and shorthand 'pos' for 'position'.
        
        Args:
            player_id (int): The player ID
            property (str): The property name to modify
            value (Any): The new value
            
        Returns:
            bool: True if successful, False if player not found or property invalid
        """
        player = next((p for p in self.server.players if p.id == player_id), None)
        if not player:
            return False

        prop_name_for_event = property

        # Position updates (allow 'pos' or 'position')
        if property in ('pos', 'position'):
            try:
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    x, y = int(value[0]), int(value[1])
                    # Validate bounds
                    if 0 <= x < self.server.map.size and 0 <= y < self.server.map.size:
                        old_value = getattr(player, 'position')
                        player.position = (x, y)
                        prop_name_for_event = 'position'
                        self.trigger_event('player_property_changed', {
                            'player_id': player_id,
                            'property': prop_name_for_event,
                            'old_value': old_value,
                            'new_value': player.position
                        })
                        return True
                    return False
            except Exception:
                return False

        # Inventory updates via dot path e.g., 'inventory.food'
        if property.startswith('inventory.'):
            resource_key = property.split('.', 1)[1]
            if resource_key in player.inventory:
                try:
                    old_value = player.inventory[resource_key]
                    player.inventory[resource_key] = int(value)
                    self.trigger_event('player_property_changed', {
                        'player_id': player_id,
                        'property': property,
                        'old_value': old_value,
                        'new_value': player.inventory[resource_key]
                    })
                    return True
                except Exception:
                    return False
            # Accept common resource names mapping if needed
            # Try lowercase fallback
            if resource_key.lower() in player.inventory:
                try:
                    old_value = player.inventory[resource_key.lower()]
                    player.inventory[resource_key.lower()] = int(value)
                    self.trigger_event('player_property_changed', {
                        'player_id': player_id,
                        'property': property,
                        'old_value': old_value,
                        'new_value': player.inventory[resource_key.lower()]
                    })
                    return True
                except Exception:
                    return False
            return False

        # Generic attribute update
        if hasattr(player, property):
            old_value = getattr(player, property)
            try:
                setattr(player, property, value)
            except Exception:
                return False
            self.trigger_event('player_property_changed', {
                'player_id': player_id,
                'property': prop_name_for_event,
                'old_value': old_value,
                'new_value': value
            })
            return True

        return False
    
    def set_map_tile(self, x: int, y: int, resource: str, amount: int) -> bool:
        """
        Set resource amount on a specific map tile.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            resource (str): Resource type
            amount (int): New amount
            
        Returns:
            bool: True if successful, False if coordinates invalid
        """
        # Validate coords
        if not (0 <= x < self.server.map.size and 0 <= y < self.server.map.size):
            return False
        # Map resource string to enum used in map tiles
        res_enum = self._resource_from_string(resource)
        if res_enum is None:
            return False
        # Update
        tile = self.server.map.tiles[y][x]
        old_amount = tile.get(res_enum, 0)
        tile[res_enum] = int(amount)
        self.trigger_event('tile_changed', {
            'position': (x, y),
            'resource': resource,
            'old_amount': old_amount,
            'new_amount': amount
        })
        return True
    
    def add_team(self, team: str) -> bool:
        """
        Add a new team to the server.
        
        Args:
            team (str): Team name to add
            
        Returns:
            bool: True if added, False if team already exists
        """
        if team not in self.server.teams:
            self.server.teams.append(team)
            self.server.map.eggs_tile[team] = []
            self.trigger_event('team_added', {'team': team})
            return True
        return False
    
    def remove_team(self, team: str) -> bool:
        """
        Remove a team from the server.
        
        Args:
            team (str): Team name to remove
            
        Returns:
            bool: True if removed, False if team doesn't exist
        """
        if team in self.server.teams:
            removed_players = [p.id for p in self.server.players if p.team == team]
            self.server.teams.remove(team)
            self.server.players = [p for p in self.server.players if p.team != team]
            if team in self.server.map.eggs_tile:
                del self.server.map.eggs_tile[team]
            self.trigger_event('team_removed', {
                'team': team,
                'removed_player_ids': removed_players
            })
            return True
        return False
    
    # Utility Methods
    def _resource_to_string(self, resource) -> str:
        """Convert a resource (enum or other) to a string safely."""
        if hasattr(resource, 'value'):
            return resource.value
        else:
            return str(resource)
    
    def _resource_from_string(self, resource: str):
        """Convert a string to a Resource enum if possible; otherwise return None."""
        if isinstance(resource, str):
            return constant.RESOURCE_ENUM.get(resource)
        return None
    
    def _serialize_resources(self, resources_dict) -> Dict[str, Any]:
        """Convert a resources dictionary to a serializable format."""
        return {self._resource_to_string(res): amount for res, amount in resources_dict.items()}
    
    def _serialize_player(self, player, lightweight: bool = False) -> Dict[str, Any]:
        """
        Convert a player object to a dictionary.
        
        Args:
            player: Player object to serialize
            lightweight: If True, return only essential data for transfer efficiency
            
        Returns:
            Dict containing player data (full or lightweight)
        """
        if not player:
            return None
        
        # Lightweight version for efficient data transfer
        if lightweight:
            return {
                'id': player.id,
                'team': player.team,
                'pos': player.position,  # Shortened field name
                'lvl': player.level,     # Shortened field name
                'alive': player.is_alive
            }
        
        # Serialize inventory safely
        inventory = {}
        for resource, amount in player.inventory.items():
            inventory[self._resource_to_string(resource)] = amount
        
        # Full version with all details    
        return {
            'id': player.id,
            'team': player.team,
            'position': player.position,
            'level': player.level,
            'orientation': str(player.direction),  # Convert Direction enum to string
            'inventory': inventory,
            'is_alive': player.is_alive,
            'lifetime_tick': player.lifetime_tick,
            'remain_tick': player.remain_tick,
            'current_command': str(player.cur_cmd) if player.cur_cmd else None,
            'command_queue_length': len(player.command_queue)
        }
    
    def _serialize_map_tiles(self) -> List[List[Dict[str, int]]]:
        """
        Convert map tiles to a serializable format.
        
        Returns:
            2D list of dictionaries representing tile resources
        """
        serialized_tiles = []
        for row in self.server.map.tiles:
            serialized_row = []
            for tile in row:
                # Convert Resource enum keys to strings
                serialized_tile = {}
                for resource, amount in tile.items():
                    resource_key = self._resource_to_string(resource)
                    serialized_tile[resource_key] = amount
                serialized_row.append(serialized_tile)
            serialized_tiles.append(serialized_row)
        return serialized_tiles
    
    # Integration with Server Step
    def on_step_start(self):
        """Called at the beginning of each server step."""
        self.trigger_event('step_start', {'ticks': self.server.ticks})
    
    def on_step_end(self):
        """Called at the end of each server step."""
        self.trigger_event('step_end', {
            'ticks': self.server.ticks,
            'server_data': self.get_server_stats_light()  # Use lightweight version for events
        })
    
    def on_player_death(self, player_id: int):
        """Called when a player dies."""
        player_data = self.get_player_data(player_id)
        if player_data:
            self.trigger_event('player_died', player_data)
    
    def on_incantation_start(self, position: tuple, level: int):
        """Called when an incantation starts."""
        self.trigger_event('incantation_started', {
            'position': position,
            'level': level,
            'tick': self.server.ticks
        })
    
    def on_incantation_complete(self, position: tuple, level: int, success: bool):
        """Called when an incantation completes."""
        self.trigger_event('incantation_completed', {
            'position': position,
            'level': level,
            'success': success,
            'tick': self.server.ticks
        })
