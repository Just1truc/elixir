from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from server.constant import RESOURCE_DENSITY, Resource, ElevationRequirement
from typing import TYPE_CHECKING
from server.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from server.player import Player


class Map:
    DEFAULT_MAP_SIZE = 42
    def __init__(self, teams: list[str], players: list['Player'], nb_eggs: int, size: int = DEFAULT_MAP_SIZE):
        """
        Initialize the Map with teams, players, number of eggs, and map size.
        Args:
            teams (list[str]): List of team names.
            players (list[Player]): List of Player objects.
            nb_eggs (int): Number of eggs per team.
            size (int, optional): Size of the map (width and height). Defaults to 42.
        """
        self.size: int = size
        self.players: list[Player] = players
        self.teams: list[str] = teams
        self.total_resources: dict[Resource: int] = {}
        self.tiles: list[list[Resource]] = []
        self.incanted_tile: dict[(int, int): IncantationTile] = {}  # (x, y)
        self.eggs_tile: dict[str, list[tuple[int, int]]] = {}  # (x, y)
        self.init_nb_eggs = nb_eggs
        self.reset(nb_eggs)
        logger.info("Map initialized | size=%s teams=%s nb_eggs=%s", size, len(teams), nb_eggs)

    def __map_resource(self):
        for resource in Resource:
            while self.total_resources[resource]:
                x = random.randint(0, self.size - 1)
                y = random.randint(0, self.size - 1)
                self.tiles[y][x][resource] += 1
                self.total_resources[resource] -= 1
        logger.debug("Resources distributed")

    def __map_eggs(self, nb_eggs: int):
        for team in self.teams:
            for _ in range(nb_eggs):
                self.__add_egg(team)
        logger.debug("Eggs placed | per_team=%s", nb_eggs)

    def __add_egg(self, team: str):
        x = random.randint(0, self.size - 1)
        y = random.randint(0, self.size - 1)
        self.eggs_tile[team].append((x, y))

    def clone(self):
        """
        Create a deep copy of the map, including players, tiles, incantations, and resources.
        Returns:
            Map: A new Map instance with the same state.
        """
        players = [p.clone() for p in self.players]
        new_map = Map(self.teams.copy(), players, self.init_nb_eggs, self.size)
        new_map.tiles = [row.copy() for row in self.tiles]
        new_map.incanted_tile = self.incanted_tile.copy()
        new_map.total_resources = self.total_resources.copy()
        new_map.eggs_tile = defaultdict(list, {k: v.copy() for k, v in self.eggs_tile.items()})
        logger.debug("Map cloned | eggs=%s incantations=%s", sum(len(v) for v in new_map.eggs_tile.values()), len(new_map.incanted_tile))
        return new_map

    def nb_eggs(self, team: str) -> int:
        """
        Get the number of eggs available for a given team.
        Args:
            team (str): The team name.
        Returns:
            int: Number of eggs for the team.
        """
        return len(self.eggs_tile[team])

    # Initialize the map
    # Reset total resources and tiles in case they are distributed unequally due to agent strategy
    def reset(self, nb_eggs: int):
        """
        Reset the map state, redistributing resources and eggs.
        Args:
            nb_eggs (int): Number of eggs per team to place on the map.
        """
        self.total_resources = {
            res: int(self.size * self.size * RESOURCE_DENSITY[res]) for res in Resource}
        self.tiles = [[{res: 0 for res in Resource}
                       for _ in range(self.size)] for _ in range(self.size)]
        self.incanted_tile: dict[(int, int): IncantationTile] = {}  # (x, y)
        self.eggs_tile = defaultdict(list)
        self.__map_resource()
        self.__map_eggs(nb_eggs)
        logger.info("Map reset | nb_eggs=%s", nb_eggs)

    def evolve(self, cur_tile: IncantationTile, coords: tuple[int, int]):
        """
        Evolve the tile at the given coordinates after an incantation completes.
        Args:
            cur_tile (IncantationTile): The incantation tile object.
            coords (tuple): The (x, y) coordinates of the tile.
        """
        x, y = coords
        players_on_tile = [p for p in self.players if p.position == (x, y) and p.level == cur_tile.level]
        if len(players_on_tile) <= ElevationRequirement.requirements[cur_tile.level][0]:
            for player in players_on_tile:
                player.add_res("ko")
        # Level up all qualifying players on the tile
        for player in players_on_tile:
            if player.level == cur_tile.level:
                player.level += 1
                player.add_res(f"Current level: {player.level}\n")
        logger.debug("Tile evolved | coords=%s level=%s players=%s", coords, cur_tile.level, len(players_on_tile))


@dataclass
class IncantationTile:
    remaining_ticks: int
    level: int
