from __future__ import annotations

""" Global Imports """
import random
from typing import TYPE_CHECKING
from dataclasses import dataclass

""" Local Imports """
if TYPE_CHECKING:
    from server.player import Player

from server.constant import RESOURCE_DENSITY, Resource, ElevationRequirement

class Map:
    def __init__(self, teams: list[str], players: dict[int: Player], nb_eggs: int, size: int = 42):
        # Type definition
        self.size: int = size
        self.players: dict[int: Player] = players
        self.teams: list[str] = teams
        self.total_resources: dict[Resource: int] = {}
        self.tiles: list[list[Resource]] = []
        self.incanted_tile: dict[(int, int): IncantationTile] = {}  # (x, y)
        self.eggs_tile: dict[str: [Coordinate]] = {}  # (x, y)
        self.reset(nb_eggs)

    def __map_resource(self):
        for resource in Resource:
            if self.total_resources[resource] > 0:
                while self.total_resources[resource]:
                    x = random.randint(0, self.size - 1)
                    y = random.randint(0, self.size - 1)
                    self.tiles[y][x][resource] += 1
                    self.total_resources[resource] -= 1

    def __map_eggs(self, nb_eggs: int):
        for team in self.teams:
            for _ in range(nb_eggs):
                if self.eggs_tile.get(team) == None:
                    self.eggs_tile[team] = []
                self.__add_egg(team)

    def __add_egg(self, team: str):
        x = random.randint(0, self.size - 1)
        y = random.randint(0, self.size - 1)
        self.eggs_tile[team].append(Coordinate(x, y))

    def clone(self):
        """Deep copy of the map"""
        new_map = Map(self.size)
        new_map.tiles = [row.copy() for row in self.tiles]
        new_map.incanted_tile = self.incanted_tile.copy()
        new_map.total_resources = self.total_resources.copy()
        return new_map

    def nb_eggs(self, team: str):
        return len(self.eggs_tile[team])

    # Initialize the map
    # Reset total resources and tiles in case they are distributed unequally due to agent strategy
    def reset(self, nb_eggs: int):
        self.total_resources = {
            res: int(self.size * self.size * RESOURCE_DENSITY[res]) for res in Resource}
        self.tiles = [[{res: 0 for res in Resource}
                       for _ in range(self.size)] for _ in range(self.size)]
        self.incanted_tile: dict[(int, int): IncantationTile] = {}  # (x, y)
        self.eggs_tile: dict[str: [Coordinate]] = {}  # (x, y)
        self.__map_resource()
        self.__map_eggs(nb_eggs)

    def evolve(self, cur_tile: IncantationTile, coords: tuple):
        x, y = coords[0], coords[1]
        players_on_tile = [p for p in self.players.values() if p.position == (
            x, y) and p.level == cur_tile.level and p is not self]
        if len(players_on_tile) <= ElevationRequirement.requirements[cur_tile.level][0]:
            for player in players_on_tile:
                player.add_res("ko")
        # Level up all qualifying players on the tile
        for player in players_on_tile:
            if player.level == cur_tile.level:
                player.level += 1
                player.add_res("Current level: " + str(player.level) + "\n")


@dataclass
class Coordinate:
    x: int
    y: int


@dataclass
class IncantationTile:
    remaining_ticks: int
    level: int
