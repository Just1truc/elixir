from __future__ import annotations

import math
import random
# import constant

from typing import TYPE_CHECKING

from server.constant import Direction, Resource, RESOURCE_ENUM, ElevationRequirement
from server.map import Map
# import map

if TYPE_CHECKING:
    from player import Command

class Player:
    def __init__(self, id: int, map: Map, team: str):
        self.id = id
        self.team = team
        self.map = map
        self.level = 1
        self.direction = Direction.NORTH
        self.position = (random.randint(0, map.size - 1),
                         random.randint(0, map.size - 1))
        self.inventory = {res.value: 0 for res in Resource}
        self.inventory[Resource.FOOD.value] = 10  # Starting food
        self.cur_cmd: function = None
        self.cur_cmd_param: list[str] = None
        self.remain_tick = 0
        self.lifetime_tick = 1
        self.is_alive = True
        self.command_queue: list[(function, list[str])] = []
        self.response_queue: list[str] = []

    def add_cmd(self, cmd: str):
        cmd, param = self.get_cmd_func(cmd)
        if cmd == None:
            return
        self.command_queue.append((cmd, param))

    @staticmethod
    def get_cmd_func(cmd: str):
        words = cmd.split()
        func = Command.get(words[0])
        if func == None:
            print(cmd + " cmd does not exist")
            return None
        return func, words[1:]

    def add_res(self, res: str):
        self.response_queue.append(res)

    def get_res(self) -> str:
        if len(self.response_queue) != 0:
            return self.response_queue.pop(0)
        return None

    def check_food(self):
        if self.lifetime_tick % 128 == 0 and self.inventory[Resource.FOOD.value] > 0:
            self.inventory[Resource.FOOD.value] -= 1

    def forward(self):
        x, y = self.position
        if self.direction == Direction.NORTH:
            self.position = (x, (y - 1) % self.map.size)
        elif self.direction == Direction.EAST:
            self.position = ((x + 1) % self.map.size, y)
        elif self.direction == Direction.SOUTH:
            self.position = (x, (y + 1) % self.map.size)
        elif self.direction == Direction.WEST:
            self.position = ((x - 1) % self.map.size, y)
        self.add_res("ok")

    def right(self):
        self.direction = Direction((self.direction.value + 1) % 4)
        self.add_res("ok")

    def left(self):
        self.direction = Direction((self.direction.value - 1) % 4)
        self.add_res("ok")

    def look(self):
        x, y = self.position
        vision = []
        for depth in range(self.level):
            tiles_in_view = []
            for offset in range(-depth, depth + 1):
                if self.direction == Direction.NORTH:
                    tx = (x + offset) % self.map.size
                    ty = (y - depth) % self.map.size
                elif self.direction == Direction.SOUTH:
                    tx = (x + offset) % self.map.size
                    ty = (y + depth) % self.map.size
                elif self.direction == Direction.EAST:
                    tx = (x + depth) % self.map.size
                    ty = (y + offset) % self.map.size
                elif self.direction == Direction.WEST:
                    tx = (x - depth) % self.map.size
                    ty = (y + offset) % self.map.size
                tile = self.map.tiles[ty][tx]
                tile_contents = []
                for res, qty in tile.items():
                    if qty > 0:
                        tile_contents.extend([res] * qty)
                tiles_in_view.append(" ".join(tile_contents)
                                     if tile_contents else "")
            vision.extend(tiles_in_view)
        self.add_res(f"[{', '.join(vision)}]")

    def inventory_cmd(self):
        inv_items = [f"{res} {qty}" for res,
                     qty in self.inventory.items() if qty > 0]
        self.add_res(f"[{', '.join(inv_items)}]")

    def broadcast(self, text: str):
        # Calculate direction for each player
        for player in self.map.players:
            if player is self:
                # Same tile
                player.add_res("message 0, " + text)
                continue

            # Calculate relative position considering toroidal map
            dx = (self.position[0] - player.position[0]) % self.map.size
            if dx > self.map.size // 2:
                dx -= self.map.size

            dy = (self.position[1] - player.position[1]) % self.map.size
            if dy > self.map.size // 2:
                dy -= self.map.size

            # Calculate angle in radians
            angle_rad = math.atan2(dy, dx)

            # Convert to direction based on player's orientation
            if player.direction == Direction.NORTH:
                player_angle = -math.pi/2
            elif player.direction == Direction.EAST:
                player_angle = 0
            elif player.direction == Direction.SOUTH:
                player_angle = math.pi/2
            else:  # WEST
                player_angle = math.pi

            relative_angle = angle_rad - player_angle
            if relative_angle < 0:
                relative_angle += 2 * math.pi

            relative_angle_deg = math.degrees(relative_angle) % 360

            if relative_angle_deg <= 22.5 or relative_angle_deg > 337.5:
                direction_num = 1
            elif relative_angle_deg <= 67.5:
                direction_num = 2
            elif relative_angle_deg <= 112.5:
                direction_num = 3
            elif relative_angle_deg <= 157.5:
                direction_num = 4
            elif relative_angle_deg <= 202.5:
                direction_num = 5
            elif relative_angle_deg <= 247.5:
                direction_num = 6
            elif relative_angle_deg <= 292.5:
                direction_num = 7
            else:
                direction_num = 8

            player.add_res(f"message {direction_num}, {text}")
        self.add_res("ok")

    def connect_nbr(self):
        self.add_res(str(self.map.nb_eggs(self.team)))

    def fork(self):
        self.map.eggs_tile[self.team].append(
            map.Coordinate(self.position[0], self.position[1]))
        self.add_res("ok")

    def eject(self):
        x, y = self.position
        ejected_players = []
        for player in self.map.players:
            if player is not self and player.position == (x, y):
                px, py = player.position
                if self.direction == Direction.NORTH:
                    player.position = (px, (py - 1) % self.map.size)
                elif self.direction == Direction.EAST:
                    player.position = ((px + 1) % self.map.size, py)
                elif self.direction == Direction.SOUTH:
                    player.position = (px, (py + 1) % self.map.size)
                elif self.direction == Direction.WEST:
                    player.position = ((px - 1) % self.map.size, py)
                ejected_players.append(player)
        self.add_res("ok" if ejected_players else "ko")

    def take(self, resource: str):
        if resource not in [res.value for res in Resource]:
            return self.add_res("ko")
        x, y = self.position
        if self.map.tiles[y][x][RESOURCE_ENUM[resource]] <= 0:
            return self.add_res("ko")
        self.map.tiles[y][x][RESOURCE_ENUM[resource]] -= 1
        self.inventory[resource] += 1
        self.add_res("ok")

    def set(self, resource: str):
        if resource not in [res.value for res in Resource] or self.inventory.get(resource) <= 0:
            return self.add_res("ko")
        x, y = self.position
        self.inventory[resource] -= 1
        self.map.tiles[y][x][RESOURCE_ENUM[resource]] += 1
        self.add_res("ok")

    def incantation(self):
        if self.level >= 8:
            return self.add_res("ko")
        req = ElevationRequirement.requirements.get(self.level)
        if not req:
            return self.add_res("ko")
        players_required, resources_required = req
        x, y = self.position
        players_on_tile = [p for p in self.map.players if p.position == (
            x, y) and p.level == self.level and p is not self]
        if len(players_on_tile) + 1 < players_required:  # +1 for self
            return self.add_res("ko")
        tile = self.map.tiles[y][x]
        for resource, count in resources_required.items():
            if tile[resource] < count:
                return self.add_res("ko")
        for resource, count in resources_required.items():
            tile[resource] -= count
        self.map.incanted_tile[(x, y)] = map.IncantationTile(
            CommandTick[Player.incantation], self.level)
        self.add_res("Elevation underway\n")


Command = {
    "Forward": Player.forward,
    "Right": Player.right,
    "Left": Player.left,
    "Look": Player.look,
    "Inventory_cmd": Player.inventory_cmd,
    "Broadcast": Player.broadcast,
    "Connect_nbr": Player.connect_nbr,
    "Fork": Player.fork,
    "Eject": Player.eject,
    "Take": Player.take,
    "Set": Player.set,
    "Incantation": Player.incantation
}

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
