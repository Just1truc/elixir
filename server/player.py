from __future__ import annotations

import math
import random
from server import constant

from typing import TYPE_CHECKING, Callable, Optional

from server import map
from server.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from server.player import Command


class Player:
    def __init__(self, id: int, map: map.Map, team: str):
        """
        Initialize a Player instance with ID, map, and team.
        Args:
            id (int): Player ID.
            map (Map): Reference to the game map.
            team (str): Team name.
        """
        self.id: int = id
        self.team: str = team
        self.map: map.Map = map
        self.level: int = 1
        self.position: tuple = (random.randint(0, map.size - 1),
                         random.randint(0, map.size - 1))
        self.direction = constant.Direction.NORTH
        self.inventory = {res.value: 0 for res in constant.Resource}
        self.inventory[constant.Resource.FOOD.value] = 10  # Starting food
        # Current command state
        self.cur_cmd: Optional[Callable] = None
        self.cur_cmd_param: Optional[list[str]] = None
        self.remain_tick = 0
        self.lifetime_tick = 1
        self.is_alive = True
        # Queues
        self.command_queue: list[tuple[Callable, list[str]]] = []
        self.response_queue: list[str] = []
        logger.debug("Player created | id=%s team=%s pos=%s", self.id, self.team, self.position)

    def add_cmd(self, cmd: str):
        """
        Add a command to the player's command queue.
        Args:
            cmd (str): Command string to be parsed and added.
        """
        if not cmd:
            return
        cmd_func_param = self.get_cmd_func(cmd)
        if cmd_func_param is None:
            return
        func, param = cmd_func_param
        self.command_queue.append((func, param))

    @staticmethod
    def get_cmd_func(cmd: str):
        """
        Parse a command string and return the corresponding function and parameters.
        Args:
            cmd (str): Command string.
        Returns:
            tuple: (function, list of parameters) or None if not found.
        """
        words = cmd.split()
        if not words:
            return None
        func = Command.get(words[0])
        if func == None:
            print(cmd + " cmd does not exist")
            return None
        return func, words[1:]

    def add_res(self, res: str):
        """
        Add a response to the player's response queue.
        Args:
            res (str): Response string to add.
        """
        self.response_queue.append(res)

    def get_res(self) -> str:
        """
        Retrieve and remove the next response from the response queue.
        Returns:
            str: The next response, or None if the queue is empty.
        """
        if len(self.response_queue) != 0:
            return self.response_queue.pop(0)
        return None

    def clone(self):
        """
        Create a deep copy of the player, including state and queues.
        Returns:
            Player: A new Player instance with the same state.
        """
        p = Player(self.id, self.map, self.team)
        p.level = self.level
        p.direction = self.direction
        p.position = self.position  
        p.inventory = self.inventory.copy() 
        p.cur_cmd = self.cur_cmd
        p.cur_cmd_param = self.cur_cmd_param.copy() if self.cur_cmd_param else None
        p.remain_tick = self.remain_tick
        p.lifetime_tick = self.lifetime_tick
        p.is_alive = self.is_alive
        p.command_queue = self.command_queue.copy()  
        p.response_queue = self.response_queue.copy()
        logger.debug("Player cloned | id=%s team=%s pos=%s", p.id, p.team, p.position)
        return p
    
    def check_food(self):
        """
        Decrease food in inventory every 128 ticks if food is available.
        """
        if self.lifetime_tick % 128 == 0 and self.inventory[constant.Resource.FOOD.value] > 0:
            self.inventory[constant.Resource.FOOD.value] -= 1

    def forward(self):
        """
        Move the player forward in the direction they are facing, with map wrapping.
        """
        x, y = self.position
        size = self.map.size
        if self.direction == constant.Direction.NORTH:
            y = (y - 1) % size
        elif self.direction == constant.Direction.EAST:
            x = (x + 1) % size
        elif self.direction == constant.Direction.SOUTH:
            y = (y + 1) % size
        elif self.direction == constant.Direction.WEST:
            x = (x - 1) % size
        self.position = (x, y)
        self.add_res("ok")
        logger.debug("Player %s forward to %s", self.id, self.position)

    def right(self):
        """
        Turn the player to the right (clockwise).
        """
        self.direction = constant.Direction((self.direction.value + 1) % 4)
        self.add_res("ok")
        logger.debug("Player %s right dir=%s", self.id, self.direction)

    def left(self):
        """
        Turn the player to the left (counter-clockwise).
        """
        self.direction = constant.Direction((self.direction.value - 1) % 4)
        self.add_res("ok")
        logger.debug("Player %s left dir=%s", self.id, self.direction)

    def look(self):
        """
        Look around and return a simple representation of the current tile resources.
        """
        x, y = self.position
        vision = []
        for depth in range(self.level):
            tiles_in_view = []
            for offset in range(-depth, depth + 1):
                if self.direction == constant.Direction.NORTH:
                    tx = (x + offset) % self.map.size
                    ty = (y - depth) % self.map.size
                elif self.direction == constant.Direction.SOUTH:
                    tx = (x + offset) % self.map.size
                    ty = (y + depth) % self.map.size
                elif self.direction == constant.Direction.EAST:
                    tx = (x + depth) % self.map.size
                    ty = (y + offset) % self.map.size
                elif self.direction == constant.Direction.WEST:
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
        logger.debug("Player %s look pos=%s lvl=%s", self.id, self.position, self.level)

    def inventory_cmd(self):
        """
        Report the player's current inventory as a formatted string.
        """
        inv_items = [f"{res} {qty}" for res,
                     qty in self.inventory.items() if qty > 0]
        self.add_res(f"[{', '.join(inv_items)}]")
        logger.debug("Player %s inventory queried", self.id)

    def broadcast(self, text: str):
        """
        Broadcast a message to all players, indicating direction relative to each, using shortest toroidal path and trigonometric tile numbering.
        Args:
            text (str): The message to broadcast.
        """
        for player in self.map.players:
            if player is self:
                player.add_res("message 0, " + text)
                continue

            # Compute shortest toroidal vector from emitter (self) to receiver (player)
            sx, sy = self.position
            rx, ry = player.position
            size = self.map.size

            # Find minimal dx, dy (toroidal)
            dx_options = [rx - sx, rx - sx + size, rx - sx - size]
            dy_options = [ry - sy, ry - sy + size, ry - sy - size]
            dx = min(dx_options, key=lambda d: abs(d))
            dy = min(dy_options, key=lambda d: abs(d))

            # If on same tile, direction is 0
            if dx == 0 and dy == 0:
                player.add_res("message 0, " + text)
                continue

            # Rotate vector according to receiver's orientation (trigonometric, 1 is in front)
            # North: (0,-1), East: (1,0), South: (0,1), West: (-1,0)
            # We want to rotate the vector so that 'front' is always (0,-1)
            if player.direction == constant.Direction.NORTH:
                tx, ty = dx, dy
            elif player.direction == constant.Direction.EAST:
                tx, ty = -dy, dx
            elif player.direction == constant.Direction.SOUTH:
                tx, ty = -dx, -dy
            elif player.direction == constant.Direction.WEST:
                tx, ty = dy, -dx

            # Calculate angle from (0,0) to (tx,ty), 0 is front, increases counterclockwise
            angle = math.atan2(tx, -ty)  # y axis points down, so -ty
            angle_deg = (math.degrees(angle) + 360) % 360

            # Tile numbering: 1 is front, then counterclockwise (2: front-left, ..., 8: front-right)
            # Each sector is 45 degrees
            direction_num = int(((angle_deg + 22.5) % 360) // 45) + 1

            player.add_res(f"message {direction_num}, {text}")
        self.add_res("ok")
        logger.debug("Player %s broadcast '%s'", self.id, text)

    def connect_nbr(self):
        """
        Return the number of available connection slots (eggs) for the player's team.
        """
        self.add_res(str(self.map.nb_eggs(self.team)))
        logger.debug("Player %s connect_nbr", self.id)

    def fork(self):
        """
        Add an egg for the player's team at the player's current position.
        """
        self.map.eggs_tile[self.team].append(self.position)
        self.add_res("ok")
        logger.debug("Player %s fork at %s", self.id, self.position)

    def eject(self):
        """
        Eject other players from the current tile in the direction the player is facing.
        Also notifies ejected players with the opposite direction value (source of push).
        """
        moved = False
        size = self.map.size
        # Opposite direction of the ejection (where the push comes from)
        push_from = constant.Direction((self.direction.value + 2) % 4)
        for p in self.map.players:
            if p is self or p.position != self.position:
                continue
            x, y = p.position
            if self.direction == constant.Direction.NORTH:
                y = (y - 1) % size
            elif self.direction == constant.Direction.EAST:
                x = (x + 1) % size
            elif self.direction == constant.Direction.SOUTH:
                y = (y + 1) % size
            elif self.direction == constant.Direction.WEST:
                x = (x - 1) % size
            p.position = (x, y)
            # Notify ejected player of the direction of the push source
            p.add_res(f"eject: {push_from.value}")
            moved = True
        self.add_res("ok" if moved else "ko")
        logger.debug("Player %s eject from %s", self.id, self.position)

    def take(self, resource: str):
        """
        Take one unit of a resource from the current tile if available.
        """
        res_enum = constant.RESOURCE_ENUM.get(resource)
        if res_enum is None:
            self.add_res("ko")
            return
        x, y = self.position
        if self.map.tiles[y][x][res_enum] > 0:
            self.map.tiles[y][x][res_enum] -= 1
            self.inventory[res_enum.value] += 1
            self.add_res("ok")
        else:
            self.add_res("ko")
        logger.debug("Player %s take %s", self.id, resource)

    def set(self, resource: str):
        """
        Set (drop) one unit of a resource from inventory onto the current tile if available.
        """
        res_enum = constant.RESOURCE_ENUM.get(resource)
        if res_enum is None:
            self.add_res("ko")
            return
        if self.inventory.get(res_enum.value, 0) > 0:
            self.inventory[res_enum.value] -= 1
            x, y = self.position
            self.map.tiles[y][x][res_enum] += 1
            self.add_res("ok")
        else:
            self.add_res("ko")
        logger.debug("Player %s set %s", self.id, resource)

    def incantation(self):
        """
        Attempt to start an incantation on the current tile if requirements are met.
        """
        lvl = self.level
        req_players, req_resources = constant.ElevationRequirement.requirements.get(lvl, (0, {}))
        x, y = self.position
        # Players of same level on tile
        same_tile_players = [p for p in self.map.players if p.position == (x, y) and p.level == lvl]
        # Check resources on tile
        tile = self.map.tiles[y][x]
        has_resources = all(tile.get(r, 0) >= cnt for r, cnt in req_resources.items())
        if len(same_tile_players) >= req_players and has_resources:
            # Consume resources
            for r, cnt in req_resources.items():
                tile[r] -= cnt
            # Start incantation
            self.map.incanted_tile[(x, y)] = map.IncantationTile(remaining_ticks=300, level=lvl)
            for p in same_tile_players:
                p.add_res("Elevation underway")
        else:
            self.add_res("ko")
        logger.debug("Player %s incantation at %s lvl=%s", self.id, self.position, self.level)


"""
Command

Maps command names (as strings) to their corresponding Player class methods. Used to dispatch player actions based on command input.
"""
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

"""
CommandTick

Maps Player class methods to the number of ticks (time units) required to execute each command. Used for scheduling and command timing.
"""
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
