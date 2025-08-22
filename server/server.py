"""
server.py

This module contains the Server class, which manages the game state, player actions, map updates, and team management for the Zappy server simulation. It provides methods for cloning, resetting, stepping through game ticks, and handling player and map logic.

The server can optionally be connected to an API system for monitoring and external control.
For API functionality, see the api/ package.
"""
import random
from server.player import Player, CommandTick
from server import map
from server import constant
from server.logging_config import get_logger

logger = get_logger(__name__)


class Server:
    def __init__(self, size: int, nb_eggs: int, teams: list[str]):
        """Initialize the Server with map size, number of eggs, and team names."""
        self.teams: list[str] = teams
        self.players: list[Player] = []
        self.map = map.Map(self.teams, self.players, nb_eggs, size)
        self.ticks = 0
        self.api = None  # Will be set by external API if needed
        logger.info("Server initialized | size=%s nb_eggs=%s teams=%s", size, nb_eggs, teams)

    def clone(self):
        """
        Create a deep copy of the current server state, including map, players, and teams.
        Returns:
            Server: A cloned Server instance with the same state.
        """
        s = Server(self.map.size, self.map.init_nb_eggs, self.teams.copy())
        s.map = self.map.clone()
        # copy teams just in case
        s.teams = self.teams.copy()
        s.players = [player.clone() for player in self.players]
        s.ticks = self.ticks
        logger.debug("Server cloned | players=%s ticks=%s", len(s.players), s.ticks)
        return s
    
    # Assume teams are the same if == to None
    def reset(self, nb_eggs: int = None, teams: list[str] = None):
        """
        Reset the server state for a new game or round.
        Args:
            nb_eggs (int): Number of eggs to reset on the map.
            teams (list[str], optional): New list of teams. If None, keep current teams.
        """
        if not nb_eggs:
            nb_eggs = self.map.init_nb_eggs
        if teams:
            self.teams = teams
        self.players.clear()
        self.map.reset(nb_eggs)
        self.ticks = 0
        logger.info("Server reset | nb_eggs=%s teams=%s", nb_eggs, self.teams)

    def execute_cmd(self, player: Player):
        """
        Execute the next command in the player's command queue if available.
        Args:
            player (Player): The player whose command is to be executed.
        """
        if player.command_queue == [] or player.cur_cmd:
            return
        cmd_param = player.command_queue.pop(0)
        tick = CommandTick[cmd_param[0]]
        player.remain_tick = tick
        player.cur_cmd = cmd_param[0]
        player.cur_cmd_param = cmd_param[1]

    def execute_latent_cmd(self, player: Player):
        """
        Execute the player's current command if its remaining tick is 1.
        Args:
            player (Player): The player whose latent command is to be executed.
        Returns:
            Any: The result of the command execution, or 0 if not executed.
        """
        if player.cur_cmd != None and player.remain_tick == 1:
            if player.cur_cmd_param == []:
                res = player.cur_cmd(player)
            else:
                res = player.cur_cmd(player, player.cur_cmd_param[0])
            player.cur_cmd = None
            player.cur_cmd_param = None
            player.remain_tick = 0
            return res
        return 0

    def step_incantation(self):
        """
        Progress all ongoing incantations on the map by one tick and evolve tiles if needed.
        """
        incantate = []
        for coords in self.map.incanted_tile:
            cur_tile = self.map.incanted_tile[coords]
            cur_tile.remaining_ticks -= 1
            if cur_tile.remaining_ticks == 0:
                self.map.evolve(cur_tile, coords)
                incantate.append(coords)
        for i in incantate:
            self.map.incanted_tile.pop(i)

    def check_death(self, player: Player):
        """
        Check if the player has run out of food and set is_alive to False if so.
        Args:
            player (Player): The player to check for death.
        """
        if player.is_alive and player.inventory[constant.Resource.FOOD.value] <= 0:
            player.is_alive = False
            logger.debug("Player %s died at tick %s", player.id, self.ticks)
            # Notify API if player just died
            if self.api:
                self.api.on_player_death(player.id)

    def add_player(self, team: str) -> Player | None:
        """
        Add a new player to the specified team if possible, placing them on a random egg tile.
        Args:
            team (str): The team to add the player to.
        Returns:
            Player: The newly added player, or None if not possible.
        """
        eggs = self.map.eggs_tile.get(team)
        if team not in self.teams or not eggs:
            logger.warning("Cannot add player | team=%s valid=%s eggs_available=%s", team, team in self.teams, bool(eggs))
            return None
        egg = random.choice(eggs)
        player = Player(len(self.players), self.map, team)
        player.position = egg
        self.players.append(player)
        eggs.remove(egg)
        logger.info("Player added | id=%s team=%s pos=%s", player.id, team, player.position)
        return player

    def step(self):
        """
        Advance the server state by one tick, processing all players and map incantations.
        """
        # Trigger API event at start of step
        if self.api:
            self.api.on_step_start()
        
        self.step_incantation()
        for player in self.players:
            res = self.execute_latent_cmd(player)
            if res:
                player.add_res(res)
            player.remain_tick -= 1
            self.execute_cmd(player)
            player.lifetime_tick += 1
            player.check_food()
            self.check_death(player)
        self.ticks += 1
        logger.debug("Tick advanced | ticks=%s players=%s", self.ticks, len(self.players))
        
        # Trigger API event at end of step
        if self.api:
            self.api.on_step_end()
