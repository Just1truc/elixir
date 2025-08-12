# import map
import random
# import constant
from server.map import Map
from server.player import Player, CommandTick
from server.constant import Resource

class Server:

    def __init__(self, size: int, nb_eggs: int, teams: list[str]):
        self.teams: list[str] = teams
        self.players: dict[int: Player] = {}
        self.map = Map(self.teams, self.players, nb_eggs, size)
        self.last_id = 0
        self.ticks = 0

    def clone(self):
        """Clone the server state"""
        new_server = Server(self.size, self.init_nb_connect, self.teams.copy())
        new_server.map = self.map.clone()
        new_server.time = self.time
        # Copy other server state as needed
        return new_server
    
    # Assume teams are the same if == to None
    def reset(self, nb_eggs: int, teams: list[str] = None):
        # self.teams is always initializated
        if teams:
            self.teams = teams
        self.players.clear()
        self.map.reset(nb_eggs)
        self.ticks = 0

    def execute_cmd(self, player: Player):
        if player.command_queue == [] or player.cur_cmd:
            return
        cmd_param = player.command_queue.pop(0)
        tick = CommandTick[cmd_param[0]]
        player.remain_tick = tick
        player.cur_cmd = cmd_param[0]
        player.cur_cmd_param = cmd_param[1]

    def execute_latent_cmd(self, player: Player):
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
        if player.inventory[Resource.FOOD.value] <= 0:
            player.is_alive = False

    def add_player(self, team: str) -> Player:
        if team not in self.teams:
            return None
        if len(self.map.eggs_tile[team]) == 0:
            return None
        i = random.randint(0, len(self.map.eggs_tile[team]) - 1)
        egg = self.map.eggs_tile[team][i]
        player = Player(self.last_id, self.map, team)
        player.position = (egg.x, egg.y)
        self.players[self.last_id] = player
        self.last_id += 1
        self.map.eggs_tile[team].pop(i)
        return player

    def step(self):
        for player in self.players.values():
            self.step_incantation()
            res = self.execute_latent_cmd(player)
            if res:
                player.add_res(res)
            player.remain_tick -= 1
            self.execute_cmd(player)
            player.lifetime_tick += 1
            player.check_food()
            self.check_death(player)
        self.ticks += 1
