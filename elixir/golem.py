
""" Global Imports """
import math
import torch
import torch as t

from collections import Counter
from torch.distributions import Categorical
from matplotlib import pyplot as plt

""" Local Imports """
from elixir.brain import GolemBrain
from elixir.filters import PosFilter
from elixir.network import NetworkTensor

def generate_2d_sinusoidal_pe(height, width, d_model):
    assert d_model % 4 == 0, "d_model must be divisible by 4"
    pe = torch.zeros(height, width, d_model, dtype=torch.float32)

    # position indices
    y = torch.arange(height, dtype=torch.float32)          # (H,)
    x = torch.arange(width, dtype=torch.float32)           # (W,)

    half = d_model // 2                                     # split half for x, half for y
    div_term = torch.exp(torch.arange(0, half, 2, dtype=torch.float32)
                         * (-math.log(10000.0) / half))     # (half/2,) = (d_model/4,)

    # X encodings: shape (1, W, d_model/4) -> broadcast to (H, W, d_model/4)
    x_term = x[None, :, None] * div_term[None, None, :]
    pe[:, :, 0::4] = torch.sin(x_term).expand(height, width, -1)
    pe[:, :, 1::4] = torch.cos(x_term).expand(height, width, -1)

    # Y encodings: shape (H, 1, d_model/4) -> broadcast to (H, W, d_model/4)
    y_term = y[:, None, None] * div_term[None, None, :]
    pe[:, :, 2::4] = torch.sin(y_term).expand(height, width, -1)
    pe[:, :, 3::4] = torch.cos(y_term).expand(height, width, -1)

    return pe  # (height, width, d_model)

R_left  = torch.tensor([[0., -1.],[1., 0.]])
R_right = torch.tensor([[0.,  1.],[-1., 0.]])

def next_action_90(a, g, h, eps=1e-9):

    a = a.float(); g = g.float(); h = h.float()
    d = g - a
    if torch.allclose(d, torch.zeros_like(d)):
        return 'Forward'

    curr_d2 = (d*d).sum()
    new_d2  = ((g - (a + h))**2).sum()
    if new_d2 + eps < curr_d2:
        return 'Forward'

    t = d / (torch.norm(d) + eps)
    hL = (R_left  @ h)
    hR = (R_right @ h)

    dotL = (hL * t).sum()
    dotR = (hR * t).sum()
    if dotL >= dotR:
        return 'Left'
    else:
        return 'Right'
        
class Golem:
    
    possible_actions = ["Broadcast", "Fork", "Incantation", "Eject"]
    action_cost = [1, 6, 43, 1]
    
    orientations        = ["north", "west", "south", "east"]
    orientation_start   = t.tensor([[-1,-1],[-1,-1],[1,-1],[-1,1]])
    orientation_end     = t.tensor([[-1,1],[1,-1],[1,1],[1,1]])
    ejection_update     = t.tensor([[-1,0],[0,-1],[1,0],[0,1]])
    
    # level_up_restrictions = [
    #     [0, 1, 0, 0, 0, 0, 0, 1],
    #     [0, 1, 1, 1, 0, 0, 0, 2],
    #     [0, 2, 0, 1, 0, 2, 0, 2],
    #     [0, 1, 1, 2, 0, 1, 0, 4],
    #     [0, 1, 2, 1, 3, 0, 0, 4],
    #     [0, 1, 2, 3, 0, 1, 0, 6],
    #     [0, 2, 2, 2, 2, 2, 1, 6]
    # ]
    level_up_restrictions = t.tensor([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 2, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
        [0, 1, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 1, 2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0],
        [0, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6]
    ])
    # TileResource("food", "linemate", "deraumere", "sibur", "mendiane", "phiras", "thystame", "player"),
    # AgentInventory("food", "linemate", "deraumere", "sibur", "mendiane", "phiras", "thystame"),
    # PlayerLevels(player_lvl_0, player_lvl_1, player_lvl_2, player_lvl_3, player_lvl_4, player_lvl_5, player_lvl_6, player_lvl_7)
    resources = ["food", "linemate", "deraumere", "sibur", "mendiane", "phiras", "thystame", "player", *([""]* 14)]
    tile_resources_end = resources.index("player")
    agent_inventory_start = resources.index("player") + 1
    agent_inventory_end = agent_inventory_start + 7
    player_position = resources.index("player")
    food_position = resources.index("food")
    
    def __init__(
        self,
        team_index : int,
        d : int,
        pos : t.Tensor,
        start_level : int,
        pos_filter : PosFilter,
        d_model : int = 64,
        observation_threshold : int = 3,
        circular_view : int = 3,
        decaying_factor : float = 0.9,
        spawn_tick_index : int = 0,
        state_caching : bool = False,
    ):
        
        self.team_index : int = team_index
        
        self.inventory : t.Tensor = t.zeros(7, dtype=t.int32)
        # Insert 10 food by default
        self.inventory[self.food_position] += 10
        self.map = t.zeros(d, d, len(self.resources))
        self.reliance_map = t.zeros(d, d)
        self.orientation = 0
        self.pos = t.tensor(pos) if not(isinstance(pos, t.Tensor)) else pos
        assert self.pos.size() == (2,), "Invalid start position given"
        self.gamma = decaying_factor
        
        self.spawn_tick_index = spawn_tick_index

        # Position where the model calculated it should go     
        self.map_size = d
        self.observation_threshold = observation_threshold
        self.reliability_level = 0
        self.view_circle = circular_view
        
        # How many turns until the agent can act again
        self.frozen = 0

        # filter = np.zeros((d * 2 - 1, d * 2 - 1))
        # filter[d - 1] = np.concatenate([((d - np.arange(d)) + 1)[:-1], np.arange(d)])
        # corners = t.arange(d * 2).as_strided((d - 1, d - 1), (1, 1)).numpy() + 3
        # filter[d:,d:] = corners
        # filter[:d - 1,d:] = np.rot90(corners)
        # filter[:d - 1,:d-1] = np.rot90(np.rot90(corners)) + 1
        # filter[d:,:d-1] = np.rot90(np.rot90(np.rot90(corners))) + 1
        # filter[:,d - 1] = np.concatenate([(d - np.arange(d-1)), np.array([0]), (np.arange(d-1) + 2)])

        # self.filters = {}

        # for i in range(1, 5):
        #     filter = np.rot90(filter)
        #     # print(filter)
        #     self.filters[i % 4] = t.tensor(filter.copy())
        #     # print(f"Filling filter {i}")
        self.filters = pos_filter
            
        self.pe_grid = generate_2d_sinusoidal_pe(84, 84, d_model=d_model)  # or another dim
        
        self.level = start_level
        self.alive = True
        
        self.running_actions = []
        self.state_caching = state_caching
        self.cache = None

    def get_agent_window(self):
        
        half = self.map_size // 2
        x0 = self.pos[0] + 42 - half
        y0 = self.pos[1] + 42 - half
        x1 = x0 + self.map_size
        y1 = y0 + self.map_size
        return self.pe_grid[y0:y1, x0:x1, :]  # shape: (42, 42, d_model)
    
    def tick_end(self):
        self.reliance_map *= self.gamma
        self.reliability_level -= 1
        self.frozen -= (self.frozen > 0)
        
    def observation_results(
        self,
        obtained_vision : list[list[str]]
    ):
        """Let's suppose the obtained_vision is as written in the subject: [0-15]

        Args:
            obtained_vision (list[list[str]]): Take a split version of the zappy vision format
        """
        
        # Translate api format to vector form 
        counts = [Counter(square) for square in obtained_vision]
        vectorized_vision = t.tensor([[count[resource_name] for resource_name in self.resources] for count in counts])
        
        # (n-1) + 2
        n = 1
        i = 0
        start_pos = self.pos
        end_pos = self.pos + 1
        offset = 0
        while offset < len(vectorized_vision):
            offset_start_pos = start_pos
            # If player on position, keep old info on player data
            # Else, remove info
            # self.map[*[(t.arange(ax) + offset_start_pos[id]) % self.map_size for id, ax in enumerate(end_pos - start_pos)]] = vectorized_vision[offset:offset + n]
            index = [(t.arange(ax) + offset_start_pos[id]) % self.map_size for id, ax in enumerate(end_pos - start_pos)]
            mask = self.map[index][:,self.player_position] != vectorized_vision[offset:offset + n][:,self.player_position]
            self.map[*index][:,:self.tile_resources_end] = vectorized_vision[offset:offset + n][:,:self.tile_resources_end]
            self.map[*index][:,self.agent_inventory_start:self.agent_inventory_end] = t.where(mask, vectorized_vision[offset:offset + n][:,self.agent_inventory_start:self.agent_inventory_end], self.map[*index][:,self.agent_inventory_start:self.agent_inventory_end])
            self.reliance_map[*[(t.arange(ax) + offset_start_pos[id]) % self.map_size for id, ax in enumerate(end_pos - start_pos)]] = t.ones(n)
            i += 1
            offset += n
            n = n + 2
            start_pos = start_pos + t.tensor(self.orientation_start[self.orientation])
            end_pos = end_pos + t.tensor(self.orientation_end[self.orientation])
        
    def sample_actions(
        self,
        brain : GolemBrain,
        n_action : int = 1,
        from_cache : tuple[t.Tensor, list[int]]  = None,
        display_filter : bool = False
    ):
        """

        Args:
            brain (GolemBrain): _description_
            n_action (int): _description_
            from_cache: (entry, actions)

        Returns:
            _type_: _description_
        """
        
        # Possible Actions
        # - Broadcast
        # - Fork
        # - Incant (Only if possible)
        # - Push (Only if possible)
        # - Decide on a tile to go to
        # Output distribution: map_size \times map_size || 3
        # Process: (((1 / inventory) * (Matrix)) * (Map * unreliability * distance_filter)) -> Food/Resource distribution Map,
        # This math is a subset to the math of the strategy generation below
        
        # If reliability_level <= 0:
        # Mandatory Observation
        # if Observation revealed agent, mandatory broadcast (pos, inv)
        # if received mandatory broadcast and distance < circular_view, broadcast (pos, inv)
        # if didn't recieve new message after sending message, apply strategy network
        # strategy network: group agents in minimal map, apply MultiLayerTransformer((Map||Inv + PE) * unreliability * distance_filter)
        # Apply concat strategy results with resource_map as a part of the possible actions
        # The Action like broadcast, fork, incant, push, pickup and set down, all depend on the embedding of the tile the bot is on 
        
        if from_cache == None:
            if self.frozen > 0:
                return -1, None, None

            if self.reliability_level <= 0:
                self.reliability_level = self.observation_threshold
                # Forced
                return self.map_size * self.map_size + 19, None, None

            # TODO: Auto broadcast when seeing other agent aka (player >= 1 and reliance == gamma because it's next turn)
            if ((self.map[:,:,self.player_position] == 1) & (self.reliance_map == self.gamma)).any():
                return self.map_size * self.map_size, None, None

            d = self.map_size
            # Add agent info in the map before passing to brain
            # Add PE
            map_copy = self.map.clone()
            map_copy[*self.pos][self.agent_inventory_start:self.agent_inventory_end] += self.inventory

            pe = self.get_agent_window()
            # print(pe.shape, map_copy.shape)

            tiles = self.map_size * self.map_size

            entry = brain.in_proj(map_copy) + pe
        else:
            entry = from_cache[0]
        
        full_action_space, critic = brain(entry, self.pos) # N * N + 7
        
        pos_filter = self.filters.get_view(self.pos, self.orientation)
        
        if display_filter:
            # self.pos = t.tensor([30,30])
            # pos_filter = self.filters[self.orientation][d - self.pos[0] - 1:2 * d - self.pos[0] - 1, d - self.pos[1] - 1: 2 * d - self.pos[1] - 1]
            plt.imshow(pos_filter)
            # print(pos_filter.shape)
            # print(self.pos)
            # print(pos_filter.shape)
            for x in range(d):
                for y in range(d):
                    plt.text(x, y, str(pos_filter[y][x].int().item()))
            plt.show()
        
        full_action_space[:tiles] *= pos_filter.reshape(d * d)
        full_action_space[:tiles] *= self.reliance_map.reshape(d * d)
        
        # Put logits of impossible actions to 0
        # Incant (if cant incant, remove logit)
        full_action_space[tiles + 2] = full_action_space[tiles + 2] if (map_copy[*self.pos] >= self.level_up_restrictions[self.level - 1]).all() else -t.inf
        # Push/Eject
        full_action_space[tiles + 3] = full_action_space[tiles + 3] if (map_copy[*self.pos][self.player_position] >= 1) else -t.inf
        
        # TODO: Redo because take object/set is now a distribution
        # Take object
        full_action_space[tiles + 4: tiles + 11] = t.where(map_copy[*self.pos][:self.tile_resources_end] > 0, full_action_space[tiles + 4:tiles + 11], -t.inf)
        # Set Object
        full_action_space[tiles + 11: tiles + 18] = t.where(map_copy[*self.pos][self.agent_inventory_start:self.agent_inventory_end] > 0, full_action_space[tiles + 11: tiles + 18], -t.inf)

        # Sample trajectories
        # n * n, broadcast, fork, incant, push, pickup and set down, look
        
        # TODO: move the distribution setup so that the values are 
        # print(full_action_space.shape)
        distribution = Categorical(probs=t.softmax(full_action_space, dim=0))
        # TODO: The cache must be the (entry and the actions)
        if self.state_caching and from_cache == None:
            actions = distribution.sample((n_action,))
            self.cache = (entry, actions)
        else:
            actions = from_cache[1]
        
        # actions, probabilities
        return actions, -distribution.log_prob(actions), critic
    
    def raw_distribution_to_action(
        self,
        actions : list[int] | int
    ):
        actions = [actions] if isinstance(actions, int) else actions   
        
        final_actions = []        
        
        for action in actions:
            if action == -1:
                final_actions.append("Do Nothing")
                continue
            # Decided to move
            if action < (self.map_size * self.map_size):
                # Decide action based on maximum 
                target_y = action // self.map_size
                target_x = action % self.map_size
                
                target_pos = t.tensor([target_y, target_x])
                heading_pos = self.ejection_update[self.orientation]
                agent_pos = self.pos
                
                final_actions.append(next_action_90(agent_pos, target_pos, heading_pos))
            else:
                offseted_action = action - self.map_size ** 2
                # TODO: In case of broadcast, send the serialized version as text
                if offseted_action == 0: # Means broadcast
                    final_actions.append(f"Broadcast {NetworkTensor(t.concat([self.inventory, self.map.flatten(), self.reliance_map.flatten(), self.pos, t.tensor([self.level])])).serialize()}")
                elif offseted_action < len(self.possible_actions):
                    final_actions.append(self.possible_actions[offseted_action])
                    self.frozen += self.action_cost[offseted_action]
                elif offseted_action in range(len(self.possible_actions), len(self.possible_actions) + 7):
                    final_actions.append(f"Take {self.resources[offseted_action - len(self.possible_actions)]}")
                elif offseted_action in range(len(self.possible_actions) + 7, len(self.possible_actions) + 14):
                    final_actions.append(f"Set {self.resources[offseted_action - len(self.possible_actions)]}")
                else:
                    final_actions.append("Look")
        
        self.running_actions += final_actions
        
        return final_actions

    def read_player_queue(
        self,
        response_queue : list[str]
    ):
        for response in response_queue:
            if response == "dead":
                self.alive = False
                return
            elif response.split()[0] == "message":
                message = NetworkTensor(response[11:]).deserialize()
                inventory_offset = self.inventory.shape[0]
                map_offset = inventory_offset + (self.map_size ** 2) * len(self.resources)
                reliance_map_offset = map_offset + (self.map_size ** 2)
                self.receive_broadcast(message[:inventory_offset], message[inventory_offset:map_offset].reshape(self.map_size, self.map_size, len(self.resources)), message[map_offset: reliance_map_offset].reshape(self.map_size, self.map_size), message[reliance_map_offset:-1], message[-1])
            elif response.split()[0] == "eject: ":
                self.ejected(int(response.split()[1][1:]))
            elif response.split(":")[0] == "Current level":
                self.level = int(response.split(": ")[1])
            else:
                action = self.running_actions.pop(0)
                if action == "Look":
                    self.observation_results([tile.split(" ") for tile in response[1:-1].split(", ")])
                if action.split()[0] == "Take" and response == "ok":
                    self.map[*self.pos] += (t.arange(len(self.resources)) == self.resources.index(action.split()[1])).int()
                if action.split()[0] == "Set" and response == "ok":
                    self.map[*self.pos] -= (t.arange(len(self.resources)) == self.resources.index(action.split()[1])).int()

    def copy(self):
        
        golem = Golem(
            team_index=self.team_index,
            d=self.map_size,
            pos=self.pos,
            observation_threshold=self.observation_threshold,
            circular_view=self.view_circle,
            decaying_factor=self.gamma
        )
        golem.map = self.map.clone()
        golem.reliance_map = self.reliance_map.clone()
        golem.inventory = self.inventory.clone()
        golem.level = self.level
        golem.frozen = self.frozen
        golem.alive = self.alive
        golem.running_actions = self.running_actions
        golem.reliability_level = self.reliability_level
        
        return golem
    
    def ejected(
        self,
        direction : int
    ):
        # 1,3,5,7 => pushed backward, right, forward, left
        directions = [1,3,5,7]
        rotation_number = directions.index(direction)
        # self.ejection_update[self.orientation + ]
        self.pos += self.ejection_update[(rotation_number + self.orientation + 2) % 4]
        
    def receive_broadcast(
        self,
        inventory : t.Tensor,
        map : t.Tensor,
        reliance_map : t.Tensor,
        pos : t.Tensor,
        level : int
    ):
        # Free Broadcast (agent choice) (inventory,map,)
        # Strategic Broadcast
        # All broadcast: (inventory,map,reliance_map,pos)
        
        # Update map with more reliable info
        self.map = (reliance_map > self.reliance_map) * map + (1 - (reliance_map > self.reliance_map)) * self.map
        self.reliance_map = t.max(reliance_map, self.reliance_map)
        
        # Add sending agent on map
        map[*pos][self.agent_inventory_start:self.agent_inventory_end] += inventory
        map[*pos][self.player_position] += 1
        map[*pos][self.agent_inventory_end + level] += 1
        self.reliance_map[*pos] = 1