
""" Global Imports """
import io
import math
import torch
import base64
import torch as t
import numpy as np

from collections import Counter
from torch.distributions import Categorical

""" Local Imports """
from sources.brain import GolemBrain

def generate_2d_sinusoidal_pe(height, width, d_model):
    assert d_model % 2 == 0, "d_model must be even"
    pe = torch.zeros(height, width, d_model)

    # Position indices
    y_pos = torch.arange(0, height, dtype=torch.float32).unsqueeze(1)
    x_pos = torch.arange(0, width, dtype=torch.float32).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))

    # X encodings
    pe[:, :, 0::4] = torch.sin(x_pos * div_term).T.unsqueeze(0).repeat(height, 1, 1)
    pe[:, :, 1::4] = torch.cos(x_pos * div_term).T.unsqueeze(0).repeat(height, 1, 1)

    # Y encodings
    pe[:, :, 2::4] = torch.sin(y_pos * div_term).T.unsqueeze(1).repeat(1, width, 1)
    pe[:, :, 3::4] = torch.cos(y_pos * div_term).T.unsqueeze(1).repeat(1, width, 1)

    return pe  # shape: (height, width, d_model)

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

class NetworkTensor:
    
    def __init__(
        self,
        data : str | t.Tensor
    ):
        """
        Args:
            data (str | t.Tensor): If it's a tensor, then we will serialize it
            # If it's a serialized version, we deserialize it
        """
        self.data = data

    def deserialize(self):
        
        def apply(data : str):
            b = base64.b64decode(data.encode("ascii"))
            return torch.load(io.BytesIO(b), map_location="cpu")

        return self.data if isinstance(self.data, t.Tensor) else apply(self.data)
        
    def serialize(self):
        
        def apply(data : t.Tensor):
            buf = io.BytesIO()
            t.save(data, buf)
            
            return base64.b64encode(buf.getvalue()).decode("ascii")
        
        return self.data if isinstance(self.data, str) else apply(self.data)

class Golem:
    
    possible_actions = ["Broadcast", "Fork", "Incantation", "Eject"]
    action_cost = [1, 6, 43, 1]
    
    orientations        = ["north", "west", "south", "east"]
    orientation_start   = [[-1,-1],[-1,-1],[1,-1],[-1,1]]
    orientation_end     = [[-1,1],[1,-1],[1,1],[1,1]]
    ejection_update     = [[-1,0],[0,-1],[1,0],[0,1]]
    
    # level_up_restrictions = [
    #     [0, 1, 0, 0, 0, 0, 0, 1],
    #     [0, 1, 1, 1, 0, 0, 0, 2],
    #     [0, 2, 0, 1, 0, 2, 0, 2],
    #     [0, 1, 1, 2, 0, 1, 0, 4],
    #     [0, 1, 2, 1, 3, 0, 0, 4],
    #     [0, 1, 2, 3, 0, 1, 0, 6],
    #     [0, 2, 2, 2, 2, 2, 1, 6]
    # ]
    level_up_restrictions = [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 2, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
        [0, 1, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 1, 2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0],
        [0, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6]
    ]
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
        observation_threshold : int = 3,
        circular_view : int = 3,
        decaying_factor : float = 0.9
    ):
        assert pos.size() == (2,), "Invalid start position given"
        
        self.team_index : int = team_index
        
        self.inventory : t.Tensor = t.zeros(7, dtype=t.int32)
        # Insert 10 food by default
        self.inventory[self.food_position] += 10
        self.map = t.zeros(d, d, len(self.resources))
        self.reliance_map = t.zeros(d, d)
        self.orientation = 0
        self.pos = pos
        self.gamma = decaying_factor

        # Position where the model calculated it should go        
        self.future_position = pos
        self.map_size = d
        self.observation_threshold = observation_threshold
        self.reliability_level = 0
        self.view_circle = circular_view
        
        # How many turns until the agent can act again
        self.frozen = 0

        filter = np.zeros((d * 2 - 1, d * 2 - 1))
        filter[d - 1] = np.concatenate([((d - np.arange(d)) + 1)[:-1], np.arange(d)])
        corners = t.arange(d * 2).as_strided((d - 1, d - 1), (1, 1)).numpy() + 3
        filter[d:,d:] = corners
        filter[:d - 1,d:] = np.rot90(corners)
        filter[:d - 1,:d-1] = np.rot90(np.rot90(corners)) + 1
        filter[d:,:d-1] = np.rot90(np.rot90(np.rot90(corners))) + 1
        filter[:,d - 1] = np.concatenate([(d - np.arange(d-1)), np.array([0]), (np.arange(d-1) + 2)])

        self.filters = {}

        for i in range(1, 4):
            filter = np.rot90(filter)
            self.filters[i % 4] = t.tensor(filter)
            
        self.pe_grid = generate_2d_sinusoidal_pe(84, 84, d_model=64)  # or another dim
        
        self.level = 1
        self.alive = True
        
        self.running_actions = []

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
        n_action : int
    ):
        """

        Args:
            brain (GolemBrain): _description_
            n_action (int): _description_

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
        
        if self.frozen > 0:
            return -1, None
        
        if self.reliability_level <= 0:
            self.reliability_level = self.observation_threshold
            # Forced
            return self.map_size * self.map_size + 6, None
        
        # TODO: Auto broadcast when seeing other agent aka (player >= 1 and reliance == gamma because it's next turn)
        if ((self.map[:,:,self.player_position] == 1) & (self.reliance_map == self.gamma)).any():
            return self.map_size * self.map_size, None

        d = self.map_size
        # Add agent info in the map before passing to brain
        # Add PE
        map_copy = self.map.clone()
        map_copy[*self.pos][self.agent_inventory_start:self.agent_inventory_end] += self.inventory
        
        pe = self.get_agent_window()
        
        tiles = self.map_size * self.map_size
        
        full_action_space = brain(map_copy + pe) # N * N + 7
        full_action_space[:tiles] *= self.filters[self.orientation][d - self.pos[0]:2 * d - self.pos[0], d - self.pos[1]: 2 * d - self.pos[1]].reshape(d * d)
        full_action_space[:tiles] *= self.reliance_map.reshape(d * d)
        
        # Put logits of impossible actions to 0
        # Incant (if cant incant, remove logit)
        full_action_space[tiles + 2] = full_action_space[tiles + 2] if (map_copy[*self.pos] >= self.level_up_restrictions[self.level - 1]) else -t.inf
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
        distribution = Categorical(probs=t.softmax(full_action_space))
        actions = distribution.sample((n_action,))
        
        # actions, probabilities
        return actions, -distribution.log_prob(actions)
    
    def raw_distribution_to_action(
        self,
        actions : list[int] | int
    ):
        actions = [actions] if isinstance(actions, int) else actions   
        
        final_actions = []        
        
        for action in actions:
            if action == -1:
                final_actions.append("Do Nothing")
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
                    final_actions.append(f"Broadcast {NetworkTensor(t.concat([self.inventory, self.map.flatten(), self.reliance_map.flatten(), self.pos]))}")
                elif offseted_action < len(self.possible_actions):
                    final_actions.append(self.possible_actions[offseted_action])
                    self.frozen += self.action_cost[offseted_action]
                elif offseted_action in range(len(self.possible_actions), len(self.possible_actions) + 7):
                    final_actions.append(f"Take {self.resources[offseted_action - len(self.possible_actions)]}")
                elif offseted_action in range(len(self.possible_actions) + 7, len(self.possible_actions) + 14):
                    final_actions.append(f"Set {self.resources[offseted_action - len(self.possible_actions)]}")
                else:
                    final_actions.append("Look")
        
        self.running_actions = final_actions
        
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
                self.receive_broadcast(message[:inventory_offset], message[inventory_offset:map_offset].reshape(self.map_size, self.map_size, len(self.resources)), message[map_offset: reliance_map_offset].reshape(self.map_size, self.map_size), message[reliance_map_offset:])
            elif response.split()[0] == "eject: ":
                self.ejected(int(response.split()[1][1:]))
            elif response.split(":")[0] == "Current level":
                self.level = int(response.split(": ")[1])
            else:
                action = self.running_actions.pop(0)
                if action == "Look":
                    self.observation_results([tile.split(" ") for tile in response[1:-1].split(", ")])
                if action.split()[0] == "Take" and response == "ok":
                    self.map[*self.pos] += (t.arange(len(self.resources)) == self.resources.index(action.split()[1]))
                if action.split()[0] == "Set" and response == "ok":
                    self.map[*self.pos] -= (t.arange(len(self.resources)) == self.resources.index(action.split()[1]))
                    
                
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
        pos : t.Tensor
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
        self.reliance_map[*pos] = 1