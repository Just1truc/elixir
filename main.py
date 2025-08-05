import os
import torch as t
import numpy as np

from torch import nn
from collections import Counter

from torch.distributions import Categorical

import torch
import math

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

class GolemBrain(nn.Module):
    
    def __init__(
        self,
        d_model : int = 15,
        n_head : int = 8,
        n_layers : int = 4,
        dim_feedforward : int = 128
    ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            n_head  = n_head,
            dim_feedforward = dim_feedforward,
            dropout = 0
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=n_layers
        )
        
        # Prediction Heads broadcast, fork, incant, push, pickup and set down
        self.action_heads   = nn.Linear(d_model, 6)
        self.heatmap_head   = nn.Linear(d_model, 1)

    def forward(
        self,
        x : t.Tensor,
        position : t.Tensor
    ):
        """
        Args:
            x (t.Tensor): (N, N, 15)
            returns (N * N + 7)
        """
        N, _, D = x.shape 
        out = self.encoder(x.reshape(N * N, D))
        
        return t.concat([self.heatmap_head(out), self.action_heads(out[*position])])
        # return t.softmax(t.concat([self.heatmap_head(out), self.action_heads(out[*position])]), dim=0)


class Golem:
    
    orientations = ["north", "west", "south", "east"]
    orientation_start = {
        0 : [-1, -1],
        1 : [-1, -1],
        2 : [ 1, -1],
        3 : [-1,  1]
    }
    orientation_end = {
        0 : [-1,  1],
        1 : [ 1, -1],
        2 : [ 1,  1],
        3 : [ 1,  1]
    }
    ejection_update = {
        0 : [-1, 0],
        1 : [0, -1],
        2 : [ 1, 0],
        3 : [ 0, 1]
    }
    
    level_up_restrictions = [
        [0, 1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 0, 2],
        [0, 2, 0, 1, 0, 2, 0, 2],
        [0, 4, 1, 1, 2, 0, 1, 4],
        [0, 4, 1, 2, 1, 3, 0, 4],
        [0, 1, 2, 3, 0, 1, 0, 6],
        [0, 2, 2, 2, 2, 2, 1, 6]
    ]
    
    resources = ["food", "linemate", "deraumere", "sibur", "mendiane", "phiras", "thystame", "player", "", "", "", "", "", "", ""]
    
    def __init__(
        self,
        slot_index : int,
        team_index : int,
        d : int,
        pos : t.Tensor,
        observation_threshold : int = 3,
        circular_view : int = 3,
        decaying_factor : float = 0.9
    ):
        assert pos.size() == (2,), "Invalid start position given"
        
        self.slot_index : int = slot_index
        self.team_index : int = team_index
        
        self.inventory : t.Tensor = t.zeros(7, dtype=t.int32)
        # Insert 10 food by default
        self.inventory[0] += 10
        self.map = t.zeros(d, d, len(self.resources))
        self.reliance_map = t.zeros(d, d)
        self.orientation = 0
        self.pos = pos
        self.gamma = decaying_factor

        # Position where the model calculated it should go        
        self.future_position = pos
        self.map_size = d
        self.observation_threshold = observation_threshold
        self.reliability_level = observation_threshold
        self.view_circle = circular_view

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
            self.map[*[(t.arange(ax) + offset_start_pos[id]) % self.map_size for id, ax in enumerate(end_pos - start_pos)]] = vectorized_vision[offset:offset + n]
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
        
        if self.reliability_level <= 0:
            self.reliability_level = 3
            # Forced
            return 0, None
        
        # TODO: Auto broadcast when seeing other agent aka (player >= 1 and reliance == gamma because it's next turn)
        if ((self.map[:,:,7] == 1) & (self.reliance_map == self.gamma)).any():
            return 1, None

        d = self.map_size
        # Add agent info in the map before passing to brain
        # Add PE
        map_copy = self.map.clone()
        map_copy[*self.pos][8:] += self.inventory
        
        pe = self.get_agent_window()
        
        full_action_space = brain(map_copy + pe) # N * N + 7
        full_action_space[:self.map_size * self.map_size] *= self.filters[self.orientation][d - self.pos[0]:2 * d - self.pos[0], d - self.pos[1]: 2 * d - self.pos[1]].reshape(d * d)
        full_action_space[:self.map_size * self.map_size] *= self.reliance_map.reshape(d * d)
        
        # Put logits of impossible actions to 0
        # Incant (if cant incant, remove logit)
        full_action_space[self.map_size + 2] = (map_copy[*self.pos][:7] >= self.level_up_restrictions[self.level - 1]).all().int() * full_action_space[self.map_size + 2]
        # Push/Eject
        full_action_space[self.map_size + 3] = (map_copy[*self.pos][7] >= 1).int() * full_action_space[self.map_size + 3]
        # Take object
        full_action_space[self.map_size + 4] = (map_copy[*self.pos][:6].sum() > 0).int() * full_action_space[self.map_size + 3]
        # Set Object
        full_action_space[self.map_size + 4] = (map_copy[*self.pos][8:].sum() > 0).int() * full_action_space[self.map_size + 4]

        # Sample trajectories
        # look, broadcast, fork, incant, push, pickup and set down
        distribution = Categorical(probs=t.softmax(t.log(full_action_space)))
        actions = distribution.sample((n_action,))
        
        # actions, probabilities
        return actions + 1, -distribution.log_prob(actions)
    
    def ejected(
        self,
        direction : int
    ):
        # 1,3,5,7 => pushed backward, right, forward, left
        directions = [1,3,5,7]
        rotation_number = directions.index(direction)
        # self.ejection_update[self.orientation + ]
        ... 
        
    def receive_broadcast(
        self,
        broadcast_type : str,
        inventory : t.Tensor,
        map : t.Tensor,
        reliance_map : t.Tensor,
        pos : t.Tensor
    ):
        assert broadcast_type in self.broadcasts, "Invalid Broadcast Format Message Received"
        
        # Free Broadcast (agent choice) (inventory,map,)
        # Strategic Broadcast
        # All broadcast: (inventory,map,pos)
        
        # Update map with more reliable info
        self.map = (reliance_map > self.reliance_map) * map + (1 - (reliance_map > self.reliance_map)) * self.map
        self.reliance_map = t.max(reliance_map, self.reliance_map)
        
        # Add sending agent on map
        map[*pos][8:] += inventory
        map[*pos][7] += 1
        self.reliance_map[*pos] = 1

class Elixir:
    
    def __init__(
        self,
        n_teams : int = 2,
        n_slots : int = 10,
        pretrained_brain : str | None = None
    ):  
        assert (pretrained_brain == None or os.path.exists(pretrained_brain)), f"Pretrained brain path {pretrained_brain} given does not exist"
        
        self.golem_brain = GolemBrain()
        self.golems = [Golem(id) for slot_id in range(n_slots) for team_id in n_teams]
        
import tkinter as tk

# Assign fixed color for each resource
RESOURCE_COLORS = {
    "food": "#FF5733",
    "linemate": "#FFC300",
    "deraumere": "#DAF7A6",
    "sibur": "#8E44AD",
    "mendiane": "#3498DB",
    "phiras": "#1ABC9C",
    "thystame": "#2ECC71",
    "player": "#000000"
}

def draw_golem_map(golem : Golem, cell_size=40):
    map_tensor = golem.map
    h, w, r = map_tensor.shape
    assert r == 8, "Expected 8 resource layers"
    
    root = tk.Tk()
    root.title("Golem Map Viewer")

    canvas = tk.Canvas(root, width=w * cell_size, height=h * cell_size, bg='white')
    canvas.pack()

    for y in range(h):
        for x in range(w):
            # Draw grid cell
            x0, y0 = x * cell_size, y * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            val = 1 - golem.reliance_map[y, x]
            intensity = int(255 * val)
            color = f'#ff{intensity:02x}{intensity:02x}'
            canvas.create_rectangle(x0, y0, x1, y1, outline='gray', fill=color)

            # Draw resource indicators inside the cell
            count = 0
            for idx, resource in enumerate(Golem.resources):
                value = int(map_tensor[y, x, idx].item())
                if value > 0:
                    # Dot size depends on count
                    dot_size = min(6 + value * 2, cell_size // 2)
                    dx = count % 3
                    dy = count // 3
                    cx = x0 + 5 + dx * 12
                    cy = y0 + 5 + dy * 12
                    canvas.create_oval(
                        cx, cy, cx + dot_size, cy + dot_size,
                        fill=RESOURCE_COLORS.get(resource, "gray"),
                        outline=""
                    )
                    count += 1
                    

    root.mainloop()

d = 10
golem_test = Golem(slot_index=2, team_index=0, d=d, pos=t.tensor([0, 1]))
golem_test.observation_results([["food", "linemate", "deraumere"], ["food", "phiras"], ["phiras"], [], [], ["phiras"], ["food"], [], ["player"]])

# print(golem_test.map)
# draw_golem_map(golem_test)

for i in range(10):
    golem_test.tick_end()

draw_golem_map(golem_test)
