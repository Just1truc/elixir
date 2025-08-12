import os
import torch as t

from sources.brain import GolemBrain
from sources.golem import Golem

from server.environment import Environment
from server.player import Player

class ElixirConfig:
    
    def __init__(
        self,
        n_steps : int = 100,
        map_size : int = 42,
        info_decay : float = 0.9,
        circular_view : int = 3,
        n_trajectories : int = 32,
        observation_step : int = 3
    ):        
        
        self.n_steps    : int   = n_steps
        self.map_size   : int   = map_size
        self.info_decay : float = info_decay
        
        self.circular_view  : int   = circular_view
        self.n_trajectories : int   = n_trajectories
        
        self.observation_steps  : int        = observation_step
        

class Trajectory:
    
    def __init__(
        self,
        env : Environment,
        elixir_config : ElixirConfig
    ):
    
        self.env = env.clone()
        self.golems = [
            Golem(
                team_index=player.id,
                d=elixir_config.map_size,
                pos=player.position,
                observation_threshold=elixir_config.observation_steps,
                circular_view=elixir_config.observation_steps,
                decaying_factor=elixir_config.info_decay
            ) for player in env.players
        ]
        self.config = elixir_config
        
    @property
    def players(self) -> list[Player]:
        self.env.players
        
    def new_golem(
        self,
        position : t.Tensor,
        team_id : int
    ):
        self.golems.append(Golem(
            team_index=team_id,
            d=self.config.map_size,
            pos=position,
            observation_threshold=self.config.observation_steps,
            circular_view=self.config.observation_steps,
            decaying_factor=self.config.info_decay
        ))
        
    # Method to sample all golems from a trajectory
    def sample(
        self,
        brain : GolemBrain
    ):
        actions     = []
        log_probs   = []
        
        # Get all logprob
        for golem in self.golems:
            action, log_prob = golem.sample_actions(brain)
            actions.append(golem.raw_distribution_to_action(action))
            log_probs.append(log_prob)
            
            # End turn
            golem.tick_end()
            
        return actions, log_probs
    
    def step(
        self
    ):
        # Apply a step to the universe
        return self.env.step()

class Elixir:
    
    # Trainer
    
    def __init__(
        self,
        env : Environment,
        elixir_config : ElixirConfig,
        pretrained_brain : str | None = None
    ):  
        assert (pretrained_brain == None or os.path.exists(pretrained_brain)), f"Pretrained brain path {pretrained_brain} given does not exist"
        
        # TODO: Implement the pretrained_brain loading
        
        self.brain = GolemBrain()
        self.env : Environment = env
        self.trajectories = [
            Trajectory(env=env, elixir_config=elixir_config) for _ in range(elixir_config.n_trajectories)
        ]
    
    # Make a clone method for Golems 
    # Add the train method
    # Use ppo clip algorithm for update:
    # https://spinningup.openai.com/en/latest/algorithms/ppo.html
    # Convert the output of sample trajectories into meaningful actions Ex: (0,1,2,3,4) -> ("Look", "broadcast", ...)
    # Make the mathod to handle forking
    # Forking creates a new Agent that will be in the trajectory it was born in
    # To sample a trajectory:
    # Use the sample action method with the number of trajectories
    # Clone the (golem, env) n_trajectory time.
    # Apply the actions of each trajectory to each env (Each golem that have the same trajectory index will be in the same env)
    # Make reward function
    # Decide what to reward
    # Use the log prob to update the weight by using Adam + kl div (see link above for details)
    
    # The Elixir Class will take an environment as parameter,
    # When training, we will prompt set an environement
    # Then we project trajectories
    # When ask the environment for rewards (env dependent rewards)
    # 
        
    def train(
        self,
        epochs : int = 25,
        lr : int = 1e-5
    ):
        
        for epoch in range(epochs):
            
            # Sample All Golems from all env
            for id, trajectory in enumerate(self.trajectories):
                t_actions, t_logprobs = trajectory.sample(self.brain)
                
                # TODO: Add all the forks,
                # Add the loop to give the answer to the golems
                # Add the math optimization
                
                # Queuing all golem commands
                # TODO: Add check on alive or not
                for golem_id, actions in enumerate(t_actions):
                    for action in actions:
                        if action == "Do Nothing":
                            continue
                        trajectory.players[golem_id].add_cmd(action)
                    
                state, rewards, terminated = trajectory.step()
                # Fork + ok, spawn new golem based on the one that forked
                for golem_id in range(len(t_actions)):

                    all_res = []
                    while (response := trajectory.players[golem_id].get_res()):
                        all_res.append(response)
                        if trajectory.golems[golem_id].running_actions[0] == "Fork" and response == "ok":
                            trajectory.new_golem(
                                trajectory.golems[golem_id].pos,
                                trajectory.golems[golem_id].team_index
                            )
                
                    # Send results of all commands to Golems
                    trajectory.golems[golem_id].read_player_queue(all_res)
                    
                # Calculate rewards + opti
                
                
            # Fork -> Elixir (Frozen for 6 ticks)
            # Broadcast -> Golem
            # Incantation -> Golem (Frozen for 43 ticks)
            # Eject -> Golem
            # Take -> Golem
            # Set -> Golem
            # Look -> Golem
            
            # Players don't know if they have been ejected
            # 
            
            # Add golem choices into players
            # Scrap Answers from Server
            # Feed answers to players