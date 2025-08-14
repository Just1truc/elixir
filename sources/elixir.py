import os
import torch as t

from torch import optim

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
        repeat_step : int = 5,
        circular_view : int = 3,
        n_trajectories : int = 32,
        observation_step : int = 3
    ):        
        
        self.n_steps    : int   = n_steps
        self.map_size   : int   = map_size
        self.info_decay : float = info_decay
        
        self.repeat_step    : int   = repeat_step
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
                start_level=player.level,
                observation_threshold=elixir_config.observation_steps,
                circular_view=elixir_config.observation_steps,
                decaying_factor=elixir_config.info_decay
            ) for player in env.players
        ]
        self.config = elixir_config
        self.state_cache = []
        
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
        values      = []
        
        # Get all logprob
        for golem in self.golems:
            
            # TODO: if golem is dead, I put a Do Nothing in the queue and None in log_prob/values
            
            if golem.alive == False:
                actions.append(["Do Nothing"])
                log_probs.append(None)
                values.append(None)
                continue
            
            action, log_prob, critic = golem.sample_actions(brain)
            actions.append(golem.raw_distribution_to_action(action))
            log_probs.append(log_prob)
            values.append(critic)
            
            # End turn
            golem.tick_end()
            
        return actions, log_probs, values
    
    # Method to sample all golems from a trajectory
    def sample_from_cache(
        self,
        brain : GolemBrain,
        step : int
    ):
        log_probs   = []
        values      = []
        
        # Get all logprob
        for id, golem in enumerate(self.golems):
            
            if self.state_cache[step][id] == None:
                # No cache means not computation means no grad
                log_probs.append(None)
                values.append(None)
                continue
            
            # TODO: if golem is dead, I put a Do Nothing in the queue and None in log_prob/values
            _, log_prob, critic = golem.sample_actions(brain, from_cache=self.state_cache[step][id])
            log_probs.append(log_prob)
            values.append(critic)
            
        return log_probs, values
    
    def cache_states(self):
        state_cache = [golem.cache for golem in self.golems]
        self.state_cache.append(state_cache)
    
    def step(
        self
    ):
        # Apply a step to the universe
        return self.env.step()
    
    def compute(
        self,
        brain : GolemBrain
    ):
        # Add caching stat to golems
        for golem in self.golems:
            golem.state_caching = True
        
        all_log_probs = []
        all_values = []
        all_rewards = []
        
        for step in range(self.config.n_steps):
            # Start a trajectory sampling and save the old_log_probs/cache the info in the brain
            t_actions, t_logprobs, t_values = self.sample(brain)
            self.cache_states()

            # Queuing all golem commands
            # Add check on alive or not
            for golem_id, actions in enumerate(t_actions):
                for action in actions:
                    if action == "Do Nothing":
                        continue
                    self.players[golem_id].add_cmd(action)

            _, rewards, _ = self.env.step()
            # Fork + ok, spawn new golem based on the one that forked
            for golem_id in range(len(t_actions)):
                all_res = []
                while (response := self.players[golem_id].get_res()):
                    all_res.append(response)
                    if self.golems[golem_id].running_actions[0] == "Fork" and response == "ok":
                        self.new_golem(
                            self.golems[golem_id].pos,
                            self.golems[golem_id].team_index
                        )

                # Send results of all commands to Golems
                self.golems[golem_id].read_player_queue(all_res)
                
            all_log_probs.append(t_logprobs)
            all_values.append(t_values)
            all_rewards.append(rewards)
            
        for golem in self.golems:
            golem.state_caching = False
            golem.cache = None
               
        # Should return log_prob, rewards, values 
        return all_log_probs, all_rewards, all_values
        
    def recompute(
        self,
        brain : GolemBrain
    ):
        
        all_probs = []
        all_values = []
        
        # Recompute the log_prob/values based on the cache
        for step in range(self.config.n_steps):
            probs, values = self.sample_from_cache(brain, step)
            all_probs.append(probs)
            all_values.append(values)
            
        # Should return log_prob, values
        return all_probs, all_values

class AlgorithmConfig:
    
    def __init__(
        self,
        optimizer : type[optim.Optimizer] = optim.Adam,
        optimizer_params : dict[str] = {"lr" : 1e-4}
    ):
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

class PPOConfig(AlgorithmConfig):
    
    def __init__(
        self,
        delta : float = 0.9,
        epsilon : float = 0.1
    ):
        super().__init__()
        
        self.delta = delta
        self.epsilon = epsilon

class Algorithm:
    
    def __init__(
        self,
        brain : GolemBrain,
        config : AlgorithmConfig
    ):
        
        self.properties = {
            # (Nb of steps, Nb of projections, Nb of players, Nb of actions (default 1))
            # If last dim is None, then that means this step should be taken into account
            "old_log_probs" : [],
            "new_log_probs" : [],
            "rewards" : [],
            "values" : []
        }
        self.optimizer = config.optimizer(brain.parameters(), **config.optimizer_params)
    
    # @property
    def register_value(self, property : str, value : list):
        assert property in self.properties.keys(), f"Tried to register unknown property {property}. Availble properties are {self.properties.keys()}"
        self.properties[property] = value
    
    def optimize(self):
        raise NotImplementedError(f"The optimize method is not implemented on the class {self.__class__.__name__}")

# TODO LEFT: Make the GAE & PPO Algorithms
# Call the method to add the reward, value, logprob in the optimize from the 
    
class PPO(Algorithm):
    
    def __init__(
        self,
        brain : GolemBrain,
        config : PPOConfig
    ):
        super().__init__(brain, config)
    
    def optimize(self):
        # Decide on the Future algorithm
        self.optimizer.zero_grad()
        
        # Calculate advantage using GAE
        
        # Calculate the CLIP ppo objective
        # backward on clip ppo objective
        # calculate critic error
        # backpropagate
        
class Elixir:
    
    # Trainer
    
    def __init__(
        self,
        env : Environment,
        elixir_config : ElixirConfig,
        algorithm : Algorithm,
        pretrained_brain : str | None = None
    ):  
        assert (pretrained_brain == None or os.path.exists(pretrained_brain)), f"Pretrained brain path {pretrained_brain} given does not exist"
        
        # TODO: Implement the pretrained_brain loading
        
        self.algorithm = algorithm
        self.brain = GolemBrain()
        self.env : Environment = env
        self.trajectories = [
            Trajectory(env=env, elixir_config=elixir_config) for _ in range(elixir_config.n_trajectories)
        ]
        self.elixir_config = elixir_config

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
        epochs : int = 25
    ):
        for epoch in range(epochs):
            
            old_log_probs = []
            rewards = []
            
            for id, trajectory in enumerate(self.trajectories):
                log_probs, reward, _ = trajectory.compute(self.brain)
                old_log_probs.append(log_probs)
                rewards.append(reward)
                
            self.algorithm.register_value("old_log_probs", old_log_probs)
            self.algorithm.register_value("rewards", rewards)
            
            for k in range(self.elixir_config.repeat_step):
                new_log_probs = []
                values = []
                for id, trajectory in enumerate(self.trajectories):
                    new_log_prob, value = trajectory.recompute(self.brain)
                    new_log_probs.append(new_log_prob)
                    values.append(value)
                    
                self.algorithm.register_value("new_log_probs", new_log_probs)
                self.algorithm.register_value("values", values)
                
                self.algorithm.optimize()
            
            # TODO: Go over multiple step in the trajectories
            
            # Make a copy of each env+players when 
            # Sample All Golems from all env
            # for id, trajectory in enumerate(self.trajectories):
            #     t_actions, t_logprobs, t_values = trajectory.sample(self.brain)
                
            #     # Queuing all golem commands
            #     # Add check on alive or not
            #     for golem_id, actions in enumerate(t_actions):
            #         for action in actions:
            #             if action == "Do Nothing":
            #                 continue
            #             trajectory.players[golem_id].add_cmd(action)
                    
            #     _, rewards, _ = trajectory.step()
            #     # Fork + ok, spawn new golem based on the one that forked
            #     for golem_id in range(len(t_actions)):

            #         all_res = []
            #         while (response := trajectory.players[golem_id].get_res()):
            #             all_res.append(response)
            #             if trajectory.golems[golem_id].running_actions[0] == "Fork" and response == "ok":
            #                 trajectory.new_golem(
            #                     trajectory.golems[golem_id].pos,
            #                     trajectory.golems[golem_id].team_index
            #                 )
                
            #         # Send results of all commands to Golems
            #         trajectory.golems[golem_id].read_player_queue(all_res)
                    
            # TODO: Call algoritm.optimize()
            self.algorithm.optimize()
            
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