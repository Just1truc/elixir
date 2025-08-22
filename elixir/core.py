import os
import torch
import torch as t

from torch import optim, nn

from elixir.brain import GolemBrain, BrainConfig
from elixir.golem import Golem

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
            ) for player in self.env.players
        ]
        self.config = elixir_config
        self.state_cache = []
        print("Noen", self.env.players == None)
        
    @property
    def players(self) -> list[Player]:
        return self.env.players
        
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
                    print(self.players == None)
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
    
# class PPO(Algorithm):
    
#     def __init__(
#         self,
#         brain : GolemBrain,
#         config : PPOConfig
#     ):
#         super().__init__(brain, config)
    
#     def optimize(self):
#         # Decide on the Future algorithm
#         self.optimizer.zero_grad()
        
#         # Calculate advantage using GAE
        
#         # Calculate the CLIP ppo objective
#         # backward on clip ppo objective
#         # calculate critic error
#         # backpropagate
        

class PPO(Algorithm):

    # ----- helpers -----
    @staticmethod
    def _to_tensor(x, device, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    @staticmethod
    def _stack_with_mask(seq, device, allow_tplus1: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        seq: list (len T or T+1) where elements can be Tensor/array/number or None.
        Returns:
          stacked: tensor [T or T+1, *shape] with zeros at None positions
          mask:    bool   [T or T+1, *shape], True where valid
        """
        L = len(seq)
        if L == 0:
            return torch.zeros((0, 1), device=device), torch.zeros((0, 1), dtype=torch.bool, device=device)
        if (not allow_tplus1) and L > 0:
            # nothing special, just proceed
            pass

        tmpl = None
        for item in seq:
            if item is not None:
                tmpl = item
                break
        if tmpl is None:
            return torch.zeros((L, 1), device=device), torch.zeros((L, 1), dtype=torch.bool, device=device)

        filled, mask = [], []
        for item in seq:
            if item is None:
                t = torch.zeros_like(tmpl, device=device)
                m = torch.zeros_like(tmpl, device=device, dtype=torch.bool)
            else:
                t = item if isinstance(item, torch.Tensor) else torch.as_tensor(item, device=device, dtype=getattr(tmpl, "dtype", torch.float32))
                t = t.to(device=device, dtype=getattr(tmpl, "dtype", torch.float32))
                m = torch.ones_like(t, device=device, dtype=torch.bool)
            filled.append(t)
            mask.append(m)
        return torch.stack(filled, dim=0), torch.stack(mask, dim=0)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
        mask = mask.to(dtype=x.dtype)
        num = (x * mask).sum()
        den = mask.sum().clamp_min(eps)
        return num / den

    def _compute_masked_gae(
        self,
        rewards: torch.Tensor,            # [T, ...]
        values: torch.Tensor,             # [T, ...] or [T+1, ...], zeros where invalid
        vmask: torch.Tensor,              # bool [T, ...] or [T+1, ...]
        gamma: float,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mask-aware GAE:
        - Only uses value predictions where vmask is True.
        - If next value is invalid or done==1, bootstrapping stops (like terminal).
        - Returns:
            advantages [T, ...], returns [T, ...], valid mask for value terms at t (vmask_t)
        """
        T = rewards.shape[0]
        assert values.shape[0] in (T, T + 1)
        if dones is None:
            dones = torch.zeros_like(rewards)

        if values.shape[0] == T + 1:
            v_t    = values[:-1]
            v_t1   = values[1:]
            vm_t   = vmask[:-1]
            vm_t1  = vmask[1:]
        else:
            # fabricate next values by shifting; next validity is that of shifted current
            v_t  = values
            v_t1 = torch.cat([values[1:], torch.zeros_like(values[-1:])], dim=0)
            vm_t = vmask
            vm_t1 = torch.cat([vmask[1:], torch.zeros_like(vmask[-1:], dtype=torch.bool)], dim=0)

        # delta_t only meaningful where current value is valid
        # when next is invalid or done, we zero out its contribution (stop bootstrap)
        next_ok = (~dones.bool()) & vm_t1
        deltas = rewards + gamma * v_t1 * next_ok.to(v_t1.dtype) - v_t
        deltas = torch.where(vm_t, deltas, torch.zeros_like(deltas))

        adv = torch.zeros_like(deltas)
        gae = torch.zeros_like(deltas[-1])
        for t in reversed(range(T)):
            nonterminal = (1.0 - dones[t]) * next_ok[t].to(deltas.dtype)
            gae = deltas[t] + gamma * lam * nonterminal * gae
            # Only store advantage where current value is valid
            adv[t] = torch.where(vm_t[t], gae, torch.zeros_like(gae))

        returns = adv + v_t  # returns only meaningful where vm_t
        return adv, returns, vm_t

    # ----- main -----
    def optimize(self):
        cfg: PPOConfig = self.config
        self.optimizer.zero_grad()

        try:
            device = next(self.brain.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        # Core tensors
        rewards = self._to_tensor(self.properties["rewards"], device)
        dones = None
        if "dones" in self.properties and len(self.properties["dones"]) != 0:
            dones = self._to_tensor(self.properties["dones"], device)

        # Log-probs (policy) with mask for forced steps
        old_lp, mask_old = self._stack_with_mask(self.properties["old_log_probs"], device)
        new_lp, mask_new = self._stack_with_mask(self.properties["new_log_probs"], device)
        policy_mask = mask_old & mask_new   # only where both are present

        # Values may also be None (same positions as forced steps)
        values, vmask = self._stack_with_mask(self.properties["values"], device, allow_tplus1=True)

        # Optional entropies (mask them too; otherwise use policy_mask)
        if "entropies" in self.properties and len(self.properties["entropies"]) != 0:
            entropies, ent_mask = self._stack_with_mask(self.properties["entropies"], device)
            entropy_mask = policy_mask & ent_mask
        else:
            entropies, entropy_mask = None, None

        # Shapes / lengths
        T = rewards.shape[0]
        assert old_lp.shape[0] == T and new_lp.shape[0] == T, "log_probs must have length T"
        assert values.shape[0] in (T, T + 1), "values must have length T or T+1"
        if dones is not None:
            assert dones.shape[0] == T, "dones must have length T"

        # ----- Masked GAE (value None breaks bootstrap chain) -----
        advantages, returns, vmask_t = self._compute_masked_gae( # LA
            rewards=rewards,
            values=values,
            vmask=vmask,
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
            dones=dones
        )

        # Normalize advantages across valid entries
        if cfg.normalize_advantage:
            valid_adv = advantages[vmask_t]
            if valid_adv.numel() > 0:
                mean = valid_adv.mean()
                std = valid_adv.std(unbiased=False).clamp_min(cfg.eps)
                advantages = torch.where(vmask_t, (advantages - mean) / std, torch.zeros_like(advantages))

        # ----- PPO clipped objective (masked by policy_mask) -----
        # Note: policy_mask might have extra dims (e.g., multi-agent); broadcast works.
        if policy_mask.sum().item() == 0:
            # Nothing to optimize this pass; keep API predictable
            dummy = rewards.sum() * 0.0
            dummy.backward()
            self.optimizer.step()
            return {
                "loss_total": 0.0,
                "loss_policy": 0.0,
                "loss_value": 0.0,
                "entropy": 0.0,
                "adv_mean": float(advantages[vmask_t].mean().detach().cpu()) if vmask_t.any() else 0.0,
                "adv_std": float(advantages[vmask_t].std(unbiased=False).detach().cpu()) if vmask_t.any() else 0.0,
                "ratio_mean": 1.0,
                "valid_frac_actor": 0.0,
                "valid_frac_critic": float(vmask_t.float().mean().detach().cpu()) if vmask_t.numel() else 0.0,
                "skipped_all": True,
            }

        ratios = torch.exp(new_lp - old_lp)
        unclipped = ratios * advantages
        clipped   = torch.clamp(ratios, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * advantages
        policy_loss = -self._masked_mean(torch.minimum(unclipped, clipped), policy_mask, cfg.eps)

        # ----- Value loss (masked by vmask_t) -----
        v_pred_t = values[:-1] if values.shape[0] == T + 1 else values
        td = (v_pred_t - returns)
        value_loss = 0.5 * self._masked_mean(td * td, vmask_t, cfg.eps)

        # ----- Entropy bonus (masked) -----
        if entropies is not None and entropy_mask is not None and entropy_mask.any():
            entropy_bonus = self._masked_mean(entropies, entropy_mask, cfg.eps)
        else:
            entropy_bonus = torch.tensor(0.0, device=device)

        loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_bonus
        loss.backward()

        if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.brain.parameters(), cfg.max_grad_norm)

        self.optimizer.step()

        with torch.no_grad():
            ratio_mean = self._masked_mean(ratios, policy_mask, cfg.eps).item() if policy_mask.any() else 1.0

        return {
            "loss_total": float(loss.detach().cpu()),
            "loss_policy": float(policy_loss.detach().cpu()),
            "loss_value": float(value_loss.detach().cpu()),
            "entropy": float(entropy_bonus.detach().cpu()) if isinstance(entropy_bonus, torch.Tensor) else float(entropy_bonus),
            "adv_mean": float(advantages[vmask_t].mean().detach().cpu()) if vmask_t.any() else 0.0,
            "adv_std": float(advantages[vmask_t].std(unbiased=False).detach().cpu()) if vmask_t.any() else 0.0,
            "ratio_mean": ratio_mean,
            "valid_frac_actor": float(policy_mask.float().mean().detach().cpu()),
            "valid_frac_critic": float(vmask_t.float().mean().detach().cpu()),
        }
        
class Elixir:
    
    # Trainer
    
    def __init__(
        self,
        env : Environment,
        elixir_config : ElixirConfig,
        algorithm : type[Algorithm],
        algorithm_config : AlgorithmConfig,
        brain_config : BrainConfig,
        pretrained_brain : str | None = None
    ):  
        assert (pretrained_brain == None or os.path.exists(pretrained_brain)), f"Pretrained brain path {pretrained_brain} given does not exist"
        
        # TODO: Implement the pretrained_brain loading
        
        self.brain = GolemBrain(config=brain_config, entry_dim=len(Golem.resources))
        if pretrained_brain != None:
            self.brain.load_state_dict(torch.load(pretrained_brain))
        
        self.algorithm      = algorithm(self.brain, algorithm_config)
        self.trajectories   = [
            Trajectory(env=env, elixir_config=elixir_config) for _ in range(elixir_config.n_trajectories)
        ]
        self.elixir_config  = elixir_config
        
        self.env : Environment = env

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
        log_step : int = 2
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
                
                metrics = self.algorithm.optimize()
                if (k % log_step) == 0:
                    print(f"Epoch [{epoch}], step [{k}], total loss {metrics['total_loss']}")
            
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