import torch

from torch import optim
from torch import nn

from elixir.brain import GolemBrain

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