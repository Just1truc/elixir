from torch import nn
import torch as t

class BrainConfig:
    
    def __init__(
        self,
        d_model : int = 64,
        n_head : int = 2,
        n_layers : int = 2,
        dim_feedforward : int = 128
    ):
        
        self.d_model = d_model
        
        self.n_head     = n_head
        self.n_layers   = n_layers
        
        self.dim_feedforward    = dim_feedforward


class GolemBrain(nn.Module):
    
    def __init__(
        self,
        config : BrainConfig,
        entry_dim : int = 22
    ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model = config.d_model,
            nhead  = config.n_head,
            dim_feedforward = config.dim_feedforward,
            dropout = 0,
            batch_first = True
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=config.n_layers
        )
        
        # Prediction Heads broadcast, fork, incant, push (eject), pickup (7) and set down (7)
        self.action_heads   = nn.Linear(config.d_model, 4+7+7)
        self.heatmap_head   = nn.Linear(config.d_model, 1)
        self.map_critic     = nn.Linear(config.d_model, 1)

        self.in_proj = nn.Linear(entry_dim, config.d_model)

    def forward(
        self,
        x : t.Tensor,
        position : t.Tensor
    ):
        """
        Args:
            x (t.Tensor): (N, N, 15)
            returns (N * N + 18), (1,) 
        """
        N, _, D = x.shape 
        out = self.encoder(x.reshape(1, N * N, D)).reshape(N, N, D)
        
        # print(out.shape, position)
        # print(self.heatmap_head(out).shape, self.action_heads(out[*position]).shape)
        action = t.concat([self.heatmap_head(out).flatten(), self.action_heads(out[*position])])
        critic = self.map_critic(out.mean(0))
    
        return action, critic
