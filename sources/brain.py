from torch import nn
import torch as t

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
        
        # Prediction Heads broadcast, fork, incant, push (eject), pickup (7) and set down (7)
        self.action_heads   = nn.Linear(d_model, 4+7+7)
        self.heatmap_head   = nn.Linear(d_model, 1)


    def forward(
        self,
        x : t.Tensor,
        position : t.Tensor
    ):
        """
        Args:
            x (t.Tensor): (N, N, 15)
            returns (N * N + 6)
        """
        N, _, D = x.shape 
        out = self.encoder(x.reshape(N * N, D))
        
        return t.concat([self.heatmap_head(out), self.action_heads(out[*position])])
    

