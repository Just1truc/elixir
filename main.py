from server.environment import Category, Environment
from elixir import Elixir, PPO, PPOConfig, ElixirConfig, BrainConfig

env: Environment = Category["survival"](nb_players=3, food_amount=2, size=8)

algorithm_config = PPOConfig()
elixir_config = ElixirConfig(repeat_step=10)
brain_config = BrainConfig(
    d_model=64,
    n_head=2,
    n_layers=3,
    dim_feedforward=128
)

elixir = Elixir(
    env              = env,
    elixir_config    = elixir_config,
    algorithm        = PPO,
    algorithm_config = algorithm_config,
    brain_config     = brain_config
)
elixir.train()