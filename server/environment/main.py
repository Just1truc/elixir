"""
main.py

Entry point for running and testing the Zappy environment simulation. Handles environment setup and execution of test scenarios.
"""
from server.environment.category import Category
from server.environment.environment import Environment

from server import constant

# Create survival environment
env: Environment = Category["survival"](nb_players=3, food_amount=2, size=8)

# Reset environment and get initial state
state = env.reset()

for step in range(1000):
    # AI makes decisions here
    for id, player in enumerate(state["players"]):
        if player.inventory.get(constant.Resource.FOOD.value, 0) < 2:
            player.add_cmd("Take food")
        else:
            player.add_cmd("Forward")
    
    # Advance simulation
    state, rewards, terminated = env.step()
    
    # Print rewards for debugging
    print(f"Step {step}: Rewards = {rewards}")
    for i, _ in enumerate(env.players):
        print(env.get_player_state(i))

    # Check termination
    if terminated:
        print("Simulation ended!")
        break