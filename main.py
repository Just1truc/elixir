from server.environment import Category, Environment
from server.constant import Resource

# Create survival environment
env: Environment = Category["survival"](nb_players=3, food_amount=2, size=8)

# Reset environment and get initial state
state = env.reset()

for step in range(1000):
    # AI makes decisions here
    # for id, player in state["players"].items():
    #     if player.inventory.get(Resource.FOOD.value, 0) < 2:
    #         player.add_cmd("Take food")
    #     else:
    #         player.add_cmd("Forward")
    
    # Advance simulation
    state, rewards, terminated = env.step()
    
    # Print rewards for debugging
    print(f"Step {step}: Rewards = {rewards}")
    for i in env.players:
        print(env.get_player_state(i))

    # Check termination
    if terminated:
        print("Simulation ended!")
        break