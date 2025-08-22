"""
reward_constant.py

Defines reward-related constants and configuration values for the Zappy environment simulation.
"""
class RewardConstants:
    # Base rewards (applies to all environments)
    STEP_PENALTY_FACTOR = -0.01
    DEATH_PENALTY = -100
    LEVEL_PROGRESSION_BASE = 10
    FOOD_COLLECTION_BASE = 1.0
    STARVATION_PENALTY = -0.5
    
    # Alone environment
    ALONE_LEVEL_BONUS = 5
    
    # Alone with eggs
    FORK_REWARD = 2
    
    # Group environment
    BROADCAST_REWARD = 0.5
    TEAMMATE_PROXIMITY_REWARD = 0.1
    
    # Random group
    HELPING_REWARD = 0.2
    
    # Survival environment
    CRITICAL_FOOD_REWARD = 1.0
    
    # Incantation environment
    INCANTATION_START_REWARD = 5
    RESOURCE_READINESS_REWARD = 2
    
    # Close to incantation
    RESOURCE_COLLECTION_REWARD = 1.5
    RESOURCE_DELIVERY_REWARD = 1.0
    
    # Strict incantation environment
    MOVEMENT_DURING_INCANTATION_PENALTY = -50
    MOVEMENT_COMMAND_PENALTY = -30
    ABANDONING_INCANTATION_PENALTY = -100
    INCANTATION_DISCIPLINE_REWARD = 0.5