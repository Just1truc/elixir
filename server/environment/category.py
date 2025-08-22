from server.environment.alone.env import EnvironmentAlone
from server.environment.alone_eggs.env import EnvironmentAloneWithConnectSlot
from server.environment.group.env import EnvironmentGroup
from server.environment.random_group.env import EnvironmentRandomGroup
from server.environment.survival.env import EnvironmentSurvival
from server.environment.incantation.env import EnvironmentIncantation
from server.environment.close_incantation.env import EnvironmentCloseToIncantation
from server.environment.strict_incantation.env import EnvironmentStrictIncantation

# Environment category mapping
Category = {
    "alone": EnvironmentAlone,
    "alone_eggs": EnvironmentAloneWithConnectSlot,
    "group": EnvironmentGroup,
    "random_group": EnvironmentRandomGroup,
    "survival": EnvironmentSurvival,
    "incantation": EnvironmentIncantation,
    "close_incantation": EnvironmentCloseToIncantation,
    "strict_incantation": EnvironmentStrictIncantation,
}
