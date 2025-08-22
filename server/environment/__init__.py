
# Category mapping and environment imports
from .environment import Environment
from .alone.env import EnvironmentAlone
from .alone_eggs.env import EnvironmentAloneWithConnectSlot
from .group.env import EnvironmentGroup
from .random_group.env import EnvironmentRandomGroup
from .survival.env import EnvironmentSurvival
from .incantation.env import EnvironmentIncantation
from .close_incantation.env import EnvironmentCloseToIncantation
from .strict_incantation.env import EnvironmentStrictIncantation

# Category mapping for easy environment selection
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
