from enum import Enum


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Resource(Enum):
    FOOD = "food"
    LINEMATE = "linemate"
    DERAUMERE = "deraumere"
    SIBUR = "sibur"
    MENDIANE = "mendiane"
    PHIRAS = "phiras"
    THYSTAME = "thystame"


RESOURCE_ENUM = {
    "food": Resource.FOOD,
    "linemate": Resource.LINEMATE,
    "deraumere": Resource.DERAUMERE,
    "sibur": Resource.SIBUR,
    "mendiane": Resource.MENDIANE,
    "phiras": Resource.PHIRAS,
    "thystame": Resource.THYSTAME,
}

RESOURCE_DENSITY = {
    Resource.FOOD: 0.5,
    Resource.LINEMATE: 0.3,
    Resource.DERAUMERE: 0.15,
    Resource.SIBUR: 0.1,
    Resource.MENDIANE: 0.1,
    Resource.PHIRAS: 0.08,
    Resource.THYSTAME: 0.05,
}


class ElevationRequirement:
    requirements = {
        1: (1, {Resource.LINEMATE: 1}),
        2: (2, {Resource.LINEMATE: 1, Resource.DERAUMERE: 1, Resource.SIBUR: 1}),
        3: (2, {Resource.LINEMATE: 2, Resource.SIBUR: 1, Resource.PHIRAS: 2}),
        4: (4, {Resource.LINEMATE: 1, Resource.DERAUMERE: 1, Resource.SIBUR: 2, Resource.PHIRAS: 1}),
        5: (4, {Resource.LINEMATE: 1, Resource.DERAUMERE: 2, Resource.SIBUR: 1, Resource.MENDIANE: 3}),
        6: (6, {Resource.LINEMATE: 1, Resource.DERAUMERE: 2, Resource.SIBUR: 3, Resource.PHIRAS: 1}),
        7: (6, {Resource.LINEMATE: 2, Resource.DERAUMERE: 2, Resource.SIBUR: 2,
                Resource.MENDIANE: 2, Resource.PHIRAS: 2, Resource.THYSTAME: 1})
    }
