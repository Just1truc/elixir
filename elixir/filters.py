import torch as t

from typing import Callable, Tuple
from elixir.auto_grid import grid_value

from matplotlib import pyplot as plt

def gen_filter_grid(w : int, h : int):

    full_w = w * 2
    full_y = h * 2
    
    filters = t.empty(4, h * 2, w * 2)
    
    for y in range(full_y):
        for x in range(full_w):
            for i in range(1, 5):
                filters[i - 1][y][x] = grid_value(x, y, i % 4, h, w, w, h)
    
    return filters

class Memoization:
    
    memoized : dict[str] = {}
    
    def safe_get(self, key : str):
        return Memoization.memoized.get(key, None)
    
    def __getitem__(self, key : str):
        return self.safe_get(key)
    
    def register(self, key : str, o):
        Memoization.memoized.__setitem__(key, o)
        
    def conditional_register(self, key : str, function : Callable, args : Tuple):
        retrieved = self.safe_get(key)
        if retrieved != None:
            return retrieved
        else:
            value = function(*args)
            self.register(key,value)
            return value

class PosFilter(Memoization):
    
    def __init__(
        self,
        h : int,
        w : int,
        memoization : bool = True
    ):
        self.filters = self.conditional_register("filters", gen_filter_grid, (w, h)) if memoization else gen_filter_grid(w, h)
        self.w = w
        self.h = h
        
    def get_view(
        self,
        pos : t.Tensor,
        orientation : int
    ):   
        return self.filters[orientation][self.h - pos[0] - 1:2 * self.h - pos[0] - 1, self.w - pos[1] - 1: 2 * self.w - pos[1] - 1]
