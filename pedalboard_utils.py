from pedalboard import Plugin
from pedalboard import Pedalboard
from pedalboard import Chorus, Distortion, Phaser, Clipping # guitar-style effects
from pedalboard import Compressor, Gain, Limiter # dynamic range effects
from pedalboard import Delay, Reverb # spacial effects 

from typing import List, Tuple
from enum import Enum

from torch import tensor

import numpy as np
import random 

# skipping delay for now because there's some time variant stuff going on
class Pedal(Enum): 
    CLEAN = 0
    CHORUS = 1
    DISTORTION = 2
    PHASER = 3
    CLIPPING = 4
    COMPRESSOR = 5
    GAIN = 6
    LIMITER = 7 
    REVERB = 8
    
# using defaut parameters for now
DEFAULT_PEDAL_PROBS = [ 
    (Chorus(), 0.3),
    (Distortion(), 0.3),
    (Phaser(), 0.3),
    (Clipping(), 0.3),
    (Compressor(), 0.3),
    (Gain(), 0.3),
    (Limiter(), 0.3),
    (Reverb(), 0.3)
]

def generate_tokens() -> List[str]:
    pass

def get_random_board(pedal_probs: List[Tuple[Plugin, float]]) -> Pedalboard: 
    """
    Given a list of plugins and dropout probability, generate a pedalboard with 
    random order. 
    """
    board = Pedalboard()
    random.shuffle(pedal_probs)
    for pedal, prob in pedal_probs[:-1]:
        if random.random() >= 1 - prob:        
            board.append(pedal)
    return board

def get_board_string(pedalboard: Pedalboard) -> str: 
    """
    Given a pedalboard, convert it into order vector indexed by enum values 
    """
    pedal_nums = [0] * 9
    for index, pedal in enumerate(pedalboard):
        pedal_nums[Pedal[pedal.__class__.__name__.upper()].value] = index + 1
    return tensor(pedal_nums)

def get_pedal_string(order_vec):
    arr = order_vec.numpy() 
    idxs=np.where(arr!=0)
    names = [Pedal(val).name for val in idxs[0]]
    order = arr[idxs]
    names = [pedal for pedal, _ in sorted(list(zip(names,order)), key=lambda tup: tup[-1])] 
    return names if len(names) > 0 else "CLEAN"