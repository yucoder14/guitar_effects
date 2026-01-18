from pedalboard import Plugin
from pedalboard import Pedalboard
from pedalboard import Chorus, Distortion, Phaser, Clipping # guitar-style effects
from pedalboard import Compressor, Gain, Limiter # dynamic range effects
from pedalboard import Delay, Reverb # spacial effects 
from typing import List, Tuple

import random 

DEFAULT_PEDAL_PROBS = [ 
    (Chorus(), 0.3),
    (Distortion(), 0.3),
    (Phaser(), 0.3),
    (Clipping(), 0.3),
    (Compressor(), 0.3),
    (Gain(), 0.3),
    (Limiter(), 0.3),
    (Delay(), 0.3),
    (Reverb(), 0.3)
]

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
    Given a pedalboard, convert it space separated name strings
    """
    names = []
    for pedal in pedalboard:
        names.append(pedal.__class__.__name__.lower())
    return " ".join(names) if len(names) > 0 else "clean" 
