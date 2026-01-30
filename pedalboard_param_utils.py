from pedalboard import Pedalboard
from pedalboard import Chorus, Distortion, Phaser # guitar-style effects
from pedalboard import Compressor # dynamic range effects
from pedalboard import Reverb # spacial effects 

import random 

pedal_dict = {
    "compressor": {
        "pedal": Compressor, 
        "params": {
            "threshold_db": [i in range(-60,0,10)],
            "ratio": [1.5, 2, 4, 10],
            "attack_ms": [10, 50, 70, 100],
            "release_ms": [50, 100, 150, 300, 500]
        },
        "dropout": 0
    },
    "distortion": {
        "pedal": Distortion,
        "params": {
            "drive_db": [12, 18, 24]
        },
        "dropout": 0.7
    },
    "chorus": {
        "pedal": Chorus, 
        "params": {
            "rate_hz" : [0.5, 0.8, 1],
            "centre_delay_ms" : [i for i in range(5, 35, 5)],
            "depth" : [0.05, 0.15, 0.25],
            "feedback" : [0.1, 0.2, 0.4],
            "mix" : [0.5, 0.7, 1]
        },
        "dropout": 0.7
    },
    "phaser": {
        "pedal": Phaser, 
        "params": {
            "rate_hz": [0.2, 0.5, 1, 3],
            "depth": [0.2, 0.4, 0.7, 1],
            "centre_frequency_hz": [300, 600, 800, 1500, 2000],
            "feedback": [0, 0.4, 0.6], 
            "mix": [0.5]
        },
        "dropout":0.7
    },
    "reverb": {
        "pedal": Reverb,
        "params": {
            "room_size": [0.1, 0.3, 0.5, 0.7, 1],
            "damping": [0, 0.5, 1],
            "wet_level": [0.5, 0.7, 1],
            "dry_level": [0.5, 0.7, 1],
            "width": [0.5, 0.7, 1], 
        },
        "dropout": 0.7
    }
}

def tokenize_pedal_dict(pedal_dict): 
    tokens = []
    for pedal in pedal_dict.items(): 
        tokens.append(pedal[0])
        params = pedal[1]['params']
        for param, choices in params.items(): 
            tokens.extend([pedal[0] + "_" + param + "_" + str(i) for i in choices])
    return tokens
    
def get_pedal_board(pedal_dict, shuffle=True):
    board = []
    board_string = []

    for name, param_dict in pedal_dict.items(): 
        pedal_class = param_dict['pedal']
        params = param_dict['params']
        dropout = param_dict['dropout']
        
        args = {}
        for param, bins in params.items(): 
            args[param] = random.choice(bins)
        
        if random.random() < dropout:  
            board_string.append(" ".join([name + "_" + param + "_" + str(choice) for param, choice in args.items()]))
            board.append(pedal_class(**args))
    if shuffle: 
        combined = list(zip(board, board_string))
        random.shuffle(combined)
        board, board_string = zip(*combined)

    board = Pedalboard(board)
        
    return board, list(board_string)